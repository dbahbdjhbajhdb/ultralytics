# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Standard YOLO detection loss plus aligned EMA feature targets for occluded pairs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.data.occlusion_pairing import validate_pair_metadata
from ultralytics.utils.loss import v8DetectionLoss


def sample_pair_regions(feature, boxes, batch_idx, roi_size=3):
    """Sample normalized xywh regions as spatial N,C,R,R tensors, retaining GT order.

    Teacher and student must use the same transformed boxes and this same grid.
    Sampling a tiny region at several bin centers does not create additional
    image detail. This helper uses bilinear grid sampling, not ROIAlign.
    """
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or min(feature.shape) < 1:
        raise ValueError("Pair features must be a nonempty floating-point BCHW tensor")
    if not feature.is_floating_point():
        raise ValueError("Pair features must be floating point")
    if isinstance(roi_size, bool) or not isinstance(roi_size, int) or roi_size < 1:
        raise ValueError("Pair ROI grid size must be a positive integer")
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Pair ROI boxes must have shape (N, 4) in normalized xywh")
    if not isinstance(batch_idx, torch.Tensor) or batch_idx.ndim != 1 or len(batch_idx) != len(boxes):
        raise ValueError("Pair ROI boxes and image indices must align")
    if boxes.is_complex() or boxes.dtype == torch.bool or batch_idx.is_complex() or batch_idx.dtype == torch.bool:
        raise ValueError("Pair ROI boxes and image indices must be real numeric values")
    boxes = boxes.to(device=feature.device, dtype=torch.float32)
    batch_idx = batch_idx.to(device=feature.device)
    if not torch.isfinite(boxes).all() or not torch.isfinite(batch_idx).all():
        raise ValueError("Pair ROI boxes and image indices must be finite")
    if ((boxes < -1e-6) | (boxes > 1 + 1e-6)).any() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Pair ROI boxes must be normalized with positive width and height")
    if (batch_idx != batch_idx.floor()).any() or ((batch_idx < 0) | (batch_idx >= feature.shape[0])).any():
        raise ValueError("Pair ROI image indices must be integers within the feature batch")
    batch_idx = batch_idx.long()
    if not len(boxes):
        return feature.new_empty((0, feature.shape[1], roi_size, roi_size), dtype=torch.float32)
    fractions = (torch.arange(roi_size, device=feature.device, dtype=torch.float32) + 0.5) / roi_size - 0.5
    offset_y, offset_x = torch.meshgrid(fractions, fractions, indexing="ij")
    offsets = torch.stack((offset_x, offset_y), dim=-1)
    patches, positions = [], []
    for image_index in range(feature.shape[0]):
        selected = torch.where(batch_idx == image_index)[0]
        if not len(selected):
            continue
        selected_boxes = boxes[selected]
        grid = selected_boxes[:, None, None, :2] + offsets[None] * selected_boxes[:, None, None, 2:]
        grid = (grid * 2 - 1).reshape(1, len(selected) * roi_size, roi_size, 2)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            sampled = F.grid_sample(
                feature[image_index : image_index + 1].float(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        patches.append(sampled.reshape(feature.shape[1], len(selected), roi_size, roi_size).permute(1, 0, 2, 3))
        positions.append(selected)
    order = torch.cat(positions)
    return torch.cat(patches, dim=0)[order.argsort()]


class PairedOcclusionLoss(v8DetectionLoss):
    """Leave BCE/CIoU/DFL/TAL unchanged and add a detached EMA pair feature target.

    ``pair_loss`` is the fourth logged item, averaged over valid ROI locations.
    Standard detection validation does not carry teacher targets, so its fourth
    item is zero. Training diagnostics record the actual pair supervision.
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        self.config = getattr(model, "occlusion_pair_config", {}).copy()
        value = self.config.get("coefficient", 0.05)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Pair coefficient must be numeric")
        self.coefficient = float(value)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("Pair coefficient must be finite and nonnegative")
        self.roi_size = self.config.get("roi_size", 3)
        if isinstance(self.roi_size, bool) or not isinstance(self.roi_size, int) or self.roi_size < 1:
            raise ValueError("Pair ROI grid size must be a positive integer")
        self.set_progress(self.config.get("progress", 0.0))
        self.reset_statistics()

    def set_progress(self, progress):
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise ValueError("Pair schedule progress must be numeric")
        self.progress = float(progress)
        if not math.isfinite(self.progress) or not 0 <= self.progress <= 1:
            raise ValueError("Pair schedule progress must lie in [0, 1]")

    def reset_statistics(self):
        self.statistics = {
            "gt_by_occlusion": [0, 0, 0],
            "unknown_gt": 0,
            "paired_gt": 0,
            "paired_by_occlusion": [0, 0, 0],
            "cosine_sum": 0.0,
            "valid_locations": 0,
            "batch_count": 0,
            "weighted_pair_sum": 0.0,
        }

    def _pair_loss(self, feature, batch):
        boxes, image_indices, _, grades, valid = validate_pair_metadata(batch, nc=self.nc)
        self.statistics["batch_count"] += 1
        self.statistics["unknown_gt"] += int((~valid).sum())
        for degree in range(3):
            self.statistics["gt_by_occlusion"][degree] += int((valid & (grades == degree)).sum())
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or min(feature.shape) < 1:
            raise ValueError("Pair training predictions must expose a BCHW P3 feature tensor")
        if feature.shape[0] != batch["img"].shape[0] or feature.device != batch["img"].device:
            raise ValueError("Pair feature and image batches must align")
        zero = feature.reshape(-1)[0].float() * 0
        indices = batch.get("pair_indices")
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
            raise ValueError("Paired training batches must contain one-dimensional pair_indices")
        if (
            indices.is_complex()
            or indices.dtype == torch.bool
            or not torch.isfinite(indices).all()
            or (indices != indices.floor()).any()
            or ((indices < 0) | (indices >= len(boxes))).any()
        ):
            raise ValueError("Pair indices must refer to existing global GT rows")
        indices = indices.to(device=feature.device, dtype=torch.long)
        if len(indices) != len(indices.unique()):
            raise ValueError("Pair indices must not contain duplicate GT rows")
        if not len(indices) or self.coefficient == 0 or self.progress == 0:
            return zero
        if (~valid[indices]).any() or ((grades[indices] != 0) & (grades[indices] != 1)).any():
            raise ValueError("Pair targets must be known grade 0/1 GTs; severe or unknown targets cannot be erased")
        targets = batch.get("pair_targets")
        expected_shape = (len(indices), feature.shape[1], self.roi_size, self.roi_size)
        if not isinstance(targets, torch.Tensor) or tuple(targets.shape) != expected_shape:
            raise ValueError(f"Detached EMA pair_targets must have shape {expected_shape}")
        if not targets.is_floating_point() or targets.device != feature.device:
            raise ValueError("EMA pair targets must be floating-point tensors on the feature device")
        student = sample_pair_regions(feature, boxes[indices], image_indices[indices], roi_size=self.roi_size)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            targets = targets.detach().float()
            valid_locations = torch.isfinite(targets).all(dim=1) & (targets.norm(dim=1) > 1e-6)
            location_count = int(valid_locations.sum())
            if not location_count:
                return zero
            # Drop invalid teacher locations before normalization, so NaN/Inf
            # teacher values cannot leak through a multiply-by-zero mask.
            teacher_vectors = targets.permute(0, 2, 3, 1)[valid_locations]
            student_vectors = student.permute(0, 2, 3, 1)[valid_locations]
            if not torch.isfinite(student_vectors).all():
                raise ValueError("Student pair features contain nonfinite values at supervised locations")
            teacher_vectors = F.normalize(teacher_vectors, dim=1, eps=1e-6)
            student_vectors = F.normalize(student_vectors.float(), dim=1, eps=1e-6)
            distances = (1 - (student_vectors * teacher_vectors).sum(dim=1)).clamp(0, 2)
            raw_loss = distances.mean()
        self.statistics["paired_gt"] += len(indices)
        for degree in range(3):
            self.statistics["paired_by_occlusion"][degree] += int((grades[indices] == degree).sum())
        self.statistics["cosine_sum"] += float(distances.detach().sum())
        self.statistics["valid_locations"] += location_count
        self.statistics["weighted_pair_sum"] += float(raw_loss.detach()) * self.coefficient * self.progress
        return raw_loss

    def __call__(self, preds, batch):
        if isinstance(preds, dict):
            if "detect" not in preds or "pair_features" not in preds:
                raise ValueError("Pair training predictions require detect outputs and pair_features")
            feature = preds["pair_features"]
            if isinstance(feature, (tuple, list)):
                if len(feature) != 1:
                    raise ValueError("Pair supervision uses exactly one aligned P3 feature level")
                feature = feature[0]
            raw_pair_loss = self._pair_loss(feature, batch)
            detection_loss, detection_items = super().__call__(preds["detect"], batch)
            pair_item = raw_pair_loss * self.coefficient * self.progress
            pair_loss = pair_item * feature.shape[0]
        else:
            detection_loss, detection_items = super().__call__(preds, batch)
            pair_loss = detection_loss.sum() * 0
            pair_item = pair_loss.detach()
        return (
            torch.cat((detection_loss, pair_loss.reshape(1))),
            torch.cat((detection_items, pair_item.detach().reshape(1))),
        )
