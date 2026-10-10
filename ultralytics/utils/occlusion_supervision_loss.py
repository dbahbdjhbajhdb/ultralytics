# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unchanged detection loss plus direct, training-only GT occlusion classification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.occlusion_aux import OcclusionAuxDetect
from ultralytics.utils.loss import v8DetectionLoss


class OcclusionSupervisionLoss(v8DetectionLoss):
    """Supervise each valid GT region directly, independently of TAL matches.

    Detection BCE, CIoU and DFL retain the base implementation and normalization.
    The fourth logged loss is coefficient * progress * mean GT cross entropy.
    Standard eval predictions do not contain auxiliary features, so that fourth
    entry is zero during normal detection validation; training diagnostics are
    the source for auxiliary GT accuracy and raw cross entropy.
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        self.head = model.model[-1]
        if not isinstance(self.head, OcclusionAuxDetect):
            raise TypeError("Direct occlusion supervision requires OcclusionAuxDetect")
        self.config = getattr(model, "occlusion_aux_config", {}).copy()
        self.coefficient = float(self.config.get("coefficient", 0.1))
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("Auxiliary occlusion coefficient must be finite and nonnegative")
        self.set_progress(self.config.get("progress", 0.0))
        self.reset_statistics()

    def set_progress(self, progress):
        self.progress = float(progress)
        if not math.isfinite(self.progress) or not 0 <= self.progress <= 1:
            raise ValueError("Auxiliary occlusion schedule progress must lie in [0, 1]")

    def reset_statistics(self):
        self.statistics = {
            "gt_by_occlusion": [0, 0, 0],
            "evaluated_gt_by_occlusion": [0, 0, 0],
            "correct_by_occlusion": [0, 0, 0],
            "unknown_gt": 0,
            "supervised_gt": 0,
            "ce_sum": 0.0,
            "batch_count": 0,
            "weighted_aux_sum": 0.0,
        }

    def _graph_zero(self, features):
        # Connect all feature levels and classifier parameters, even for batches
        # with no valid occlusion labels. Sampling one entry avoids fp16 sum overflow.
        zero = sum(feature.reshape(-1)[0].float() * 0 for feature in features)
        return zero + sum(parameter.reshape(-1)[0].float() * 0 for parameter in self.head.aux_classifier.parameters())

    def _validate_metadata(self, features, batch):
        for name in ("bboxes", "batch_idx", "cls", "occlusion", "occ_valid"):
            if name not in batch or not isinstance(batch[name], torch.Tensor):
                raise ValueError(f"Direct occlusion supervision requires tensor metadata: {name}")
        boxes, indices = self.head._validate_regions(features, batch["bboxes"], batch["batch_idx"])
        count = len(boxes)
        classes, occlusion, valid = batch["cls"], batch["occlusion"], batch["occ_valid"]
        if classes.ndim != 2 or classes.shape != (count, 1):
            raise ValueError("Detection classes must align one-to-one with GT occlusion boxes")
        if (
            classes.is_complex()
            or classes.dtype == torch.bool
            or not torch.isfinite(classes).all()
            or (classes != classes.floor()).any()
            or ((classes < 0) | (classes >= self.nc)).any()
        ):
            raise ValueError("Detection classes must be finite integer indices within the model's classes")
        if occlusion.ndim != 1 or valid.ndim != 1 or len(occlusion) != count or len(valid) != count:
            raise ValueError("GT boxes, occlusion degrees and validity masks do not align")
        if valid.dtype != torch.bool:
            raise ValueError("Occlusion validity masks must have boolean dtype")
        if occlusion.is_complex() or occlusion.dtype == torch.bool:
            raise ValueError("Occlusion degrees must be real numeric values")
        valid = valid.to(device=boxes.device)
        occlusion = occlusion.to(device=boxes.device)
        known = (occlusion == 0) | (occlusion == 1) | (occlusion == 2)
        # Check BEFORE casting to long so e.g. 1.5 cannot silently become level 1.
        if (valid & ~known).any():
            raise ValueError("Known occlusion targets must use numeric levels 0, 1 or 2")
        return boxes, indices, occlusion, valid

    def _auxiliary_loss(self, features, batch):
        """Calculate and record GT-region CE, without consulting detection assignments."""
        boxes, indices, occlusion, valid = self._validate_metadata(features, batch)
        labels = occlusion[valid].long()
        self.statistics["batch_count"] += 1
        self.statistics["unknown_gt"] += int((~valid).sum())
        for degree in range(3):
            self.statistics["gt_by_occlusion"][degree] += int((labels == degree).sum())
        if not len(labels) or self.coefficient == 0:
            return self._graph_zero(features)
        logits = self.head.classify_regions(features, boxes[valid], indices[valid])
        with torch.autocast(device_type=logits.device.type, enabled=False):
            per_gt = F.cross_entropy(logits.float(), labels, reduction="none")
            raw_ce = per_gt.mean()
        predicted = logits.detach().argmax(dim=-1)
        for degree in range(3):
            self.statistics["evaluated_gt_by_occlusion"][degree] += int((labels == degree).sum())
            self.statistics["correct_by_occlusion"][degree] += int(((labels == degree) & (predicted == degree)).sum())
        self.statistics["supervised_gt"] += len(labels)
        self.statistics["ce_sum"] += float(per_gt.detach().sum())
        self.statistics["weighted_aux_sum"] += float(raw_ce.detach()) * self.coefficient * self.progress
        return raw_ce

    def __call__(self, preds, batch):
        if isinstance(preds, dict):
            if "detect" not in preds or "occlusion_features" not in preds:
                raise ValueError("Auxiliary training predictions require detection outputs and original neck features")
            features = preds["occlusion_features"]
            raw_ce = self._auxiliary_loss(features, batch)
            detection_loss, detection_items = super().__call__(preds["detect"], batch)
            auxiliary_item = raw_ce * self.coefficient * self.progress
            auxiliary_loss = auxiliary_item * features[0].shape[0]
        else:
            detection_loss, detection_items = super().__call__(preds, batch)
            auxiliary_loss = detection_loss.sum() * 0
            auxiliary_item = auxiliary_loss.detach()
        return (
            torch.cat((detection_loss, auxiliary_loss.reshape(1))),
            torch.cat((detection_items, auxiliary_item.detach().reshape(1))),
        )
