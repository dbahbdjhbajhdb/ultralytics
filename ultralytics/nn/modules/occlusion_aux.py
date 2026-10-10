# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Training-only GT-region occlusion supervision attached to an unchanged Detect head."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .head import Detect


class OcclusionAuxDetect(Detect):
    """Keep standard detection outputs and classify GT regions only during loss calculation.

    Construct this head under a caller-owned ``torch.random.fork_rng(devices=[])``
    context to leave the baseline model's random-number stream unchanged.
    """

    def __init__(self, source_head: Detect, hidden: int = 64, roi_size: int = 3):
        if type(source_head) is not Detect or source_head.end2end:
            raise ValueError("Occlusion supervision requires a standard, non-end-to-end Detect head")
        if source_head.nl != 3:
            raise ValueError("Occlusion supervision requires exactly three detection feature levels")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("Occlusion classifier hidden width must be a positive integer")
        if isinstance(roi_size, bool) or not isinstance(roi_size, int) or roi_size < 1:
            raise ValueError("ROI grid size must be a positive integer")
        channels = tuple(block[0].conv.in_channels for block in source_head.cv2)
        # Detect reads these flags while constructing its classification branches.
        self.legacy = source_head.legacy
        self.end2end = source_head.end2end
        super().__init__(nc=source_head.nc, ch=channels)
        source_parameter = next(source_head.parameters())
        self.to(device=source_parameter.device, dtype=source_parameter.dtype)
        # BN epsilon/momentum and activation options are absent from state_dict.
        # Preserve complete initialized modules, not just their parameter tensors.
        for name, module in source_head.named_children():
            setattr(self, name, copy.deepcopy(module))
        # Load before adding the auxiliary classifier so strict=True verifies every
        # original detection parameter without accepting missing auxiliary keys.
        self.load_state_dict(source_head.state_dict(), strict=True)
        for name, parameter in source_head.named_parameters():
            self.get_parameter(name).requires_grad_(parameter.requires_grad)
        for name in (
            "stride",
            "dynamic",
            "export",
            "format",
            "max_det",
            "shape",
            "anchors",
            "strides",
            "legacy",
            "xyxy",
            "inplace",
            "i",
            "f",
            "type",
            "np",
        ):
            if hasattr(source_head, name):
                setattr(self, name, copy.deepcopy(getattr(source_head, name)))
        for name, module in source_head.named_modules():
            self.get_submodule(name).training = module.training
        self.input_channels = channels
        self.roi_size = roi_size
        self.aux_classifier = nn.Sequential(nn.Linear(sum(channels), hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.aux_classifier.to(device=source_parameter.device, dtype=source_parameter.dtype)
        self.aux_classifier.train(source_head.training)
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: list[torch.Tensor]):
        """Expose unmodified neck features in training; use Detect's exact eval protocol."""
        if not self.training:
            return super().forward(x)
        features = tuple(x)  # Detect replaces list entries with its raw predictions.
        raw = super().forward(x)
        return {"detect": raw, "occlusion_features": features}

    def _validate_regions(self, features, boxes, batch_idx):
        if not isinstance(features, (tuple, list)) or len(features) != self.nl:
            raise ValueError("Occlusion ROI pooling needs all three detection feature maps")
        if any(not isinstance(feature, torch.Tensor) or feature.ndim != 4 for feature in features):
            raise ValueError("Occlusion feature maps must be BCHW tensors")
        batch_size, device = features[0].shape[0], features[0].device
        if batch_size < 1:
            raise ValueError("Occlusion feature batches must not be empty")
        for feature, channels in zip(features, self.input_channels):
            if (
                not feature.is_floating_point()
                or feature.device != device
                or feature.shape[0] != batch_size
                or feature.shape[1] != channels
                or min(feature.shape[2:]) < 1
            ):
                raise ValueError("Occlusion feature map channels, batch sizes or devices do not align")
        if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Occlusion ROI boxes must have shape (number of GTs, 4) in normalized xywh")
        if not isinstance(batch_idx, torch.Tensor) or batch_idx.ndim != 1 or len(batch_idx) != len(boxes):
            raise ValueError("Occlusion ROI boxes and image indices do not align")
        if boxes.is_complex() or batch_idx.is_complex() or batch_idx.dtype == torch.bool:
            raise ValueError("Occlusion ROI coordinates and image indices must be real numeric values")
        boxes = boxes.to(device=device, dtype=torch.float32)
        indices = batch_idx.to(device=device)
        if not torch.isfinite(boxes).all() or not torch.isfinite(indices).all():
            raise ValueError("Occlusion ROI coordinates and image indices must be finite")
        if ((boxes < -1e-6) | (boxes > 1 + 1e-6)).any() or (boxes[:, 2:] <= 0).any():
            raise ValueError("Occlusion ROI boxes must be normalized with strictly positive width and height")
        if (indices != indices.floor()).any() or ((indices < 0) | (indices >= batch_size)).any():
            raise ValueError("Occlusion ROI image indices must be integers within the feature batch")
        return boxes, indices.long()

    def pool_regions(self, features, boxes, batch_idx):
        """Return one differentiable multi-scale descriptor per GT, in the input GT order.

        A 3x3 grid samples uniformly spaced bin centers in each normalized xywh
        region. Regions are grouped by image so a feature map is never repeated
        once per object. This is grid sampling, not torchvision ROIAlign.
        """
        boxes, batch_idx = self._validate_regions(features, boxes, batch_idx)
        if not len(boxes):
            return features[0].new_empty((0, sum(self.input_channels)), dtype=torch.float32)
        fractions = (torch.arange(self.roi_size, device=boxes.device, dtype=torch.float32) + 0.5) / self.roi_size
        offset_y, offset_x = torch.meshgrid(fractions - 0.5, fractions - 0.5, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1)
        descriptors, positions = [], []
        for image_index in range(features[0].shape[0]):
            selected = torch.where(batch_idx == image_index)[0]
            if not len(selected):
                continue
            regions = boxes[selected]
            grid = regions[:, None, None, :2] + offsets[None] * regions[:, None, None, 2:]
            grid = (grid * 2 - 1).reshape(1, len(selected) * self.roi_size, self.roi_size, 2)
            levels = []
            for feature in features:
                with torch.autocast(device_type=feature.device.type, enabled=False):
                    sampled = F.grid_sample(
                        feature[image_index : image_index + 1].float(),
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                pooled = sampled.reshape(feature.shape[1], len(selected), self.roi_size, self.roi_size)
                levels.append(pooled.mean(dim=(-1, -2)).transpose(0, 1))
            descriptors.append(torch.cat(levels, dim=1))
            positions.append(selected)
        image_order = torch.cat(positions)
        return torch.cat(descriptors, dim=0)[image_order.argsort()]

    def classify_regions(self, features, boxes, batch_idx):
        """Predict three occlusion classes for supplied GT regions during training."""
        descriptors = self.pool_regions(features, boxes, batch_idx)
        return self.aux_classifier(descriptors.to(dtype=self.aux_classifier[0].weight.dtype))
