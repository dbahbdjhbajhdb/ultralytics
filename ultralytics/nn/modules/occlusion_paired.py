# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Local feature compensation and spatial GT regions for paired occlusion training."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .head import Detect


def _compact_projection(in_channels: int, out_channels: int) -> nn.Sequential:
    """Project a feature map without introducing batch-dependent normalization."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(math.gcd(out_channels, 8), out_channels),
        nn.SiLU(),
    )


class _LocalFeatureCompensation(nn.Module):
    """Add a gated local residual from P2 detail, P3 neighborhoods and P4 context.

    The final projection starts at zero. Its initial output therefore leaves the
    already initialized detector's P3 features exactly unchanged. A spatial gate
    is predicted from image features; no occlusion grades are needed at inference.
    """

    def __init__(self, p2_channels: int, p3_channels: int, p4_channels: int, hidden: int):
        super().__init__()
        self.detail = _compact_projection(p2_channels, hidden)
        self.local_projection = _compact_projection(p3_channels, hidden)
        self.local_context = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(math.gcd(hidden, 8), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=3, dilation=3, groups=hidden, bias=False),
            nn.GroupNorm(math.gcd(hidden, 8), hidden),
            nn.SiLU(),
        )
        self.context = _compact_projection(p4_channels, hidden)
        self.merge = _compact_projection(3 * hidden, hidden)
        self.gate = nn.Conv2d(3 * hidden, 1, 1)
        self.output = nn.Conv2d(hidden, p3_channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        target_size = p3.shape[-2:]
        detail = F.avg_pool2d(p2, kernel_size=2, stride=2)
        if detail.shape[-2:] != target_size:
            detail = F.interpolate(detail, size=target_size, mode="bilinear", align_corners=False)
        detail = self.detail(detail)
        local = self.local_context(self.local_projection(p3))
        context = F.interpolate(self.context(p4), size=target_size, mode="bilinear", align_corners=False)
        joined = torch.cat((detail, local, context), dim=1)
        residual = self.output(self.merge(joined))
        return p3 + self.gate(joined).sigmoid() * residual


class PairedOcclusionDetect(Detect):
    """Preserve standard YOLO detection and expose regions for paired-view training.

    Construct this replacement inside a caller-owned
    ``torch.random.fork_rng(devices=[])`` context so the extra layers do not alter
    the baseline model's random-number stream. Detection head modules, parameter
    values, normalization settings and runtime metadata are copied from the
    initialized standard head.

    With compensation enabled the input list is ``[P2, P3, P4, P5]``. Otherwise
    it is the ordinary ``[P3, P4, P5]`` list. Only P3 receives compensation.
    """

    def __init__(
        self,
        source_head: Detect,
        p2_channels: int,
        hidden: int = 32,
        roi_size: int = 3,
        compensation: bool = True,
    ):
        if type(source_head) is not Detect or source_head.end2end:
            raise ValueError("Paired occlusion training requires a standard, non-end-to-end Detect head")
        if source_head.nl != 3 or tuple(source_head.stride.tolist()) != (8.0, 16.0, 32.0):
            raise ValueError("Paired occlusion training requires three Detect levels with strides 8, 16 and 32")
        for name, value in (("P2 channel count", p2_channels), ("Hidden width", hidden), ("ROI grid size", roi_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(compensation, bool):
            raise ValueError("Feature compensation must be a boolean")

        channels = tuple(block[0].conv.in_channels for block in source_head.cv2)
        self.legacy = source_head.legacy
        self.end2end = source_head.end2end
        super().__init__(nc=source_head.nc, ch=channels)
        source_parameter = next(source_head.parameters())
        self.to(device=source_parameter.device, dtype=source_parameter.dtype)
        # Epsilon, momentum and activation settings are not encoded in state_dict.
        for name, module in source_head.named_children():
            setattr(self, name, copy.deepcopy(module))
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
        self.p2_channels = p2_channels
        self.hidden = hidden
        self.roi_size = roi_size
        self.roi_channels = (channels[0],)
        self.compensation_enabled = compensation
        self.capture_pair_features = False
        if compensation:
            self.local_compensation = _LocalFeatureCompensation(p2_channels, channels[0], channels[1], hidden)
            self.local_compensation.to(device=source_parameter.device, dtype=source_parameter.dtype)
            self.local_compensation.train(source_head.training)
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: list[torch.Tensor]):
        """Keep Detect's inference protocol and provide spatial training features."""
        expected = 4 if self.compensation_enabled else 3
        if not isinstance(x, (tuple, list)) or len(x) != expected:
            raise ValueError(f"Paired Detect expects {expected} input feature maps")
        channels = (self.p2_channels, *self.input_channels) if self.compensation_enabled else self.input_channels
        if any(
            not isinstance(feature, torch.Tensor)
            or feature.ndim != 4
            or feature.shape[1] != channel
            or min(feature.shape[-2:]) < 1
            for feature, channel in zip(x, channels)
        ):
            raise ValueError("Paired Detect input features must be nonempty BCHW tensors with configured channels")
        if any(feature.shape[0] != x[0].shape[0] or feature.device != x[0].device for feature in x):
            raise ValueError("Paired Detect input batches and devices must align")
        if self.compensation_enabled:
            p2, p3, p4, p5 = x
            if min(p2.shape[-2:]) < 2:
                raise ValueError("P2 features must be at least 2 by 2 for detail downsampling")
            enhanced = self.local_compensation(p2, p3, p4)
            detection_features = [enhanced, p4, p5]
        else:
            p3, p4, p5 = x
            enhanced = p3
            detection_features = [p3, p4, p5]
        raw = Detect.forward(self, detection_features)
        if self.training or self.capture_pair_features:
            return {"detect": raw, "pair_features": (enhanced,), "reference_features": (p3,)}
        return raw

    def _validate_regions(self, features, boxes, batch_idx):
        if not isinstance(features, (tuple, list)) or len(features) != len(self.roi_channels):
            raise ValueError("Paired ROI feature count does not match its input levels")
        if any(not isinstance(feature, torch.Tensor) or feature.ndim != 4 for feature in features):
            raise ValueError("Paired ROI feature maps must be BCHW tensors")
        batch_size, device = features[0].shape[0], features[0].device
        if batch_size < 1:
            raise ValueError("Paired ROI feature batches must not be empty")
        for feature, channels in zip(features, self.roi_channels):
            if (
                not feature.is_floating_point()
                or feature.device != device
                or feature.shape[0] != batch_size
                or feature.shape[1] != channels
                or min(feature.shape[2:]) < 1
            ):
                raise ValueError("Paired ROI feature channels, batch sizes or devices do not align")
        if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Paired ROI boxes must have shape (number of GTs, 4) in normalized xywh")
        if not isinstance(batch_idx, torch.Tensor) or batch_idx.ndim != 1 or len(batch_idx) != len(boxes):
            raise ValueError("Paired ROI boxes and image indices do not align")
        if boxes.is_complex() or batch_idx.is_complex() or batch_idx.dtype == torch.bool:
            raise ValueError("Paired ROI coordinates and image indices must be real numeric values")
        boxes = boxes.to(device=device, dtype=torch.float32)
        indices = batch_idx.to(device=device)
        if not torch.isfinite(boxes).all() or not torch.isfinite(indices).all():
            raise ValueError("Paired ROI coordinates and image indices must be finite")
        if ((boxes < -1e-6) | (boxes > 1 + 1e-6)).any() or (boxes[:, 2:] <= 0).any():
            raise ValueError("Paired ROI boxes must be normalized with strictly positive width and height")
        if (indices != indices.floor()).any() or ((indices < 0) | (indices >= batch_size)).any():
            raise ValueError("Paired ROI image indices must be integers within the feature batch")
        return boxes, indices.long()

    def pool_regions(self, features, boxes, batch_idx):
        """Sample spatial ROI grids in GT order without averaging their bins.

        Each map is processed once per image rather than repeated for each GT.
        FP32 bilinear sampling retains an ``N x C x roi_size x roi_size`` tensor
        for the paired feature objective. Sampling is not torchvision ROIAlign.
        """
        boxes, batch_idx = self._validate_regions(features, boxes, batch_idx)
        if not len(boxes):
            return features[0].new_empty((0, sum(self.roi_channels), self.roi_size, self.roi_size), dtype=torch.float32)
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
                levels.append(
                    sampled.reshape(feature.shape[1], len(selected), self.roi_size, self.roi_size).permute(1, 0, 2, 3)
                )
            descriptors.append(torch.cat(levels, dim=1))
            positions.append(selected)
        image_order = torch.cat(positions)
        return torch.cat(descriptors, dim=0)[image_order.argsort()]
