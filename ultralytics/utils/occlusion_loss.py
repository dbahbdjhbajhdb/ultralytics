# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Experimental size- and occlusion-weighted detection loss."""

from __future__ import annotations

import math

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class WeightedDetectionLoss(v8DetectionLoss):
    """Bounded GT-derived positive weighting, not a learned dynamic controller."""

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        self.config = getattr(model, "occlusion_config", {}).copy()
        self.config.setdefault("size_weight", 0.5)
        self.config.setdefault("occ_weight", 0.5)
        self.config.setdefault("joint_weight", 0.0)
        self.config.setdefault("size_reference", 32.0)
        self.config.setdefault("cap", 2.0)
        self.set_progress(self.config.get("progress", 1.0))
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics = {
            "gt_by_occlusion": [0, 0, 0],
            "matched_gt_by_occlusion": [0, 0, 0],
            "gt_with_unknown_occlusion": 0,
            "matched_gt_with_unknown_occlusion": 0,
            "positive_anchors": 0,
            "weight_sum": 0.0,
            "weight_min": None,
            "weight_max": None,
        }

    def set_progress(self, progress):
        self.progress = float(progress)
        if not math.isfinite(self.progress) or not 0 <= self.progress <= 1:
            raise ValueError("Weight schedule progress must lie in [0, 1]")

    def positive_weights(self, batch, gt_bboxes, assigned_gt, fg_mask):
        fg_mask = fg_mask.bool()
        occ = torch.full(gt_bboxes.shape[:2], -1.0, device=self.device)
        occ_valid = torch.zeros(gt_bboxes.shape[:2], dtype=torch.bool, device=self.device)
        batch_idx = batch["batch_idx"].to(self.device)
        occlusion = batch["occlusion"].to(self.device)
        known_degrees = (occlusion == 0) | (occlusion == 1) | (occlusion == 2)
        validity = batch.get("occ_valid")
        if validity is None:
            # Backward compatibility for batches/checkpoints created before the validity mask.
            validity = known_degrees
        else:
            validity = validity.to(self.device)
            if validity.dtype != torch.bool:
                raise ValueError("Occlusion validity mask must have boolean dtype")
        if occlusion.ndim != 1 or batch_idx.ndim != 1 or validity.ndim != 1:
            raise ValueError("Occlusion values, validity and batch indices must be one-dimensional")
        if len(occlusion) != len(batch_idx) or len(validity) != len(occlusion):
            raise ValueError("Occlusion validity/target alignment failed")
        if (validity & ~known_degrees).any():
            raise ValueError("Known occlusion targets must use numeric levels 0, 1 or 2")
        for j in range(gt_bboxes.shape[0]):
            selected = batch_idx == j
            values = occlusion[selected]
            occ[j, : len(values)] = values
            occ_valid[j, : len(values)] = validity[selected]
        dimensions = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).float().clamp_min(0)
        area = dimensions.prod(-1)
        alpha = float(self.config["size_weight"])
        beta = float(self.config["occ_weight"])
        gamma = float(self.config["joint_weight"])
        reference = float(self.config["size_reference"])
        cap = float(self.config["cap"])
        if not all(math.isfinite(v) for v in (alpha, beta, gamma, reference, cap)):
            raise ValueError("Weight coefficients, reference size and cap must be finite")
        if min(alpha, beta, gamma) < 0 or reference <= 0 or cap < 1:
            raise ValueError("Weights must be nonnegative, reference size positive and cap at least 1")
        smallness = reference / (area.sqrt() + reference)
        # Unknown visibility contributes neither an occlusion bonus nor a joint bonus.
        # Keep the sentinel separate from level 0 in the supervision and statistics.
        occlusion_level = torch.where(occ_valid, occ / 2, torch.zeros_like(occ))
        difficulty = alpha * smallness + beta * occlusion_level + gamma * smallness * occlusion_level
        gt_weights = (1 + self.progress * difficulty).clamp(1, cap)
        anchor_weights = (
            gt_weights.gather(1, assigned_gt.clamp_min(0))
            if gt_weights.shape[1]
            else torch.ones_like(assigned_gt, dtype=gt_bboxes.dtype)
        )
        # Keep the original loss denominator: a batch of equally hard samples must not cancel its own weights.
        valid_gt = gt_bboxes.sum(-1) > 0
        matched_gt = torch.zeros_like(valid_gt)
        for j in range(gt_bboxes.shape[0]):
            if fg_mask[j].any():
                matched_gt[j, assigned_gt[j, fg_mask[j]].unique()] = True
        for degree in range(3):
            group = valid_gt & occ_valid & (occ == degree)
            self.statistics["gt_by_occlusion"][degree] += int(group.sum())
            self.statistics["matched_gt_by_occlusion"][degree] += int((group & matched_gt).sum())
        unknown_gt = valid_gt & ~occ_valid
        self.statistics["gt_with_unknown_occlusion"] += int(unknown_gt.sum())
        self.statistics["matched_gt_with_unknown_occlusion"] += int((unknown_gt & matched_gt).sum())
        positive = anchor_weights[fg_mask]
        if positive.numel():
            self.statistics["positive_anchors"] += positive.numel()
            self.statistics["weight_sum"] += float(positive.sum())
            old_min, old_max = self.statistics["weight_min"], self.statistics["weight_max"]
            self.statistics["weight_min"] = (
                min(float(positive.min()), old_min) if old_min is not None else float(positive.min())
            )
            self.statistics["weight_max"] = (
                max(float(positive.max()), old_max) if old_max is not None else float(positive.max())
            )
        return torch.where(fg_mask, anchor_weights, torch.ones_like(anchor_weights))

    def __call__(self, preds, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, assigned_gt = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        anchor_weights = self.positive_weights(batch, gt_bboxes, assigned_gt, fg_mask)
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype))
            * torch.where(target_scores > 0, anchor_weights.unsqueeze(-1), 1.0)
        ).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores * anchor_weights.unsqueeze(-1),
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
