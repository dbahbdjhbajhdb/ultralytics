# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Experimental size- and occlusion-weighted detection loss."""

import os
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class WeightedDetectionLoss(v8DetectionLoss):
    """Bounded GT-derived positive weighting, not a learned dynamic controller."""

    def positive_weights(self, batch, gt_bboxes, assigned_gt, fg_mask):
        fg_mask = fg_mask.bool()
        occ = torch.zeros(gt_bboxes.shape[:2], device=self.device)
        for j in range(gt_bboxes.shape[0]):
            selected = batch["batch_idx"].to(self.device) == j
            values = batch["occlusion"].to(self.device)[selected]
            occ[j, : len(values)] = values
        area = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        alpha = float(os.environ.get("SIZE_WEIGHT", "1.0"))
        beta = float(os.environ.get("OCC_WEIGHT", "1.0"))
        gt_weights = (1 + alpha * 32 / (area.clamp_min(0).sqrt() + 32) + beta * occ / 2).clamp(1, 3)
        anchor_weights = (
            gt_weights.gather(1, assigned_gt.clamp_min(0))
            if gt_weights.shape[1]
            else torch.ones_like(assigned_gt, dtype=gt_bboxes.dtype)
        )
        if fg_mask.any():
            anchor_weights = anchor_weights / anchor_weights[fg_mask].mean().clamp_min(1e-8)
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
