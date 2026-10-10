"""Pilot diagnostics and held-out monotonic score calibration; not official VisDrone evaluation."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(repository_root))).expanduser()
sys.path.insert(0, str(repository_root))
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "runs/yolo_config"))
os.environ.setdefault("MPLCONFIGDIR", str(root / "runs/matplotlib"))
import cv2
import torch
from torchvision.ops import box_iou

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import xywh2xyxy


def greedy_trace(boxes, scores, classes, threshold):
    order = scores.argsort(descending=True)
    keep = []
    trace = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        overlap = box_iou(boxes[i : i + 1], boxes[rest])[0]
        remove = (overlap > threshold) & (classes[rest] == classes[i])
        trace.extend(
            {"removed": int(j), "suppressor": i, "iou": float(o)} for j, o in zip(rest[remove], overlap[remove])
        )
        order = rest[~remove]
    return torch.tensor(keep, dtype=torch.long), trace


def match(boxes, classes, scores, gt, gcls, eta):
    labels = torch.zeros(len(boxes))
    matched = torch.zeros(len(gt), dtype=torch.bool)
    if not len(boxes) or not len(gt):
        return labels, matched
    overlaps = box_iou(boxes, gt)
    for index in scores.argsort(descending=True):
        values = overlaps[index].clone()
        values[(gcls != classes[index]) | matched] = -1
        value, j = values.max(0)
        if value >= eta:
            labels[index] = 1
            matched[j] = True
    return labels, matched


def coverage(boxes, classes, gt, gcls, eta):
    if not len(boxes):
        return torch.zeros(len(gt), dtype=torch.bool)
    overlaps = box_iou(boxes, gt)
    overlaps[classes[:, None] != gcls[None, :]] = 0
    return overlaps.max(0).values >= eta


def collect(model, data, split, imgsz, conf, nms_iou, eta):
    images = sorted(
        image
        for image in (data / "images" / split).glob("*")
        if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No supported images found in required split: {data / 'images' / split}")
    samples = []
    for image in images:
        original = cv2.imread(str(image))
        if original is None:
            raise ValueError(f"Could not read image: {image}")
        h, w = original.shape[:2]
        img = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False)(image=original)
        gain = min(imgsz / h, imgsz / w, 1.0)
        left = round((imgsz - round(w * gain)) / 2 - 0.1)
        top = round((imgsz - round(h * gain)) / 2 - 0.1)
        tensor = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.no_grad():
            output = model(tensor)
        raw = output[0] if isinstance(output, tuple) else output
        raw = raw[0].T.cpu()
        boxes = xywh2xyxy(raw[:, :4])
        scores, classes = raw[:, 4:].max(1)
        candidate_ids = torch.arange(len(scores))
        active = scores >= conf
        boxes, scores, classes, candidate_ids = boxes[active], scores[active], classes[active], candidate_ids[active]
        # No max_det/candidate cap: assess suppression separately from truncation.
        keep, trace = greedy_trace(boxes, scores, classes, nms_iou)
        meta = json.loads((data / "metadata" / split / (image.stem + ".json")).read_text())
        gt = []
        gcls = []
        occ = []
        area = []
        for obj in meta["objects"]:
            cls, cx, cy, bw, bh = obj.get("yolo", [obj.get("class_id"), *obj.get("bbox_xywh_normalized", [])])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt.append([x1 * gain + left, y1 * gain + top, x2 * gain + left, y2 * gain + top])
            gcls.append(int(cls))
            occ.append(obj["occlusion"])
            area.append(bw * w * bh * h)
        gt = torch.tensor(gt, dtype=torch.float32).reshape(-1, 4)
        gcls = torch.tensor(gcls)
        eligible = torch.ones(len(keep), dtype=torch.bool)
        # Exclude unmatched detections predominantly inside ignored regions (pilot rule, not official metric).
        prelabels, _ = match(boxes[keep], classes[keep], scores[keep], gt, gcls, eta)
        for region in meta.get("ignored", meta.get("ignored_annotations", [])):
            x, y, bw, bh = region.get("xywh", region.get("fields", [])[:4])
            r = torch.tensor([x * gain + left, y * gain + top, (x + bw) * gain + left, (y + bh) * gain + top])
            lo = torch.maximum(boxes[keep, :2], r[:2])
            hi = torch.minimum(boxes[keep, 2:], r[2:])
            inter = (hi - lo).clamp_min(0).prod(1)
            ba = (boxes[keep, 2:] - boxes[keep, :2]).clamp_min(0).prod(1).clamp_min(1e-9)
            eligible &= ~((inter / ba > 0.5) & (prelabels == 0))
        samples.append(
            {
                "image": image.stem,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "candidate_ids": candidate_ids,
                "keep": keep,
                "trace": trace,
                "gt": gt,
                "gcls": gcls,
                "occ": torch.tensor(occ),
                "area": torch.tensor(area),
                "eligible": eligible,
            }
        )
    return samples


def logit(p):
    return torch.logit(p.clamp(1e-7, 1 - 1e-7))


def ece(p, y):
    """Ten-bin ECE with every probability in [0, 1] assigned exactly once."""
    if p.shape != y.shape or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("ECE requires matching shapes and finite probabilities in [0, 1]")
    bins = (p * 10).floor().long().clamp(max=9)
    total = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            total += float(mask.float().mean() * abs(p[mask].mean() - y[mask].mean()))
    return total


def calibration_pairs(samples, eta):
    """Build calibration labels only from eligible post-NMS detections in the supplied split."""
    scores = []
    labels = []
    for s in samples:
        keep = s["keep"][s["eligible"]]
        y, _ = match(s["boxes"][keep], s["classes"][keep], s["scores"][keep], s["gt"], s["gcls"], eta)
        scores.append(s["scores"][keep])
        labels.append(y)
    p = torch.cat(scores).detach().double().cpu() if scores else torch.empty(0, dtype=torch.float64)
    y = torch.cat(labels).detach().double().cpu() if labels else torch.empty(0, dtype=torch.float64)
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration scores must be finite probabilities in [0, 1]")
    return p, y


def calibration_metrics(p, y, q):
    return {
        "cal_brier_before": float(((p - y) ** 2).mean()),
        "cal_brier_after": float(((q - y) ** 2).mean()),
        "cal_ece_before": ece(p, y),
        "cal_ece_after": ece(q, y),
    }


def fit_temperature(samples, eta):
    """Fit the original scalar-temperature control on calibration samples only."""
    p, y = calibration_pairs(samples, eta)
    if not len(p) or y.sum() == 0 or y.sum() == len(y):
        return {
            "method": "temperature",
            "status": "not_fitted_degenerate_labels",
            "n": len(p),
            "positives": int(y.sum()),
            "temperature": 1.0,
        }
    z = logit(p)
    lt = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([lt], lr=0.03)
    for _ in range(250):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / lt.exp(), y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            lt.clamp_(math.log(0.05), math.log(20.0))
    temperature = float(lt.exp())
    q = (z / temperature).sigmoid()
    return {
        "method": "temperature",
        "status": "fitted",
        "n": len(p),
        "positives": int(y.sum()),
        "temperature": temperature,
        **calibration_metrics(p, y, q),
    }


def fit_platt(samples, eta):
    """Fit sigmoid(a * logit(p) + b), constraining a > 0 to preserve score ordering.

    The intercept can lower uniformly overconfident detections, which temperature alone cannot
    do when their original scores are above 0.5. Bounds keep near-separable pilot fits finite.
    """
    p, y = calibration_pairs(samples, eta)
    result = {"method": "platt", "n": len(p), "positives": int(y.sum()), "slope": 1.0, "bias": 0.0}
    if not len(p) or y.sum() == 0 or y.sum() == len(y):
        return {**result, "status": "not_fitted_degenerate_labels"}
    z = logit(p)
    log_slope = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_slope, bias], lr=0.05)
    best_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(z, y))
    best = (1.0, 0.0)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(log_slope.exp() * z + bias, y)
        if not torch.isfinite(loss):
            return {**result, "status": "not_fitted_nonfinite_optimization"}
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_slope.clamp_(math.log(0.01), math.log(100.0))
            bias.clamp_(-30.0, 30.0)
            objective = float(torch.nn.functional.binary_cross_entropy_with_logits(log_slope.exp() * z + bias, y))
            if objective < best_loss:
                best_loss = objective
                best = (float(log_slope.exp()), float(bias))
    result.update(status="fitted", slope=best[0], bias=best[1])
    q = apply_calibration(p, result)
    return {**result, **calibration_metrics(p, y, q)}


def fit_calibration(samples, eta, method="platt"):
    if method == "none":
        return {"method": "none", "status": "not_requested", "n": 0, "positives": 0}
    if method == "temperature":
        return fit_temperature(samples, eta)
    if method == "platt":
        return fit_platt(samples, eta)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibration(p, calibration):
    """Apply a persisted mapping after NMS; never use calibrated scores to recompute NMS."""
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration scores must be finite probabilities in [0, 1]")
    method = calibration.get("method", "temperature")
    if method == "none" or calibration.get("status", "").startswith("not_fitted"):
        return p
    if method == "temperature":
        temperature = float(calibration["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Calibration temperature must be finite and positive")
        return (logit(p) / temperature).sigmoid()
    if method == "platt":
        slope, bias = float(calibration["slope"]), float(calibration["bias"])
        if not math.isfinite(slope) or slope <= 0 or not math.isfinite(bias):
            raise ValueError("Platt calibration requires a finite positive slope and finite bias")
        return (slope * logit(p) + bias).sigmoid()
    raise ValueError(f"Unknown calibration method: {method}")


def validate_calibration_splits(cal, evaluation):
    """Fail before fitting if calibration and evaluation contain the same image or image content."""
    for key in ("image", "image_sha256"):
        cal_ids = {s[key] for s in cal if key in s}
        eval_ids = {s[key] for s in evaluation if key in s}
        overlap = cal_ids & eval_ids
        if overlap:
            raise ValueError(f"Calibration/evaluation split overlap ({key}): {len(overlap)} shared images")


def summarize(samples, eta, threshold, temperature=1.0, calibration=None):
    """Summarize detections while preserving the legacy numeric-temperature interface."""
    if calibration is None:
        calibration = {"method": "temperature", "temperature": temperature}
    counts = {str(o): {"gt": 0, "pre_covered": 0, "post_covered": 0, "suppression_lost": 0, "tp": 0} for o in (0, 1, 2)}
    tp = fp = gt_count = 0
    all_p = []
    all_y = []
    small = {"gt": 0, "tp": 0}
    traces = []
    for s in samples:
        pre = coverage(s["boxes"], s["classes"], s["gt"], s["gcls"], eta)
        post = coverage(s["boxes"][s["keep"]], s["classes"][s["keep"]], s["gt"], s["gcls"], eta)
        keep = s["keep"][s["eligible"]]
        p = apply_calibration(s["scores"][keep], calibration)
        active = p >= threshold
        k = keep[active]
        # Match in the original order even if finite-precision sigmoid creates calibrated score ties.
        y, matched = match(s["boxes"][k], s["classes"][k], s["scores"][k], s["gt"], s["gcls"], eta)
        tp += int(y.sum())
        fp += len(y) - int(y.sum())
        gt_count += len(s["gt"])
        # Calibration metrics on the same eligible retained detections, before deployment score threshold.
        full_y, _ = match(s["boxes"][keep], s["classes"][keep], s["scores"][keep], s["gt"], s["gcls"], eta)
        all_p.append(p)
        all_y.append(full_y)
        for o in (0, 1, 2):
            mask = s["occ"] == o
            c = counts[str(o)]
            c["gt"] += int(mask.sum())
            c["pre_covered"] += int(pre[mask].sum())
            c["post_covered"] += int(post[mask].sum())
            c["suppression_lost"] += int((pre & ~post & mask).sum())
            c["tp"] += int(matched[mask].sum())
        mask = s["area"] < 32**2
        small["gt"] += int(mask.sum())
        small["tp"] += int(matched[mask].sum())
        traces.append(
            {
                "image": s["image"],
                "n_candidates": len(s["scores"]),
                "n_retained": len(s["keep"]),
                "candidate_ids": s["candidate_ids"].tolist(),
                "suppression": s["trace"],
            }
        )
    p = torch.cat(all_p) if all_p else torch.empty(0)
    y = torch.cat(all_y) if all_y else torch.empty(0)
    return {
        "gt": gt_count,
        "tp": tp,
        "fp": fp,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(gt_count, 1),
        "occlusion_strata": counts,
        "small_native_area_lt_1024": small,
        "brier": float(((p - y) ** 2).mean()) if len(p) else None,
        "ece": ece(p, y) if len(p) else None,
    }, traces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", type=Path, default=root / "data/processed/visdrone_pilot")
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--candidate-conf", type=float, default=0.0001)
    p.add_argument("--report-conf", type=float, default=0.01)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--calibration", choices=["none", "temperature", "platt"], default="platt")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = YOLO(args.weights).model.cpu().eval()
    cal = (
        collect(model, args.data, "cal", args.imgsz, args.candidate_conf, args.nms_iou, args.match_iou)
        if args.calibration != "none"
        else []
    )
    val = collect(model, args.data, args.eval_split, args.imgsz, args.candidate_conf, args.nms_iou, args.match_iou)
    validate_calibration_splits(cal, val)
    calibration = fit_calibration(cal, args.match_iou, args.calibration)
    calibration.update(
        fit_split="cal" if args.calibration != "none" else None,
        evaluation_split=args.eval_split,
        match_iou=args.match_iou,
        fit_images=[s["image"] for s in cal],
    )
    baseline, traces = summarize(val, args.match_iou, args.report_conf, calibration={"method": "none"})
    calibrated, _ = summarize(val, args.match_iou, args.report_conf, calibration=calibration)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "non-official VisDrone diagnostics; calibration uses cal, evaluation uses the selected split",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": calibration,
        "uncalibrated": baseline,
        "calibrated": calibrated,
        "limitations": [
            "global monotonic calibration does not reorder candidates or change NMS decisions",
            "calibration changes confidence thresholds, not localization or candidates removed by NMS",
            "calibration metrics measure correctness of eligible post-NMS detections at the specified match IoU",
            "ignore handling is a pilot convention, not official VisDrone evaluation",
            "smallness uses original image box area; training weights use resized box area",
            "no max_det cap used; not directly comparable with standard validation AP",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (args.output / "suppression_traces.json").write_text(json.dumps(traces))
    torch.save({"cal": cal, "val": val}, args.output / "candidate_cache.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
