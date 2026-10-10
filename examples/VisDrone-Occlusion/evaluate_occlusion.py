"""Pilot diagnostics and held-out scalar temperature calibration; not official VisDrone evaluation."""

import argparse, json, math, os, sys
from pathlib import Path

root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(Path(__file__).resolve().parents[2] / "runs/visdrone_occ")))
sys.path.insert(0, str(root))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/ultralytics-config")
import cv2, numpy as np, torch
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
    samples = []
    for image in sorted((data / "images" / split).glob("*.jpg")):
        original = cv2.imread(str(image))
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
        prelabels, matched = match(boxes[keep], classes[keep], scores[keep], gt, gcls, eta)
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
    total = 0.0
    for low in np.linspace(0, 0.9, 10):
        mask = (p >= low) & (p < (low + 0.1) if low < 0.9 else p <= 1)
        if mask.any():
            total += float(mask.float().mean() * abs(p[mask].mean() - y[mask].mean()))
    return total


def fit_temperature(samples, eta):
    scores = []
    labels = []
    for s in samples:
        keep = s["keep"][s["eligible"]]
        y, _ = match(s["boxes"][keep], s["classes"][keep], s["scores"][keep], s["gt"], s["gcls"], eta)
        scores.append(s["scores"][keep])
        labels.append(y)
    p = torch.cat(scores)
    y = torch.cat(labels)
    if not len(p) or y.sum() == 0 or y.sum() == len(y):
        return {"status": "not_fitted_degenerate_labels", "n": len(p), "positives": int(y.sum()), "temperature": 1.0}
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
        "status": "fitted",
        "n": len(p),
        "positives": int(y.sum()),
        "temperature": temperature,
        "cal_brier_before": float(((p - y) ** 2).mean()),
        "cal_brier_after": float(((q - y) ** 2).mean()),
        "cal_ece_before": ece(p, y),
        "cal_ece_after": ece(q, y),
    }


def summarize(samples, eta, threshold, temperature):
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
        p = (logit(s["scores"][keep]) / temperature).sigmoid()
        active = p >= threshold
        k = keep[active]
        y, matched = match(s["boxes"][k], s["classes"][k], p[active], s["gt"], s["gcls"], eta)
        tp += int(y.sum())
        fp += len(y) - int(y.sum())
        gt_count += len(s["gt"])
        # Calibration metrics on the same eligible retained detections, before deployment score threshold.
        full_y, _ = match(s["boxes"][keep], s["classes"][keep], p, s["gt"], s["gcls"], eta)
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
    p = torch.cat(all_p)
    y = torch.cat(all_y)
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
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = YOLO(args.weights).model.cpu().eval()
    cal = collect(model, args.data, "cal", args.imgsz, args.candidate_conf, args.nms_iou, args.match_iou)
    val = collect(model, args.data, args.eval_split, args.imgsz, args.candidate_conf, args.nms_iou, args.match_iou)
    calibration = fit_temperature(cal, args.match_iou)
    baseline, traces = summarize(val, args.match_iou, args.report_conf, 1.0)
    calibrated, _ = summarize(val, args.match_iou, args.report_conf, calibration["temperature"])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "pilot only; validation-derived disjoint splits; non-official VisDrone diagnostics",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": calibration,
        "uncalibrated": baseline,
        "calibrated": calibrated,
        "limitations": [
            "scalar monotonic calibration does not reorder candidates or change NMS decisions",
            "ignore handling is a pilot convention, not official VisDrone evaluation",
            "smallness uses original image box area; training weights use resized box area",
            "no max_det cap used; not directly comparable with standard validation AP",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "suppression_traces.json").write_text(json.dumps(traces))
    torch.save({"cal": cal, "val": val}, args.output / "candidate_cache.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
