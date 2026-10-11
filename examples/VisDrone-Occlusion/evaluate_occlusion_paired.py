# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Measure detection recall by original VisDrone occlusion grade at a fixed operating point.

The evaluator matches predictions to all valid detection targets, including targets
whose occlusion grade is unknown. Only the final recall summary is split by grade.
It does not compute occlusion-classification accuracy or replace ordinary mAP
validation. Importing this file, --help and --print-config do not import PyTorch.
"""

import argparse
import json
import locale
import math
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GRADE_MEANINGS = {"0": "no occlusion", "1": "partial occlusion", "2": "heavy occlusion"}


def build_parser():
    """Build the CLI without importing the detector or reading any data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local trained .pt checkpoint, usually weights/best.pt.")
    parser.add_argument("--data", required=True, help="Dataset YAML with five-column YOLO labels and JSON sidecars.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold; hold fixed across runs.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold; hold fixed across runs.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="Minimum same-class prediction/GT IoU for a match.")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=None, help="Output JSON filename or directory; defaults beside this run.")
    parser.add_argument("--print-config", action="store_true", help="Print options and exit without loading a model.")
    return parser


def build_configuration(args):
    """Validate numeric options and return the explicit evaluation contract."""
    if Path(args.model).suffix.lower() != ".pt":
        raise ValueError("--model must name a trained local .pt checkpoint")
    if Path(args.data).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--data must name a YAML file")
    if args.imgsz < 32 or args.max_det < 1:
        raise ValueError("--imgsz must be at least 32 and --max-det must be positive")
    if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
        raise ValueError("--conf must be finite and between 0 and 1")
    for name in ("iou", "match_iou"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and greater than 0, at most 1")
    return {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "matching_iou_threshold": args.match_iou,
        "max_det": args.max_det,
        "device": args.device,
        "output": args.output,
        "grade_meanings": GRADE_MEANINGS,
        "matching": "descending-confidence, one-to-one, same-class box IoU, all detection GT included",
        "scope": "recall and precision at a fixed operating point; not mAP or official VisDrone evaluation",
        "ignore_region_handling": "no ignore-region suppression; only converted YOLO detection targets are available",
    }


def _path_key(path):
    """Compare absolute paths correctly on both Windows and POSIX."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _output_path(args):
    if args.output:
        path = Path(args.output).expanduser()
        return path if path.suffix.lower() == ".json" else path / "occlusion_detection_recall.json"
    checkpoint = Path(args.model).expanduser().resolve()
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "weights" else checkpoint.parent
    return run_dir / f"occlusion_recall_{args.split}_conf_{args.conf:g}.json"


def main(argv=None):
    """Run prediction locally; do not call this from cloud implementation checks."""
    args = build_parser().parse_args(argv)
    config = build_configuration(args)
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    # A missing checkpoint must fail explicitly instead of selecting/downloading a model.
    for name, value in (("model", args.model), ("data", args.data)):
        if not Path(value).expanduser().is_file():
            raise FileNotFoundError(f"--{name} file does not exist: {value}")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    import torch

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.occlusion import OcclusionDataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xywh2xyxy

    data = check_det_dataset(str(Path(args.data).expanduser().resolve()), autodownload=False)
    if not data.get(args.split):
        raise ValueError(f"Dataset YAML has no nonempty '{args.split}' split")
    dataset = OcclusionDataset(
        img_path=data[args.split],
        imgsz=args.imgsz,
        batch_size=1,
        augment=False,
        safe_augment=False,
        hyp=get_cfg(),
        rect=True,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        task="detect",
        data=data,
        fraction=1.0,
        prefix="Occlusion recall: ",
    )
    labels = {_path_key(label["im_file"]): label for label in dataset.labels}
    if len(labels) != len(dataset.labels):
        raise ValueError("The selected split contains duplicate image paths; evaluation would count images twice")
    model = YOLO(str(Path(args.model).expanduser().resolve()), task="detect")
    if model.names != data["names"]:
        raise ValueError("Checkpoint class names/order differ from dataset YAML; same-class matching would be invalid")

    gt_counts = [0, 0, 0]
    matched_counts = [0, 0, 0]
    total_gt = total_pred = total_tp = unknown_gt = matched_unknown_gt = images = 0
    seen = set()
    print("Evaluating real detection recall by occlusion grade (not the auxiliary classifier).")
    print("Converted YOLO targets are used; raw VisDrone ignore regions are not evaluated.")

    # A path list passed directly to predict() eagerly loads every image. A TXT
    # manifest uses LoadImagesAndVideos and preserves streamed image loading.
    with tempfile.TemporaryDirectory(prefix="visdrone_occ_eval_") as directory:
        manifest = Path(directory) / "images.txt"
        manifest.write_text(
            "\n".join(str(Path(path).expanduser().resolve()) for path in dataset.im_files) + "\n",
            encoding=locale.getpreferredencoding(False),
        )
        predictions = model.predict(
            source=str(manifest),
            stream=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            batch=1,
            rect=True,
            save=False,
            verbose=False,
            augment=False,
            agnostic_nms=False,
        )
        try:
            for result in predictions:
                key = _path_key(result.path)
                if key not in labels or key in seen:
                    raise ValueError(f"Prediction path missing from selected dataset or repeated: {result.path}")
                seen.add(key)
                label = labels[key]
                if not label.get("normalized") or label.get("bbox_format") != "xywh":
                    raise ValueError(f"Unsupported GT coordinate representation: {result.path}")
                gt_cls = torch.as_tensor(label["cls"]).reshape(-1).long()
                gt_boxes = xywh2xyxy(torch.as_tensor(label["bboxes"], dtype=torch.float32))
                height, width = result.orig_shape
                gt_boxes *= torch.tensor([width, height, width, height], dtype=gt_boxes.dtype)
                grades = torch.as_tensor(label["occlusion"]).long()
                valid = torch.as_tensor(label["occ_valid"], dtype=torch.bool)
                if not (len(gt_cls) == len(gt_boxes) == len(grades) == len(valid)):
                    raise ValueError(f"GT/occlusion alignment failed: {result.path}")
                known = valid & (grades >= 0) & (grades <= 2)
                for grade in range(3):
                    gt_counts[grade] += int((known & (grades == grade)).sum())
                unknown_gt += int((~known).sum())

                # Work on CPU to avoid retaining per-image GPU graphs/tensors.
                boxes = result.boxes
                if boxes is None:
                    raise ValueError("The checkpoint did not return detection boxes")
                pred_boxes = boxes.xyxy.detach().float().cpu()
                pred_cls = boxes.cls.detach().long().cpu()
                scores = boxes.conf.detach().float().cpu()
                matched_gt = torch.zeros(len(gt_cls), dtype=torch.bool)
                if len(pred_boxes) and len(gt_boxes):
                    overlaps = box_iou(pred_boxes, gt_boxes)
                    for pred_index in scores.argsort(descending=True).tolist():
                        eligible = (~matched_gt) & (gt_cls == pred_cls[pred_index])
                        candidates = overlaps[pred_index].clone()
                        candidates[~eligible] = -1
                        best_iou, gt_index = candidates.max(dim=0)
                        if float(best_iou) >= args.match_iou:
                            matched_gt[int(gt_index)] = True
                for grade in range(3):
                    matched_counts[grade] += int((matched_gt & known & (grades == grade)).sum())
                matched_unknown_gt += int((matched_gt & ~known).sum())
                total_gt += len(gt_cls)
                total_pred += len(pred_boxes)
                total_tp += int(matched_gt.sum())
                images += 1
                if images % 100 == 0 or images == len(labels):
                    print(f"Evaluated {images}/{len(labels)} images")
        finally:
            predictions.close()
    if seen != set(labels):
        raise RuntimeError(f"Prediction stream omitted {len(set(labels) - seen)} dataset images")

    output = _output_path(args).resolve()
    report = dict(config)
    report.update(
        model=str(Path(args.model).expanduser().resolve()),
        data=str(Path(args.data).expanduser().resolve()),
        output=str(output),
        images=images,
        gt_by_occlusion=gt_counts,
        matched_gt_by_occlusion=matched_counts,
        recall_by_occlusion=[matched / count if count else None for matched, count in zip(matched_counts, gt_counts)],
        unknown_gt=unknown_gt,
        matched_unknown_gt=matched_unknown_gt,
        total_gt=total_gt,
        predictions=total_pred,
        true_positives=total_tp,
        false_positives=total_pred - total_tp,
        false_negatives=total_gt - total_tp,
        precision=total_tp / total_pred if total_pred else None,
        overall_recall=total_tp / total_gt if total_gt else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
