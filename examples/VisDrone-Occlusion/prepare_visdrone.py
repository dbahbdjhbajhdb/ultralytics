import os
import shutil

"""Preserve VisDrone occlusion/truncation alongside standard five-column YOLO labels."""
import argparse, hashlib, json, random
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def convert(source, dest, split, images):
    counts = Counter()
    for name in images:
        image = source / "images" / name
        with Image.open(image) as im:
            width, height = im.size
        rows = []
        objects = []
        ignored = []
        seen = {}
        annotation = source / "annotations" / f"{image.stem}.txt"
        for line_no, line in enumerate(annotation.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = [v.strip() for v in line.split(",")]
            if len(fields) < 8:
                raise ValueError(f"{annotation}:{line_no}: needs 8 fields")
            x, y, w, h = map(float, fields[:4])
            score, category, trunc, occ = map(int, fields[4:8])
            if score == 0 or category not in range(1, 11):
                ignored.append({"source_line": line_no, "xywh": [x, y, w, h], "category": category, "score": score})
                counts["ignored"] += 1
                continue
            if occ not in (0, 1, 2) or trunc not in (0, 1, 2):
                raise ValueError("invalid metadata")
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("invalid box")
            values = [
                category - 1,
                (x1 + x2) / 2 / width,
                (y1 + y2) / 2 / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            row = f"{values[0]} " + " ".join(f"{v:.8f}" for v in values[1:])
            if row in seen:
                if seen[row] != (trunc, occ):
                    raise ValueError("conflicting duplicate metadata")
                counts["duplicates"] += 1
                continue
            seen[row] = (trunc, occ)
            rows.append(row)
            objects.append(
                {
                    "label_row": len(rows) - 1,
                    "source_line": line_no,
                    "yolo": [float(v) for v in row.split()],
                    "occlusion": occ,
                    "truncation": trunc,
                    "original_xywh": [x, y, w, h],
                }
            )
            counts[f"occ_{occ}"] += 1
        for folder in ("images", "labels", "metadata"):
            (dest / folder / split).mkdir(parents=True, exist_ok=True)
        link = dest / "images" / split / name
        if not link.exists():
            if os.name == "nt":
                shutil.copy2(image, link)
            else:
                link.symlink_to(image.resolve())
        (dest / "labels" / split / f"{image.stem}.txt").write_text("\n".join(rows) + "\n" if rows else "")
        metadata = {
            "source": str(annotation.resolve()),
            "source_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "image_size": [width, height],
            "objects": objects,
            "ignored": ignored,
        }
        (dest / "metadata" / split / f"{image.stem}.json").write_text(json.dumps(metadata, indent=2))
        counts["images"] += 1
    return dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--split", default="train")
    args = p.parse_args()
    images = sorted(x.name for x in (args.source / "images").glob("*.jpg"))
    if not images:
        raise ValueError("No images")
    if args.pilot:
        random.Random(42).shuffle(images)
        groups = {"train": images[:8], "cal": images[8:12], "val": images[12:16]}
        if len(images) < 16:
            raise ValueError("pilot needs 16 images")
    else:
        groups = {args.split: images}
    report = {s: convert(args.source, args.output, s, items) for s, items in groups.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "conversion_report.json").write_text(
        json.dumps({"pilot": args.pilot, "seed": 42, "splits": report}, indent=2)
    )
    config = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": NAMES}
    (args.output / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if args.pilot:
        config["val"] = "images/cal"
        (args.output / "calibration.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
