import os
import shutil

"""Explicitly import six/seven-column labels without changing the source dataset."""
import argparse, json, hashlib
from pathlib import Path
from PIL import Image
import yaml


def convert(source, output, label_format):
    columns = {"yolo-occ": 6, "yolo-trunc-occ": 7}[label_format]
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    counts = {}
    for split in ("train", "val", "cal", "test"):
        folder = source / "images" / split
        if not folder.exists():
            continue
        counts[split] = 0
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            annotation = source / "labels" / split / (image.stem + ".txt")
            objects = []
            rows = []
            seen = {}
            for lineno, line in enumerate(annotation.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                v = list(map(float, line.split()))
                if len(v) != columns:
                    raise ValueError(f"{annotation}:{lineno}: expected {columns} columns")
                cls, cx, cy, w, h = v[:5]
                occ = v[-1]
                trunc = v[5] if columns == 7 else None
                if cls != int(cls) or not 0 <= cls < 10 or occ not in (0.0, 1.0, 2.0):
                    raise ValueError("invalid class or occlusion")
                if trunc is not None and trunc not in (0.0, 1.0, 2.0):
                    raise ValueError("invalid truncation")
                if not all(0 <= a <= 1 for a in [cx, cy, w, h]) or w <= 0 or h <= 0:
                    raise ValueError("invalid normalized coordinates")
                row = f"{int(cls)} " + " ".join(f"{x:.8f}" for x in [cx, cy, w, h])
                if row in seen:
                    if seen[row] != (occ, trunc):
                        raise ValueError("duplicate box with conflicting metadata")
                    continue
                seen[row] = (occ, trunc)
                rows.append(row)
                objects.append(
                    {
                        "label_row": len(rows) - 1,
                        "source_line": lineno,
                        "yolo": [float(x) for x in row.split()],
                        "occlusion": int(occ),
                        "truncation": None if trunc is None else int(trunc),
                    }
                )
            for kind in ["images", "labels", "metadata"]:
                (output / kind / split).mkdir(parents=True, exist_ok=True)
            link = output / "images" / split / image.name
            if link.is_symlink() and link.resolve() != image.resolve():
                raise ValueError("existing symlink belongs to a different input")
            if not link.exists():
                if os.name == "nt":
                    shutil.copy2(image, link)
                else:
                    link.symlink_to(image.resolve())
            (output / "labels" / split / (image.stem + ".txt")).write_text("\n".join(rows) + "\n" if rows else "")
            with Image.open(image) as im:
                size = list(im.size)
            (output / "metadata" / split / (image.stem + ".json")).write_text(
                json.dumps(
                    {
                        "source": str(annotation.resolve()),
                        "source_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
                        "image_size": size,
                        "objects": objects,
                        "ignored": [],
                    },
                    indent=2,
                )
            )
            counts[split] += 1
    if not counts:
        raise ValueError("Expected images/{train,val,cal,test} and labels/{split}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": names},
            sort_keys=False,
        )
    )
    (output / "import_report.json").write_text(
        json.dumps(
            {
                "label_format": label_format,
                "images": counts,
                "ignore_regions": "not recoverable unless preserved in original input",
            },
            indent=2,
        )
    )
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label-format", choices=["yolo-occ", "yolo-trunc-occ"], required=True)
    args = p.parse_args()
    print(convert(args.source, args.output, args.label_format))
