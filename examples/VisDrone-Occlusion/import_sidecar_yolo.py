import os
import shutil

"""Stage five-column YOLO labels plus visdrone-yolo-occlusion-v1 JSON sidecars."""
import argparse, hashlib, json
from pathlib import Path
import yaml
from PIL import Image


def convert(source, output, metadata_root=None, flat_split="train"):
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    count = {}
    flat = any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in (source / "images").glob("*"))
    for split in [flat_split] if flat else ["train", "val", "cal", "test"]:
        images = source / "images" if flat else source / "images" / split
        if not images.exists():
            continue
        count[split] = 0
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            label = (source / "labels" if flat else source / "labels" / split) / (image.stem + ".txt")
            roots = [metadata_root] if metadata_root else [source / "metadata", source / "occlusion"]
            choices = [folder / (image.stem + ".json") for root in roots for folder in [root / split, root]]
            choices.append(label.with_suffix(".json"))
            meta = next((p for p in choices if p.exists()), None)
            if meta is None:
                raise ValueError(f"Missing JSON for {label}")
            d = json.loads(meta.read_text())
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            if any(len(row) != 5 for row in rows):
                raise ValueError("Expected five-column YOLO labels")
            if d.get("label_sha256") and hashlib.sha256(label.read_bytes()).hexdigest() != d["label_sha256"]:
                raise ValueError(f"Hash mismatch: {label}")
            if len(d["objects"]) != len(rows):
                raise ValueError(f"Row count mismatch: {label}")
            with Image.open(image) as im:
                if d.get("image_size_wh", list(im.size)) != list(im.size):
                    raise ValueError("Image dimensions mismatch")
            for index, (row, obj) in enumerate(zip(rows, d["objects"]), 1):
                values = [float(v) for v in row]
                expected = obj.get("yolo", [obj.get("class_id"), *obj.get("bbox_xywh_normalized", [])])
                if len(expected) != 5 or any(abs(a - b) > 1e-7 for a, b in zip(values, expected)):
                    raise ValueError(f"Row mismatch: {label}:{index}")
                if "schema" in d and obj.get("label_row") != index:
                    raise ValueError("Expected one-based ordered label_row")
                if not obj.get("occ_valid", True) or obj.get("occlusion") not in [0, 1, 2]:
                    raise ValueError("Invalid occlusion")
            for kind in ["images", "labels", "metadata"]:
                (output / kind / split).mkdir(parents=True, exist_ok=True)
            target = output / "images" / split / image.name
            if target.is_symlink() and target.resolve() != image.resolve():
                raise ValueError("Existing image belongs to another source")
            if not target.exists():
                if os.name == "nt":
                    shutil.copy2(image, target)
                else:
                    target.symlink_to(image.resolve())
            shutil.copyfile(label, output / "labels" / split / label.name)
            shutil.copyfile(meta, output / "metadata" / split / meta.name)
            count[split] += 1
    if not any(count.values()):
        raise ValueError("Expected images/split and labels/split structure")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": names},
            sort_keys=False,
        )
    )
    (output / "import_report.json").write_text(
        json.dumps({"images": count, "schema": "YOLO five-column plus verified occlusion sidecars"}, indent=2)
    )
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata-root", type=Path)
    p.add_argument(
        "--split",
        choices=["train", "val", "cal", "test"],
        default="train",
        help="Destination split for a flat images/labels/occlusion folder",
    )
    a = p.parse_args()
    print(convert(a.source, a.output, a.metadata_root, a.split))
