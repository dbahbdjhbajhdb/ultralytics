"""Create separate development/calibration/held-out splits in a NEW staged output."""

import argparse, os, random, shutil
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output must be a new directory; original dataset remains unchanged")
    val = sorted((a.source / "images/val").glob("*"))
    val = [x for x in val if x.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    if len(val) < 5:
        raise ValueError("Need at least five validation images")
    random.Random(a.seed).shuffle(val)
    n = len(val)
    dev = max(1, n // 5)
    cal = max(1, 2 * n // 5)
    splits = {
        "train": list((a.source / "images/train").glob("*")),
        "val": val[:dev],
        "cal": val[dev : dev + cal],
        "test": val[dev + cal :],
    }
    for split, images in splits.items():
        source_split = "train" if split == "train" else "val"
        for image in images:
            if not image.is_file():
                continue
            for kind in ["images", "labels", "metadata"]:
                (a.output / kind / split).mkdir(parents=True, exist_ok=True)
            target = a.output / "images" / split / image.name
            if os.name == "nt":
                shutil.copy2(image, target)
            else:
                target.symlink_to(image.resolve())
            for kind, suffix in [("labels", ".txt"), ("metadata", ".json")]:
                shutil.copy2(
                    a.source / kind / source_split / (image.stem + suffix),
                    a.output / kind / split / (image.stem + suffix),
                )
    cfg = yaml.safe_load((a.source / "dataset.yaml").read_text())
    cfg["path"] = str(a.output.resolve())
    cfg["test"] = "images/test"
    (a.output / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    import json

    (a.output / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "origin": "split original validation into 20% development / 40% calibration / 40% internal held-out",
                "images": {s: [x.name for x in imgs] for s, imgs in splits.items()},
            },
            indent=2,
        )
    )
    print({s: len(imgs) for s, imgs in splits.items()})


if __name__ == "__main__":
    main()
