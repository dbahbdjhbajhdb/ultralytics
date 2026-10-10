# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Install occlusion supervision and P3 encoding fusion into an existing ZIP, without training."""

import argparse
import hashlib
import json
import re
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

FILES = {
    "ultralytics/data/occlusion.py": "bdfb9ae0e422b461cbae18caab17f8b51ac034d5d2088a68d641fbe5fb97be5e",
    "ultralytics/nn/modules/occlusion_aux.py": "1ba816c341960fdf1fce576feec175b60a3c17d4da9953cc6f49c02e063c131d",
    "ultralytics/utils/occlusion_supervision_loss.py": "eca4cb4f16a0e72a302133c5c38bdb8312f31c91e7299972342a9d62e0ac760f",
    "ultralytics/models/yolo/detect/occlusion_supervision.py": "5aa3ef8c19cf78262b0b915e409862c3ea3aba780cf06cc8f5a9f43955562781",
    "examples/VisDrone-Occlusion/train_visdrone.py": "51c5a861b7577ac171f589d1e00b1a6fbb2b7b85c53df138004ca9eb101ecc6c",
    "examples/VisDrone-Occlusion/train_occlusion_supervised.py": "402a2592474c81f67cf51b1700fe96c06ce7433a0cf4445351fca88b7c75caee",
    "examples/VisDrone-Occlusion/README_supervision.md": "31efee9e1f1a41e6064e98610b692076f1ed9ff8431ddde0535720f91e661980",
    "examples/VisDrone-Occlusion/README_fusion.md": "f9bcd7f22299691b586357d96ff1cff4423ebb0a2923f44d10cc525ba876f77b",
}

REQUIRED = (
    "pyproject.toml",
    "ultralytics/nn/tasks.py",
    "ultralytics/models/yolo/detect/occlusion_train.py",
    "examples/VisDrone-Occlusion/train_visdrone.py",
)


def resolve_repository(path):
    """Accept either the project directory or its outer ZIP extraction directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")

    def is_repository(candidate):
        return all((candidate / name).is_file() for name in REQUIRED)

    if is_repository(root):
        return root
    candidates = [child for child in root.iterdir() if child.is_dir() and is_repository(child)]
    if len(candidates) != 1:
        raise ValueError("Run from the project root or its single outer ZIP directory; use --root for an explicit path")
    return candidates[0].resolve()


def download_file(ref, name):
    url = f"https://raw.githubusercontent.com/dbahbdjhbajhdb/ultralytics/{ref}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "VisDrone-occlusion-supervision-updater"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def apply_update(path, ref, fetch=download_file):
    """Verify the complete payload before backup/write; restore partial writes on failure."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        raise ValueError("--ref must be the full immutable 40-character Git commit SHA")
    root = resolve_repository(path)
    payloads, previous = {}, {}
    for name, expected in FILES.items():
        target = root / name
        # Fail before downloading/writing if an existing path escapes through a symlink.
        try:
            target.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"Update target is outside the project: {name}") from error
        if target.exists() and not target.is_file():
            raise ValueError(f"Update target is not a file: {name}")
        contents = fetch(ref, name)
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch; no project files changed: {name}")
        if name.endswith(".py"):
            compile(contents, name, "exec")  # Syntax only; never imports torch or model code.
        old = target.read_bytes() if target.exists() else None
        if contents != old:
            payloads[name], previous[name] = contents, old
    if not payloads:
        print("ALREADY_UP_TO_DATE")
        print(f"ROOT: {root}")
        return root, None

    backup = root / ("occ_supervision_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    backup.mkdir()
    for name, contents in previous.items():
        if contents is not None:
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
    (backup / "manifest.json").write_text(
        json.dumps(
            {"ref": ref, "previously_existed": {name: old is not None for name, old in previous.items()}}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    touched = []
    try:
        for name, contents in payloads.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            touched.append(name)
            target.write_bytes(contents)
    except BaseException:
        recovery_errors = []
        for name in reversed(touched):
            target = root / name
            try:
                if previous[name] is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.write_bytes(previous[name])
            except OSError as error:
                recovery_errors.append(f"{name}: {error}")
        if recovery_errors:
            print(f"RESTORE_ERRORS; original files are backed up in {backup}")
            for error in recovery_errors:
                print(error)
        raise
    print("UPDATE_APPLIED")
    print(f"ROOT: {root}")
    print(f"Backup: {backup}")
    print("No model, training, evaluation or dependency installation was run.")
    return root, backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    apply_update(args.root, args.ref)


if __name__ == "__main__":
    main()
