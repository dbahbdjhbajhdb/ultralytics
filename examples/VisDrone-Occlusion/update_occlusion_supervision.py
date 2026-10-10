# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Install direct occlusion supervision into an existing ZIP, without training."""

import argparse
import hashlib
import json
import re
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

FILES = {
    "ultralytics/data/occlusion.py": "e09eb667e5643545891f8d7ca74b608f35989dc598c679229bede9495b9e4190",
    "ultralytics/nn/modules/occlusion_aux.py": "30a576e149ca1c529dd3161583a9e1fbf1904c2dcbf1895c97cc78b9fa0b38f4",
    "ultralytics/utils/occlusion_supervision_loss.py": "528397b361d5db9e51e56a414d6350f8a524b8176c28ca645c0e77520c420fa3",
    "ultralytics/models/yolo/detect/occlusion_supervision.py": "7d96e13000206e72054af77b6f1c0071ba1ce7bd10b9a8e1377d10bc6d48035a",
    "examples/VisDrone-Occlusion/train_occlusion_supervised.py": "8d74d2adbe082e8dfdb98512ef9a45329fe6e1c484a324e387c0071a1119cc05",
    "examples/VisDrone-Occlusion/README_supervision.md": "a89e5d121b03e54e282f2ae3d48c27302e50deac6fefc1167f6a74c564cb6ef2",
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
