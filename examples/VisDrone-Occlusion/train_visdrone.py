# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train matched baseline and size/occlusion-weighted VisDrone experiments.

Importing this module only defines the command-line interface. Training and heavy
imports happen under ``main`` so that Windows DataLoader workers can import it.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    """Build the reusable CLI without importing PyTorch or starting training."""
    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(root / "data/processed/visdrone_pilot/dataset.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="A pretrained checkpoint or model YAML.")
    parser.add_argument("--pretrained", default=None, help="Checkpoint to initialize a model YAML, e.g. yolo11n.pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Use 0 for the first CUDA GPU, or cpu.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size; -1 enables Ultralytics auto-batch.")
    parser.add_argument("--workers", type=int, default=4)
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", help="Enable supported mixed precision.")
    amp.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--optimizer",
        choices=("Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"),
        default="AdamW",
    )
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience; 0 disables it.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum validation detections per image.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--weighted", action="store_true", help="Enable the size/occlusion loss prototype.")
    parser.add_argument("--size-weight", type=float, default=0.5)
    parser.add_argument("--occ-weight", type=float, default=0.5)
    parser.add_argument(
        "--joint-weight", type=float, default=0.0, help="Additional weight for small AND occluded targets."
    )
    parser.add_argument(
        "--size-reference", type=float, default=32.0, help="Reference target size in model-input pixels."
    )
    parser.add_argument("--weight-cap", type=float, default=2.0)
    parser.add_argument("--weight-warmup", type=int, default=5, help="Initial epochs with baseline weights.")
    parser.add_argument("--weight-ramp", type=int, default=10, help="Epochs used to ramp up weighting.")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--safe-augment",
        dest="safe_augment",
        action="store_true",
        help="Use metadata-preserving color and flip augmentation.",
    )
    augmentation.add_argument(
        "--no-safe-augment",
        dest="safe_augment",
        action="store_false",
        help="Use letterbox only, without training augmentation.",
    )
    parser.set_defaults(safe_augment=True)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU thread count.")
    return parser


def build_configuration(args):
    """Validate CLI values and produce explicit trainer and weighting settings."""
    if args.pretrained and Path(args.model).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--pretrained initializes model YAMLs; a .pt --model already selects its own checkpoint")
    if args.epochs < 1 or args.imgsz < 32:
        raise ValueError("--epochs must be positive and --imgsz must be at least 32")
    if args.batch < -1 or args.batch == 0:
        raise ValueError("--batch must be positive or -1 for auto-batch")
    if min(args.workers, args.seed, args.patience, args.weight_warmup, args.weight_ramp) < 0:
        raise ValueError("Workers, seed, patience, warmup and ramp values must be nonnegative")
    if args.threads < 1 or args.max_det < 1:
        raise ValueError("--threads and --max-det must be positive")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.size_weight, args.occ_weight, args.joint_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    if not math.isfinite(args.size_reference) or args.size_reference <= 0:
        raise ValueError("--size-reference must be finite and positive")
    if not math.isfinite(args.weight_cap) or args.weight_cap < 1:
        raise ValueError("--weight-cap must be finite and at least 1")

    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    occlusion_config = {
        "enabled": args.weighted,
        "size_weight": args.size_weight,
        "occ_weight": args.occ_weight,
        "joint_weight": args.joint_weight,
        "size_reference": args.size_reference,
        "cap": args.weight_cap,
        "warmup_epochs": args.weight_warmup,
        "ramp_epochs": args.weight_ramp,
        "safe_augment": args.safe_augment,
    }
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": True,
        "plots": False,
        "save": True,
        "project": str(root / "runs"),
        "name": args.name or ("visdrone_weighted" if args.weighted else "visdrone_baseline"),
        "exist_ok": False,
        "pretrained": args.pretrained or True,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "patience": args.patience,
        "max_det": args.max_det,
        "val": True,
    }
    return overrides, occlusion_config


def main(argv=None):
    """Configure the local runtime, save reproducibility metadata, and train."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides, occlusion_config = build_configuration(args)
    except ValueError as error:
        parser.error(str(error))

    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    sys.path.insert(0, str(REPOSITORY_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "runs/yolo_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "runs/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import torch

    from ultralytics.models.yolo.detect.occlusion_train import OcclusionTrainer

    torch.set_num_threads(args.threads)
    trainer = OcclusionTrainer(overrides=overrides, occlusion_config=occlusion_config)
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    (trainer.save_dir / "occlusion_config.json").write_text(
        json.dumps(occlusion_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (trainer.save_dir / "run_parameters.json").write_text(
        json.dumps(
            {
                "cli": vars(args),
                "overrides": overrides,
                "resolved_trainer_args": vars(trainer.args),
                "occlusion_config": occlusion_config,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    trainer.train()
    return trainer


if __name__ == "__main__":
    main()
