# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train a standard detector with optional direct, GT-region occlusion supervision.

This guarded entry point does not import torch or start training on import.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(root / "examples/VisDrone-Occlusion/visdrone_occ.yaml"))
    parser.add_argument("--model", default="yolo11n.yaml", help="Standard detector YAML; no GOA in this experiment.")
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint to initialize a model YAML.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--optimizer", choices=("Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"), default="AdamW"
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--aux-weight", type=float, default=0.1, help="Auxiliary CE coefficient; 0 gives a matched baseline."
    )
    parser.add_argument("--aux-warmup", type=int, default=5)
    parser.add_argument("--aux-ramp", type=int, default=10)
    parser.add_argument("--aux-hidden", type=int, default=64)
    parser.add_argument("--aux-roi-size", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    determinism = parser.add_mutually_exclusive_group()
    determinism.add_argument("--deterministic", dest="deterministic", action="store_true")
    determinism.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--safe-augment", dest="safe_augment", action="store_true")
    augmentation.add_argument("--no-safe-augment", dest="safe_augment", action="store_false")
    parser.set_defaults(amp=True, deterministic=False, safe_augment=True)
    return parser


def build_configuration(args):
    if args.pretrained and Path(args.model).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--pretrained initializes model YAMLs; a .pt --model already selects its checkpoint")
    if "goa" in Path(args.model).stem.lower():
        raise ValueError("Use a standard detector YAML, e.g. yolo11n.yaml, for this experiment without GOA")
    if args.epochs < 1 or args.imgsz < 32:
        raise ValueError("--epochs must be positive and --imgsz must be at least 32")
    if args.batch < 1:
        raise ValueError("--batch must be a positive integer; AutoBatch does not profile the GT-region auxiliary loss")
    if min(args.workers, args.seed, args.patience, args.aux_warmup, args.aux_ramp) < 0:
        raise ValueError("Workers, seed, patience and auxiliary schedule must be nonnegative")
    if min(args.threads, args.max_det, args.aux_hidden, args.aux_roi_size) < 1:
        raise ValueError("Threads, max-det, auxiliary width and ROI size must be positive")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        raise ValueError("--aux-weight must be finite and nonnegative")
    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    auxiliary_config = {
        "coefficient": args.aux_weight,
        "hidden": args.aux_hidden,
        "roi_size": args.aux_roi_size,
        "warmup_epochs": args.aux_warmup,
        "ramp_epochs": args.aux_ramp,
    }
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "max_det": args.max_det,
        "project": str(root / "runs"),
        "name": args.name or ("occ_supervised_scratch_200" if args.aux_weight else "occ_supervised_baseline_200"),
        "exist_ok": False,
        "pretrained": args.pretrained or False,
        "plots": False,
        "save": True,
        "val": True,
        # This experiment uses only letterbox / color / flips. Avoid a late-epoch
        # mosaic-close callback rebuilding transforms for a mosaic that was never used.
        "close_mosaic": 0,
    }
    return overrides, auxiliary_config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides, auxiliary_config = build_configuration(args)
    except ValueError as error:
        parser.error(str(error))
    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    sys.path.insert(0, str(REPOSITORY_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "runs/yolo_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "runs/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import torch

    from ultralytics.models.yolo.detect.occlusion_supervision import OcclusionSupervisedTrainer

    torch.set_num_threads(args.threads)
    trainer = OcclusionSupervisedTrainer(
        overrides=overrides, auxiliary_config=auxiliary_config, safe_augment=args.safe_augment
    )
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    (trainer.save_dir / "run_parameters.json").write_text(
        json.dumps(
            {
                "cli": vars(args),
                "overrides": overrides,
                "resolved_trainer_args": vars(trainer.args),
                "auxiliary_config": auxiliary_config,
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
