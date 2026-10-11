# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train VisDrone with grade-guided paired occlusion and local feature compensation.

Only standard-library modules are imported until training starts. Use ``--help``
or ``--print-config`` to inspect the configuration without importing PyTorch.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    """Define the separate paired-training entry point and its controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(REPOSITORY_ROOT / "examples/VisDrone-Occlusion/visdrone_occ.yaml"))
    parser.add_argument(
        "--model", default="yolo11n.yaml", help="Standard YOLO11 YAML; the default trains from scratch."
    )
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint to initialize a model YAML.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4, help="Fixed positive batch size; AutoBatch is not supported.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--optimizer", choices=("Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"), default="AdamW"
    )
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--name", default="occ_pair_full_scratch_200")
    parser.add_argument("--threads", type=int, default=2, help="CPU thread count used by PyTorch.")
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument("--deterministic", dest="deterministic", action="store_true")
    deterministic.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=False)
    augment = parser.add_mutually_exclusive_group()
    augment.add_argument("--safe-augment", dest="safe_augment", action="store_true")
    augment.add_argument("--no-safe-augment", dest="safe_augment", action="store_false")
    parser.set_defaults(safe_augment=True)

    parser.add_argument("--pair-weight", type=float, default=0.05, help="Paired feature loss coefficient.")
    parser.add_argument(
        "--pair-warmup", type=int, default=10, help="Initial epochs without paired augmentation/loss."
    )
    parser.add_argument(
        "--pair-ramp", type=int, default=10, help="Epochs over which the paired loss is introduced."
    )
    parser.add_argument(
        "--pair-prob", type=float, default=0.5, help="Probability of attempting paired occlusion per image."
    )
    parser.add_argument(
        "--coverage-min", type=float, default=0.15, help="Minimum synthetic coverage of a selected GT box."
    )
    parser.add_argument(
        "--coverage-max", type=float, default=0.35, help="Maximum synthetic coverage of a selected GT box."
    )
    parser.add_argument(
        "--neighbor-max-coverage", type=float, default=0.35, help="Maximum allowed coverage of every neighboring GT box."
    )
    parser.add_argument(
        "--teacher-conf", type=float, default=0.15, help="Minimum teacher class confidence for a GT pair."
    )
    parser.add_argument(
        "--teacher-iou", type=float, default=0.5, help="Minimum teacher/GT IoU for a reliable reference."
    )
    parser.add_argument(
        "--pair-hidden", type=int, default=32, help="Hidden width of the local compensation module."
    )
    parser.add_argument(
        "--pair-roi-size", type=int, default=3, help="Sampling grid side length in each paired GT region."
    )
    parser.add_argument(
        "--pair-selection", choices=("grade", "random"), default="grade", help="Grade-0 selection or label-agnostic control."
    )
    parser.add_argument(
        "--no-compensation", action="store_true", help="Keep the original detection head without the module."
    )
    parser.add_argument(
        "--no-pairing", action="store_true", help="Disable paired augmentation, teacher forward and pair loss."
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print validated configuration and exit without training."
    )
    return parser


def build_configuration(args):
    """Validate flags and return the trainer arguments using only the standard library."""
    model = args.model
    suffix = Path(model).suffix.lower()
    if suffix not in {".yaml", ".yml", ".pt"}:
        raise ValueError("--model must be a model YAML or a .pt checkpoint")
    if args.pretrained and suffix not in {".yaml", ".yml"}:
        raise ValueError("--pretrained initializes a YAML; --model *.pt already selects a checkpoint")
    if "goa" in Path(model).stem.lower():
        raise ValueError("This experiment uses standard YOLO11 without GOA; use --model yolo11n.yaml")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("--epochs and --batch must be positive; --imgsz must be at least 32")
    if min(args.workers, args.seed, args.patience, args.pair_warmup, args.pair_ramp) < 0:
        raise ValueError("Workers, seed, patience, pair warmup and pair ramp must be nonnegative")
    if min(args.threads, args.max_det, args.pair_hidden, args.pair_roi_size) < 1:
        raise ValueError("Threads, max detections, hidden width and ROI size must be positive")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive")
    if not math.isfinite(args.pair_weight) or args.pair_weight < 0:
        raise ValueError("--pair-weight must be finite and nonnegative")
    if not math.isfinite(args.pair_prob) or not 0 <= args.pair_prob <= 1:
        raise ValueError("--pair-prob must be within [0, 1]")
    if not all(math.isfinite(value) for value in (args.coverage_min, args.coverage_max)):
        raise ValueError("Synthetic coverage values must be finite")
    if not 0 < args.coverage_min <= args.coverage_max < 0.5:
        raise ValueError("Synthetic coverage must satisfy 0 < minimum <= maximum < 0.5")
    if not math.isfinite(args.neighbor_max_coverage) or not 0 < args.neighbor_max_coverage < 0.5:
        raise ValueError("--neighbor-max-coverage must be within (0, 0.5)")
    if not math.isfinite(args.teacher_conf) or not 0 < args.teacher_conf <= 1:
        raise ValueError("--teacher-conf must be within (0, 1]")
    if not math.isfinite(args.teacher_iou) or not 0 < args.teacher_iou <= 1:
        raise ValueError("--teacher-iou must be within (0, 1]")

    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    overrides = {
        "model": model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "patience": args.patience,
        "max_det": args.max_det,
        "project": str(root / "runs"),
        "name": args.name,
        "exist_ok": False,
        "pretrained": args.pretrained or False,
        "close_mosaic": 0,
        "plots": False,
        "save": True,
        "val": True,
    }
    pair_config = {
        "coefficient": args.pair_weight,
        "warmup_epochs": args.pair_warmup,
        "ramp_epochs": args.pair_ramp,
        "pair_probability": args.pair_prob,
        "coverage_min": args.coverage_min,
        "coverage_max": args.coverage_max,
        "neighbor_max_coverage": args.neighbor_max_coverage,
        "teacher_conf": args.teacher_conf,
        "teacher_iou": args.teacher_iou,
        "hidden": args.pair_hidden,
        "roi_size": args.pair_roi_size,
        "selection": args.pair_selection,
        "compensation": not args.no_compensation,
        "pairing": not args.no_pairing,
    }
    return overrides, pair_config


def main(argv=None):
    """Write the resolved run parameters and start the separate paired trainer."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides, pair_config = build_configuration(args)
    except ValueError as error:
        parser.error(str(error))
    if args.print_config:
        print(json.dumps({"overrides": overrides, "pair_config": pair_config}, indent=2, ensure_ascii=False))
        return None

    root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    sys.path.insert(0, str(REPOSITORY_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "runs/yolo_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "runs/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import torch

    from ultralytics.models.yolo.detect.occlusion_paired import PairedOcclusionTrainer

    torch.set_num_threads(args.threads)
    trainer = PairedOcclusionTrainer(overrides=overrides, pair_config=pair_config, safe_augment=args.safe_augment)
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    (trainer.save_dir / "run_parameters.json").write_text(
        json.dumps(
            {
                "cli": vars(args),
                "overrides": overrides,
                "resolved_trainer_args": vars(trainer.args),
                "pair_config": pair_config,
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
