import argparse, os, sys
from pathlib import Path

root = Path(os.environ.get("OCC_EXPERIMENT_ROOT", str(Path(__file__).resolve().parents[2] / "runs/visdrone_occ")))
sys.path.insert(0, str(root))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/ultralytics-config")
os.environ.setdefault("MPLCONFIGDIR", "/workspace/.cache/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
from ultralytics.models.yolo.detect.occlusion_train import OcclusionTrainer

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(root / "data/processed/visdrone_pilot/dataset.yaml"))
p.add_argument("--weighted", action="store_true")
p.add_argument("--size-weight", type=float, default=1.0)
p.add_argument("--occ-weight", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--model", default="yolo11n.yaml")
p.add_argument("--device", default="cpu")
p.add_argument("--name", default=None)
args = p.parse_args()
torch.set_num_threads(2)
os.environ["WEIGHTED_LOSS"] = "1" if args.weighted else "0"
os.environ["SIZE_WEIGHT"] = str(args.size_weight)
os.environ["OCC_WEIGHT"] = str(args.occ_weight)
if min(args.size_weight, args.occ_weight) < 0:
    raise ValueError("Weights must be nonnegative")
trainer = OcclusionTrainer(
    overrides=dict(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=2,
        workers=0,
        device=args.device,
        amp=False,
        seed=42,
        deterministic=True,
        plots=False,
        save=True,
        project=str(root / "runs"),
        name=args.name or ("pilot_weighted" if args.weighted else "pilot_baseline"),
        exist_ok=False,
        pretrained=False,
        close_mosaic=0,
        optimizer="AdamW",
        lr0=0.001,
        val=True,
    )
)
trainer.save_dir.mkdir(parents=True, exist_ok=True)
import json

(trainer.save_dir / "occlusion_config.json").write_text(
    json.dumps(
        dict(
            weighted=args.weighted,
            size_weight=args.size_weight,
            occ_weight=args.occ_weight,
            augmentation="letterbox only",
            weight_cap=3,
            normalization="positive anchor mean",
        ),
        indent=2,
    )
)
trainer.train()
