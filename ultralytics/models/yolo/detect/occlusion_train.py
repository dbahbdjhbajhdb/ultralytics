# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Opt-in trainer for progressive size/occlusion weighting; default YOLO training is unchanged."""

import json
import math
from copy import deepcopy

from ultralytics.data.occlusion import OcclusionDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.occlusion_loss import WeightedDetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


def weight_progress(epoch, warmup_epochs=5, ramp_epochs=10):
    """Return the bounded weighting factor for a zero-based training epoch."""
    if min(warmup_epochs, ramp_epochs) < 0:
        raise ValueError("Weight warm-up and ramp durations must be nonnegative")
    if epoch < warmup_epochs:
        return 0.0
    return 1.0 if ramp_epochs == 0 else min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


class OcclusionModel(DetectionModel):
    def init_criterion(self):
        return WeightedDetectionLoss(self)


class OcclusionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, occlusion_config=None):
        self.occlusion_config = {
            "enabled": False,
            "size_weight": 0.5,
            "occ_weight": 0.5,
            "joint_weight": 0.0,
            "size_reference": 32.0,
            "cap": 2.0,
            "warmup_epochs": 5,
            "ramp_epochs": 10,
            "safe_augment": True,
        }
        if occlusion_config is not None:
            unknown = set(occlusion_config) - set(self.occlusion_config)
            if unknown:
                raise ValueError(f"Unknown occlusion options: {sorted(unknown)}")
            self.occlusion_config.update(occlusion_config)
        for key in ("size_weight", "occ_weight", "joint_weight", "cap", "size_reference"):
            value = float(self.occlusion_config[key])
            if not math.isfinite(value) or value < (1 if key == "cap" else 0):
                raise ValueError(f"Invalid occlusion option: {key}")
            if key == "size_reference" and value == 0:
                raise ValueError("size_reference must be positive")
        for key in ("warmup_epochs", "ramp_epochs"):
            value = self.occlusion_config[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        device = (overrides or {}).get("device", "cpu")
        if isinstance(device, (list, tuple)) and len(device) > 1 or isinstance(device, str) and "," in device:
            raise ValueError(
                "This experiment trainer currently supports one device; multi-GPU config propagation is not implemented"
            )
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_pretrain_routine_end", self._save_experiment_config)
        self.add_callback("on_train_epoch_start", self._start_weight_epoch)
        self.add_callback("on_train_epoch_end", self._save_weight_statistics)

    def build_dataset(self, img_path, mode="train", batch=None):
        return OcclusionDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            safe_augment=self.occlusion_config["safe_augment"],
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or False,
            single_cls=False,
            stride=32,
            pad=0.0 if mode == "train" else 0.5,
            task="detect",
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        cls = OcclusionModel if self.occlusion_config["enabled"] else DetectionModel
        model = cls(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model.occlusion_config = deepcopy(self.occlusion_config)
        model.occlusion_config["progress"] = weight_progress(
            0, self.occlusion_config["warmup_epochs"], self.occlusion_config["ramp_epochs"]
        )
        if weights:
            model.load(weights)
        return model

    def _save_experiment_config(self, trainer):
        payload = deepcopy(self.occlusion_config)
        payload.update(
            normalization="original loss denominator; no batch mean cancellation",
            training_transforms="letterbox + HSV + flips" if payload["safe_augment"] else "letterbox only",
            validation_transforms="letterbox only",
            image_size=self.args.imgsz,
            seed=self.args.seed,
            max_det=self.args.max_det,
            model=self.args.model,
            pretrained=self.args.pretrained,
            orthogonal_attention_layers=[
                module.i for module in unwrap_model(self.model).model if type(module).__name__ == "C3k2GOA"
            ],
        )
        (self.save_dir / "occlusion_config.json").write_text(json.dumps(payload, indent=2))

    def _start_weight_epoch(self, trainer):
        progress = weight_progress(
            self.epoch, self.occlusion_config["warmup_epochs"], self.occlusion_config["ramp_epochs"]
        )
        for model in (unwrap_model(self.model), self.ema.ema if self.ema else None):
            if model is None:
                continue
            model.occlusion_config["progress"] = progress
            criterion = getattr(model, "criterion", None)
            if isinstance(criterion, WeightedDetectionLoss):
                criterion.set_progress(progress)
                criterion.reset_statistics()

    def _save_weight_statistics(self, trainer):
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        if not isinstance(criterion, WeightedDetectionLoss):
            return
        statistics = deepcopy(criterion.statistics)
        statistics["epoch"] = self.epoch + 1
        statistics["progress"] = criterion.progress
        count = statistics["positive_anchors"]
        statistics["mean_positive_weight"] = statistics["weight_sum"] / count if count else None
        with (self.save_dir / "weight_diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(statistics) + "\n")
