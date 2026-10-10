# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train standard detection features with a separate, GT-region occlusion task."""

import json
import math
from copy import deepcopy

import torch

from ultralytics.models.yolo.detect.occlusion_train import OcclusionTrainer, weight_progress
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.occlusion_aux import OcclusionAuxDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.occlusion_supervision_loss import OcclusionSupervisionLoss
from ultralytics.utils.torch_utils import unwrap_model


class OcclusionSupervisedModel(DetectionModel):
    """Keep the standard detector and add a training-only occlusion classifier."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True, auxiliary_config=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if type(self.model[-1]) is not Detect or self.end2end:
            raise ValueError("Occlusion supervision currently supports a standard, non-end-to-end Detect head")
        if any(type(module).__name__ == "C3k2GOA" for module in self.model.modules()):
            raise ValueError("Use a standard detector YAML, e.g. yolo11n.yaml, for this experiment without GOA")
        self.occlusion_aux_config = {
            "coefficient": 0.1,
            "hidden": 64,
            "roi_size": 3,
            "warmup_epochs": 5,
            "ramp_epochs": 10,
            "progress": 0.0,
        }
        if auxiliary_config is not None:
            self.occlusion_aux_config.update(auxiliary_config)
        # Build the standard detector first. Copy its head weights and restore the
        # RNG state after auxiliary initialization, keeping baseline initialization
        # and the subsequent CPU random stream identical for the same seed.
        with torch.random.fork_rng(devices=[]):
            self.model[-1] = OcclusionAuxDetect(
                self.model[-1],
                hidden=self.occlusion_aux_config["hidden"],
                roi_size=self.occlusion_aux_config["roi_size"],
            )
        self.stride = self.model[-1].stride

    def init_criterion(self):
        return OcclusionSupervisionLoss(self)


class OcclusionSupervisedTrainer(OcclusionTrainer):
    """Reuse metadata-safe datasets; replace weighted detection with auxiliary CE."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, auxiliary_config=None, safe_augment=True):
        self.auxiliary_config = {
            "coefficient": 0.1,
            "hidden": 64,
            "roi_size": 3,
            "warmup_epochs": 5,
            "ramp_epochs": 10,
        }
        if auxiliary_config is not None:
            unknown = set(auxiliary_config) - set(self.auxiliary_config)
            if unknown:
                raise ValueError(f"Unknown auxiliary options: {sorted(unknown)}")
            self.auxiliary_config.update(auxiliary_config)
        coefficient = float(self.auxiliary_config["coefficient"])
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Auxiliary coefficient must be finite and nonnegative")
        for key in ("hidden", "roi_size", "warmup_epochs", "ramp_epochs"):
            value = self.auxiliary_config[key]
            minimum = 1 if key in {"hidden", "roi_size"} else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Invalid auxiliary option: {key}")
        batch_size = (overrides or {}).get("batch", 16)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("Use a fixed positive batch size; AutoBatch does not profile GT-region auxiliary training")
        # Parent callbacks bind the overridden methods below. Its weighting stays
        # disabled; only the metadata reader and controlled augmentation are reused.
        super().__init__(
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks,
            occlusion_config={"enabled": False, "safe_augment": bool(safe_augment)},
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        config = deepcopy(self.auxiliary_config)
        config["progress"] = weight_progress(0, config["warmup_epochs"], config["ramp_epochs"])
        model = OcclusionSupervisedModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            auxiliary_config=config,
        )
        if weights:
            model.load(weights)
        LOGGER.info(
            "Occlusion supervision: standard detection loss + "
            f"{config['coefficient']:g} * progress * GT-region CE; no GOA or detection-loss reweighting"
        )
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "occ_loss"
        return validator

    def _save_experiment_config(self, trainer):
        payload = deepcopy(self.auxiliary_config)
        payload.update(
            enabled=payload["coefficient"] > 0,
            objective="standard detection loss + coefficient * progress * mean GT occlusion cross entropy",
            supervision="all known GT regions, independent of detection positive assignment",
            feature_levels="Detect inputs: P3/P4/P5 for standard YOLO11",
            training_transforms="letterbox + HSV + flips"
            if self.occlusion_config["safe_augment"]
            else "letterbox only",
            validation_auxiliary_loss="not computed; val/occ_loss is zero",
            seed=self.args.seed,
            image_size=self.args.imgsz,
            model=self.args.model,
            pretrained=self.args.pretrained,
            inference_requires_occlusion_labels=False,
        )
        (self.save_dir / "occlusion_supervision_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _start_weight_epoch(self, trainer):
        progress = weight_progress(
            self.epoch, self.auxiliary_config["warmup_epochs"], self.auxiliary_config["ramp_epochs"]
        )
        for model in (unwrap_model(self.model), self.ema.ema if self.ema else None):
            if model is None:
                continue
            model.occlusion_aux_config["progress"] = progress
            criterion = getattr(model, "criterion", None)
            if isinstance(criterion, OcclusionSupervisionLoss):
                criterion.set_progress(progress)
                criterion.reset_statistics()

    def _save_weight_statistics(self, trainer):
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        if not isinstance(criterion, OcclusionSupervisionLoss):
            return
        statistics = deepcopy(criterion.statistics)
        count = statistics["supervised_gt"]
        batches = statistics["batch_count"]
        statistics.update(
            epoch=self.epoch + 1,
            progress=criterion.progress,
            coefficient=criterion.coefficient,
            raw_mean_ce=statistics["ce_sum"] / count if count else None,
            mean_weighted_aux_loss=statistics["weighted_aux_sum"] / batches if batches else None,
            accuracy_by_occlusion=[
                correct / total if total else None
                for correct, total in zip(statistics["correct_by_occlusion"], statistics["evaluated_gt_by_occlusion"])
            ],
        )
        with (self.save_dir / "occlusion_supervision_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(statistics, ensure_ascii=False) + "\n")
