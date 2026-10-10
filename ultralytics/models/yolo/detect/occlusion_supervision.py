# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train occlusion-aware detection with an optional P3 encoding and GT-region task."""

import json
import math
from collections.abc import Mapping
from copy import deepcopy

import torch

from ultralytics.models.yolo.detect.occlusion_train import OcclusionTrainer, weight_progress
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.occlusion_aux import OcclusionAuxDetect, OcclusionFusionDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.occlusion_supervision_loss import OcclusionSupervisionLoss
from ultralytics.utils.torch_utils import unwrap_model


def _validate_auxiliary_config(auxiliary_config=None, include_progress=False):
    """Copy and validate experiment options before constructing a model or trainer."""
    config = {
        "coefficient": 0.1,
        "hidden": 64,
        "roi_size": 3,
        "warmup_epochs": 5,
        "ramp_epochs": 10,
        "feature_fusion": False,
        "encoding_channels": 32,
        "fusion_init": 0.01,
    }
    if include_progress:
        config["progress"] = 0.0
    if auxiliary_config is not None:
        if not isinstance(auxiliary_config, Mapping):
            raise ValueError("Auxiliary configuration must be a mapping")
        unknown = set(auxiliary_config) - set(config)
        if unknown:
            raise ValueError(f"Unknown auxiliary options: {sorted(unknown, key=str)}")
        config.update(auxiliary_config)
    for key in ("coefficient", "fusion_init"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Auxiliary option {key} must be finite and nonnegative")
        config[key] = float(value)
    if config["fusion_init"] > 1:
        raise ValueError("Auxiliary option fusion_init must lie within [0, 1]")
    for key in ("hidden", "roi_size", "encoding_channels", "warmup_epochs", "ramp_epochs"):
        value = config[key]
        minimum = 1 if key in {"hidden", "roi_size", "encoding_channels"} else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Invalid auxiliary option: {key}")
    if not isinstance(config["feature_fusion"], bool):
        raise ValueError("Auxiliary option feature_fusion must be a boolean")
    if include_progress:
        progress = config["progress"]
        if (
            isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not math.isfinite(progress)
            or not 0 <= progress <= 1
        ):
            raise ValueError("Auxiliary occlusion schedule progress must lie in [0, 1]")
        config["progress"] = float(progress)
    return config


class OcclusionSupervisedModel(DetectionModel):
    """Add GT-region supervision, optionally feeding its P3 encoding into Detect."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True, auxiliary_config=None):
        config = _validate_auxiliary_config(auxiliary_config, include_progress=True)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if type(self.model[-1]) is not Detect or self.end2end:
            raise ValueError("Occlusion supervision currently supports a standard, non-end-to-end Detect head")
        if any(type(module).__name__ == "C3k2GOA" for module in self.model.modules()):
            raise ValueError("Use a standard detector YAML, e.g. yolo11n.yaml, for this experiment without GOA")
        self.occlusion_aux_config = config
        # Build the standard detector first. Copy its head weights and restore the
        # RNG state after auxiliary initialization, keeping baseline initialization
        # and the subsequent CPU random stream identical for the same seed.
        with torch.random.fork_rng(devices=[]):
            head_options = {"hidden": config["hidden"], "roi_size": config["roi_size"]}
            head_type = OcclusionAuxDetect
            if config["feature_fusion"]:
                head_type = OcclusionFusionDetect
                head_options.update(encoding_channels=config["encoding_channels"], fusion_init=config["fusion_init"])
            self.model[-1] = head_type(self.model[-1], **head_options)
        self.stride = self.model[-1].stride

    def init_criterion(self):
        return OcclusionSupervisionLoss(self)


class OcclusionSupervisedTrainer(OcclusionTrainer):
    """Use metadata-safe data, original detection losses, and optional encoding fusion."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, auxiliary_config=None, safe_augment=True):
        self.auxiliary_config = _validate_auxiliary_config(auxiliary_config)
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
        weight_model = weights.get("model") if isinstance(weights, dict) else weights
        weight_layers = getattr(weight_model, "model", None)
        if weight_layers is not None and isinstance(weight_layers[-1], OcclusionFusionDetect):
            if not config["feature_fusion"]:
                raise ValueError(
                    "This checkpoint contains P3 occlusion fusion; enable --occlusion-fusion to retain its detector"
                )
            checkpoint_head = weight_layers[-1]
            checkpoint_options = {
                "encoding_channels": checkpoint_head.encoding_channels,
                "hidden": checkpoint_head.aux_classifier[0].out_features,
                "roi_size": checkpoint_head.roi_size,
            }
            differences = {key: value for key, value in checkpoint_options.items() if config[key] != value}
            if differences:
                raise ValueError(f"Match the fusion checkpoint's auxiliary settings before loading: {differences}")
        model = OcclusionSupervisedModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            auxiliary_config=config,
        )
        if weights:
            model.load(weights)
        mode = (
            "P3 occlusion encoding + residual fusion retained during inference"
            if config["feature_fusion"]
            else "training-only P3/P4/P5 GT-region classifier"
        )
        LOGGER.info(
            f"Occlusion supervision ({mode}): standard detection loss + "
            f"{config['coefficient']:g} * progress * GT-region CE; no GOA or detection-loss reweighting"
        )
        if config["feature_fusion"]:
            LOGGER.info(
                f"OcclusionFusionDetect: encoding_channels={config['encoding_channels']}, "
                f"learnable residual scale initialized to {config['fusion_init']:g}"
            )
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "occ_loss"
        return validator

    def _save_experiment_config(self, trainer):
        payload = deepcopy(self.auxiliary_config)
        payload.update(
            enabled=payload["coefficient"] > 0 or payload["feature_fusion"],
            auxiliary_supervision_enabled=payload["coefficient"] > 0,
            mode="p3_encoding_fusion" if payload["feature_fusion"] else "auxiliary_only",
            objective="standard detection loss + coefficient * progress * mean GT occlusion cross entropy",
            supervision="all known GT regions, independent of detection positive assignment",
            feature_levels="P3 encoder; residual fusion modifies only P3 Detect input"
            if payload["feature_fusion"]
            else "Detect inputs: P3/P4/P5 for standard YOLO11",
            auxiliary_pooling="GT ROI pooling on the P3 occlusion encoding"
            if payload["feature_fusion"]
            else "GT ROI pooling on the original P3/P4/P5 neck features",
            training_transforms="letterbox + HSV + flips"
            if self.occlusion_config["safe_augment"]
            else "letterbox only",
            validation_auxiliary_loss="not computed; val/occ_loss is zero",
            seed=self.args.seed,
            image_size=self.args.imgsz,
            model=self.args.model,
            pretrained=self.args.pretrained,
            inference_requires_occlusion_labels=False,
            inference_retains_feature_fusion=payload["feature_fusion"],
            inference_adds_modules=payload["feature_fusion"],
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
            mode="p3_encoding_fusion" if self.auxiliary_config["feature_fusion"] else "auxiliary_only",
            raw_mean_ce=statistics["ce_sum"] / count if count else None,
            mean_weighted_aux_loss=statistics["weighted_aux_sum"] / batches if batches else None,
            accuracy_by_occlusion=[
                correct / total if total else None
                for correct, total in zip(statistics["correct_by_occlusion"], statistics["evaluated_gt_by_occlusion"])
            ],
        )
        head = unwrap_model(self.model).model[-1]
        if isinstance(head, OcclusionFusionDetect):
            statistics["fusion_scale"] = float(head.fusion_scale.detach().float().item())
        with (self.save_dir / "occlusion_supervision_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(statistics, ensure_ascii=False) + "\n")
