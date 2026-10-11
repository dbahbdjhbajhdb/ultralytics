# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Occlusion-grade-guided paired training with an EMA reference and local P3 compensation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy

import torch

from ultralytics.data.occlusion_pairing import make_occlusion_pairs
from ultralytics.models.yolo.detect.occlusion_train import OcclusionTrainer, weight_progress
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.occlusion_paired import PairedOcclusionDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.occlusion_paired_loss import PairedOcclusionLoss, sample_pair_regions
from ultralytics.utils.torch_utils import unwrap_model


def validate_pair_config(pair_config=None, include_progress=False):
    """Reject invalid or misspelled options before model construction."""
    config = {
        "coefficient": 0.05,
        "warmup_epochs": 10,
        "ramp_epochs": 10,
        "pair_probability": 0.5,
        "coverage_min": 0.15,
        "coverage_max": 0.35,
        "neighbor_max_coverage": 0.35,
        "teacher_conf": 0.15,
        "teacher_iou": 0.5,
        "hidden": 32,
        "roi_size": 3,
        "selection": "grade",
        "compensation": True,
        "pairing": True,
    }
    if include_progress:
        config["progress"] = 0.0
    if pair_config is not None:
        if not isinstance(pair_config, Mapping):
            raise ValueError("Paired occlusion options must be a mapping")
        unknown = set(pair_config) - set(config)
        if unknown:
            raise ValueError(f"Unknown paired occlusion options: {sorted(unknown, key=str)}")
        config.update(pair_config)
    for key in (
        "coefficient", "pair_probability", "coverage_min", "coverage_max", "neighbor_max_coverage",
        "teacher_conf", "teacher_iou",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Paired occlusion option {key} must be a finite number")
        if value < 0 or (key != "coefficient" and value > 1):
            raise ValueError(f"Paired occlusion option {key} is out of range")
        config[key] = float(value)
    if not 0 < config["coverage_min"] <= config["coverage_max"] < 0.5:
        raise ValueError("Use 0 < coverage_min <= coverage_max < 0.5")
    if not 0 < config["neighbor_max_coverage"] < 0.5:
        raise ValueError("neighbor_max_coverage must lie strictly between 0 and 0.5")
    if not 0 < config["teacher_conf"] <= 1 or not 0 < config["teacher_iou"] <= 1:
        raise ValueError("Teacher confidence and IoU thresholds must lie within (0, 1]")
    for key in ("hidden", "roi_size", "warmup_epochs", "ramp_epochs"):
        value = config[key]
        minimum = 1 if key in {"hidden", "roi_size"} else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid paired occlusion option: {key}")
    for key in ("compensation", "pairing"):
        if not isinstance(config[key], bool):
            raise ValueError(f"Paired occlusion option {key} must be boolean")
    if config["selection"] not in {"grade", "random"}:
        raise ValueError("Pair selection must be grade or random")
    if include_progress:
        value = config["progress"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Paired occlusion progress must lie within [0, 1]")
        config["progress"] = float(value)
    return config


class PairedOcclusionModel(DetectionModel):
    """Preserve YOLO detection outputs and optionally tap the standard YOLO11 P2 map."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True, pair_config=None):
        config = validate_pair_config(pair_config, include_progress=True)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        source = self.model[-1]
        if type(source) is not Detect or self.end2end:
            raise ValueError("Paired occlusion training requires a standard non-end-to-end Detect head")
        if tuple(source.stride.tolist()) != (8.0, 16.0, 32.0):
            raise ValueError("Paired training requires P3/P4/P5 detection strides 8/16/32")
        if any(type(module).__name__ == "C3k2GOA" for module in self.model.modules()):
            raise ValueError("Use the standard yolo11n.yaml, without GOA, for paired training")
        # The P2 tap is deliberately restricted to the unmodified YOLO11 graph.
        # Never silently read layer 2 from an unrelated architecture.
        if len(self.model) != 24 or list(source.f) != [16, 19, 22] or type(self.model[2]).__name__ != "C3k2":
            raise ValueError("This implementation supports the standard YOLO11 n/s/m/l/x architecture only")
        p2_channels = self.model[2].cv2.conv.out_channels
        self.occlusion_pair_config = config
        with torch.random.fork_rng(devices=[]):
            head = PairedOcclusionDetect(
                source, p2_channels=p2_channels, hidden=config["hidden"],
                roi_size=config["roi_size"], compensation=config["compensation"],
            )
        if config["compensation"]:
            head.f = [2, *source.f]
            self.save = sorted(set(self.save) | {2})
        self.model[-1] = head
        self.stride = head.stride

    def init_criterion(self):
        return PairedOcclusionLoss(self)


class PairedOcclusionTrainer(OcclusionTrainer):
    """Use one student forward, with a clean EMA forward only when pairing is active."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, pair_config=None, safe_augment=True):
        self.pair_config = validate_pair_config(pair_config)
        batch = (overrides or {}).get("batch", 4)
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("Paired training requires a fixed positive batch size")
        if (overrides or {}).get("compile", False):
            raise ValueError("Disable compile for the dynamic paired-region experiment")
        super().__init__(
            cfg=cfg, overrides=overrides, _callbacks=_callbacks,
            occlusion_config={"enabled": False, "safe_augment": bool(safe_augment)},
        )
        self.pair_progress = 0.0
        self.pair_epoch_statistics = {}

    def get_model(self, cfg=None, weights=None, verbose=True):
        config = deepcopy(self.pair_config)
        config["progress"] = weight_progress(0, config["warmup_epochs"], config["ramp_epochs"])
        weight_model = weights.get("model") if isinstance(weights, dict) else weights
        layers = getattr(weight_model, "model", None)
        if layers is not None and isinstance(layers[-1], PairedOcclusionDetect):
            previous = layers[-1]
            if previous.compensation_enabled != config["compensation"]:
                raise ValueError("Match the paired checkpoint's compensation setting when loading it")
            if previous.roi_size != config["roi_size"] or previous.hidden != config["hidden"]:
                raise ValueError("Match the paired checkpoint's hidden width and ROI size when loading it")
        elif layers is not None and type(layers[-1]) is not Detect:
            raise ValueError("Old GOA/auxiliary/fusion checkpoints cannot initialize this independent paired experiment")
        model = PairedOcclusionModel(
            cfg=cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose, pair_config=config,
        )
        if weights:
            model.load(weights)
        LOGGER.info(
            f"Paired occlusion: compensation={config['compensation']}, pairing={config['pairing']}, "
            f"selection={config['selection']}; standard detection + {config['coefficient']:g} * progress * feature cosine"
        )
        LOGGER.info(
            "Reference: same-run EMA, original-view P3, class/confidence/IoU-filtered GTs; "
            "no degree CE, GOA or detection-loss reweighting"
        )
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "pair_loss"
        return validator

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        # Warm-up and the no-pairing control use exactly the original student images.
        if not self.pair_config["pairing"] or self.pair_progress <= 0 or self.pair_config["pair_probability"] == 0:
            batch["pair_indices"] = torch.empty(0, dtype=torch.long, device=batch["img"].device)
            return batch
        if self.ema is None:
            raise RuntimeError("EMA is required for paired occlusion training")
        teacher = self.ema.ema
        head = teacher.model[-1]
        previous_capture = head.capture_pair_features
        try:
            head.capture_pair_features = True
            # EMA has no gradients and stays in eval mode (including its BN layers).
            # FP32 ensures teacher thresholds do not depend on the student's AMP mode.
            with torch.no_grad(), torch.autocast(device_type=batch["img"].device.type, enabled=False):
                reference = teacher(batch["img"].float())
        finally:
            head.capture_pair_features = previous_capture
        decoded = reference["detect"]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        paired = make_occlusion_pairs(batch, self.pair_config, self.pair_progress, teacher_predictions=decoded)
        statistics = paired.pop("statistics")
        for key, value in statistics.items():
            if key == "effective_pair_probability":
                self.pair_epoch_statistics[key] = value
            elif isinstance(value, (int, float)):
                key = "aug_" + key
                self.pair_epoch_statistics[key] = self.pair_epoch_statistics.get(key, 0) + value
            elif isinstance(value, list):
                key = "aug_" + key
                previous = self.pair_epoch_statistics.get(key, [0] * len(value))
                self.pair_epoch_statistics[key] = [a + b for a, b in zip(previous, value)]
        indices = paired["pair_indices"]
        if len(indices) and self.pair_config["coefficient"] > 0:
            with torch.no_grad():
                paired["pair_targets"] = sample_pair_regions(
                    reference["reference_features"][0], paired["bboxes"][indices],
                    paired["batch_idx"][indices], self.pair_config["roi_size"],
                ).detach()
        # No full teacher feature map is retained in the batch or student graph.
        return paired

    def _save_experiment_config(self, trainer):
        payload = deepcopy(self.pair_config)
        payload.update(
            mode="grade_guided_paired_local_compensation",
            objective="standard detection + coefficient * progress * paired-region feature cosine distance",
            reference="same-run EMA original-view neck P3; detached, class/confidence/IoU filtered",
            grade_supervision="grade 0 selects synthetic pairs; real grades are never rewritten",
            random_control="known grades 0/1 eligible; grade 2 protected in both selections",
            compensation_description="local P3 + backbone P2 detail + neck P4 context; zero-initialized residual projection"
            if payload["compensation"] else "disabled; original detection architecture",
            detector_stride=[8, 16, 32],
            training_transforms="letterbox + HSV + flips"
            if self.occlusion_config["safe_augment"] else "letterbox only",
            validation_pair_loss="zero; synthetic pairs and EMA reference are training-only",
            inference_requires_occlusion_labels=False,
            inference_retains_compensation=payload["compensation"],
            unknown_occlusion="retain GT detection supervision; exclude from pair selection",
            real_grade_2="retain all GT detection supervision; never add occlusion to selected grade-2 GT",
            hypothesis="unverified; original-view reference is not complete-object ground truth",
            model=self.args.model, pretrained=self.args.pretrained, seed=self.args.seed,
            image_size=self.args.imgsz, save_dir=str(self.save_dir),
        )
        (self.save_dir / "occlusion_paired_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        LOGGER.info(f"Paired configuration: {self.save_dir / 'occlusion_paired_config.json'}")
        LOGGER.info(f"Paired diagnostics: {self.save_dir / 'occlusion_paired_diagnostics.jsonl'}")

    def _start_weight_epoch(self, trainer):
        self.pair_progress = weight_progress(
            self.epoch, self.pair_config["warmup_epochs"], self.pair_config["ramp_epochs"],
        )
        self.pair_epoch_statistics = {}
        for model in (unwrap_model(self.model), self.ema.ema if self.ema else None):
            if model is None:
                continue
            model.occlusion_pair_config["progress"] = self.pair_progress
            criterion = getattr(model, "criterion", None)
            if isinstance(criterion, PairedOcclusionLoss):
                criterion.set_progress(self.pair_progress)
                criterion.reset_statistics()

    def _save_weight_statistics(self, trainer):
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        if not isinstance(criterion, PairedOcclusionLoss):
            return
        statistics = deepcopy(criterion.statistics)
        statistics.update(self.pair_epoch_statistics)
        statistics.update(
            epoch=self.epoch + 1, progress=self.pair_progress,
            effective_coefficient=self.pair_config["coefficient"] * self.pair_progress,
            compensation=self.pair_config["compensation"], pairing=self.pair_config["pairing"],
            selection=self.pair_config["selection"],
        )
        statistics["mean_weighted_pair_loss"] = (
            statistics.get("weighted_pair_sum", 0) / statistics["batch_count"] if statistics["batch_count"] else None
        )
        statistics["mean_raw_pair_loss"] = (
            statistics["cosine_sum"] / statistics["valid_locations"] if statistics["valid_locations"] else None
        )
        statistics["mean_synthetic_coverage"] = (
            statistics.get("aug_coverage_sum", 0) / statistics["aug_paired_gt"]
            if statistics.get("aug_paired_gt", 0) else None
        )
        with (self.save_dir / "occlusion_paired_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(statistics, ensure_ascii=False) + "\n")
