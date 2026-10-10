# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Opt-in trainer; default YOLO training remains unchanged."""

import os
from ultralytics.data.occlusion import OcclusionDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.occlusion_loss import WeightedDetectionLoss


class OcclusionModel(DetectionModel):
    def init_criterion(self):
        return WeightedDetectionLoss(self)


class OcclusionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return OcclusionDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=mode == "val",
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0 if mode == "train" else 0.5,
            task="detect",
            data=self.data,
            fraction=1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        cls = OcclusionModel if os.environ.get("WEIGHTED_LOSS", "0") == "1" else DetectionModel
        model = cls(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
