# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Metadata-preserving dataset for controlled occlusion experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths


class OcclusionDataset(YOLODataset):
    def __init__(self, *args, safe_augment=False, **kwargs):
        self.safe_augment = bool(safe_augment)
        self.training_split = bool(kwargs.get("augment", False))
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()
        for lb in labels:
            path = Path(img2label_paths([lb["im_file"]])[0])
            label_root = path.parent if path.parent.name == "labels" else path.parent.parent
            data_root = label_root.parent
            candidates = [
                folder / (path.stem + ".json")
                for root in [data_root / "metadata", data_root / "occlusion"]
                for folder in [root / path.parent.name, root]
            ]
            candidates.append(path.with_suffix(".json"))
            meta = next((p for p in candidates if p.exists()), None)
            if meta is None:
                raise ValueError(f"Missing occlusion metadata for {path}")
            payload = json.loads(meta.read_text())
            if payload.get("label_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != payload["label_sha256"]:
                raise ValueError(f"Label hash mismatch: {path}")
            mapping = {}
            for obj in payload["objects"]:
                values = obj.get("yolo")
                if values is None:
                    values = [obj["class_id"], *obj["bbox_xywh_normalized"]]
                if not obj.get("occ_valid", True) or obj["occlusion"] not in (0, 1, 2):
                    raise ValueError(f"Invalid occlusion metadata: {meta}")
                key = tuple(np.asarray(values, dtype=np.float32).round(6))
                if key in mapping and mapping[key] != obj["occlusion"]:
                    raise ValueError("Conflicting metadata")
                mapping[key] = obj["occlusion"]
            occ = []
            for cls, box in zip(lb["cls"], lb["bboxes"]):
                key = tuple(np.r_[cls, box].astype(np.float32).round(6))
                if key not in mapping:
                    raise ValueError(f"Label metadata mismatch: {path}")
                occ.append(mapping[key])
            lb["occlusion"] = np.array(occ, dtype=np.float32)
        return labels

    def build_transforms(self, hyp=None):
        # HSV and flips preserve visibility, row order and object count; crops and mixes do not.
        transforms = [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=self.training_split)]
        if self.training_split and self.safe_augment:
            transforms.extend(
                [
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip(p=hyp.fliplr, direction="horizontal"),
                    RandomFlip(p=hyp.flipud, direction="vertical"),
                ]
            )
        transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0))
        return Compose(transforms)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch["occlusion"] = torch.tensor(self.labels[index]["occlusion"])
        if len(batch["occlusion"]) != len(batch["cls"]):
            raise ValueError("Transformed label alignment failed")
        return batch

    @staticmethod
    def collate_fn(batch):
        occlusion = torch.cat([item.pop("occlusion") for item in batch])
        result = YOLODataset.collate_fn(batch)
        result["occlusion"] = occlusion
        return result
