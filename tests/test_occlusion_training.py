"""Check progressive weights and augmentations preserve instance-level occlusion supervision."""

import json

import pytest
import torch
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.data.occlusion import OcclusionDataset
from ultralytics.models.yolo.detect.occlusion_train import weight_progress
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.occlusion_loss import WeightedDetectionLoss


def make_loss():
    model = DetectionModel("yolo11n.yaml", nc=10, verbose=False)
    model.args = get_cfg()
    model.occlusion_config = {"size_weight": 0.5, "occ_weight": 0.5, "cap": 2.0}
    return WeightedDetectionLoss(model)


def test_schedule_warmup_then_ramp():
    assert [weight_progress(i, 1, 2) for i in range(4)] == [0.0, 0.5, 1.0, 1.0]
    assert weight_progress(0, 0, 0) == 1.0
    with pytest.raises(ValueError):
        weight_progress(0, -1, 2)


def test_identically_hard_targets_keep_extra_weight_and_track_unmatched():
    loss = make_loss()
    batch = {"batch_idx": torch.tensor([0, 0, 0]), "occlusion": torch.tensor([2.0, 2.0, 0.0])}
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 64.0, 64.0]]])
    assigned = torch.tensor([[0, 1, 0]])
    fg = torch.tensor([[True, True, False]])
    loss.set_progress(0.0)
    assert torch.equal(loss.positive_weights(batch, gt, assigned, fg), torch.ones(1, 3))
    loss.reset_statistics()
    loss.set_progress(1.0)
    weights = loss.positive_weights(batch, gt, assigned, fg)
    assert (weights[fg] > 1).all() and (weights <= 2).all() and weights[0, 2] == 1
    assert loss.statistics["gt_by_occlusion"] == [1, 0, 2]
    assert loss.statistics["matched_gt_by_occlusion"] == [0, 0, 2]


def test_size_and_occlusion_each_influence_weight():
    loss = make_loss()
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 64.0, 64.0], [0.0, 0.0, 8.0, 8.0]]])
    batch = {"batch_idx": torch.tensor([0, 0, 0]), "occlusion": torch.tensor([0.0, 0.0, 2.0])}
    weights = loss.positive_weights(batch, gt, torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.bool))
    assert weights[0, 2] > weights[0, 0] > weights[0, 1] >= 1


def test_flip_preserves_occlusion_and_instance_order(tmp_path):
    for kind in ["images", "labels", "metadata"]:
        (tmp_path / kind / "train").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "images/train/a.jpg")
    (tmp_path / "labels/train/a.txt").write_text("0 .2 .4 .1 .2\n3 .7 .6 .2 .1\n")
    objects = [{"yolo": [0, 0.2, 0.4, 0.1, 0.2], "occlusion": 2}, {"yolo": [3, 0.7, 0.6, 0.2, 0.1], "occlusion": 0}]
    (tmp_path / "metadata/train/a.json").write_text(json.dumps({"objects": objects}))
    hyp = get_cfg()
    hyp.fliplr = 1.0
    hyp.flipud = 0.0
    dataset = OcclusionDataset(
        img_path=str(tmp_path / "images/train"),
        imgsz=64,
        batch_size=1,
        augment=True,
        safe_augment=True,
        hyp=hyp,
        data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
    )
    batch = dataset[0]
    assert batch["occlusion"].tolist() == [2.0, 0.0]
    assert batch["cls"].flatten().tolist() == [0.0, 3.0]
    assert torch.allclose(batch["bboxes"][:, 0], torch.tensor([0.8, 0.3]), atol=1e-6)
