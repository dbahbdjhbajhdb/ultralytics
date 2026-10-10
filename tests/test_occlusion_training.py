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


def test_joint_bonus_requires_occlusion_and_is_stronger_for_smaller_targets():
    loss = make_loss()
    loss.config.update(size_weight=0.0, occ_weight=0.0, joint_weight=0.5)
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 64.0, 64.0], [0.0, 0.0, 8.0, 8.0]]])
    batch = {"batch_idx": torch.tensor([0, 0, 0]), "occlusion": torch.tensor([2.0, 2.0, 0.0])}
    weights = loss.positive_weights(batch, gt, torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.bool))
    assert weights[0, 0] > weights[0, 1] > weights[0, 2] == 1


def test_unknown_occlusion_keeps_size_weight_and_tracks_unique_matched_targets():
    loss = make_loss()
    loss.config.update(size_weight=0.5, occ_weight=0.5, joint_weight=0.5, cap=5.0)
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0]] * 5])
    batch = {
        "batch_idx": torch.tensor([0, 0, 0, 0, 0]),
        "occlusion": torch.tensor([-1.0, 2.0, -1.0, 2.0, 0.0]),
        "occ_valid": torch.tensor([False, False, False, True, True]),
    }
    assigned = torch.tensor([[0, 0, 1, 3, 4]])
    fg = torch.tensor([[True, True, True, True, False]])
    weights = loss.positive_weights(batch, gt, assigned, fg)

    # Unknown visibility contributes no occlusion or joint bonus, even if its stored degree is 2.
    assert weights[0, :3].tolist() == pytest.approx([1.4, 1.4, 1.4])
    assert weights[0, 3].item() == pytest.approx(2.3)
    assert weights[0, 4].item() == 1.0
    assert loss.statistics["gt_with_unknown_occlusion"] == 3
    assert loss.statistics["matched_gt_with_unknown_occlusion"] == 2
    assert loss.statistics["gt_by_occlusion"] == [1, 0, 1]
    assert loss.statistics["matched_gt_by_occlusion"] == [0, 0, 1]
    assert loss.statistics["positive_anchors"] == 4


def test_legacy_batch_infers_unknown_occlusion_without_validity_mask():
    loss = make_loss()
    loss.config.update(size_weight=0.5, occ_weight=0.5, joint_weight=0.5, cap=5.0)
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0]] * 4])
    batch = {"batch_idx": torch.tensor([0, 0, 0, 0]), "occlusion": torch.tensor([-1.0, 0.0, 1.0, 2.0])}
    weights = loss.positive_weights(batch, gt, torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 4, dtype=torch.bool))

    assert weights.flatten().tolist() == pytest.approx([1.4, 1.4, 1.85, 2.3])
    assert loss.statistics["gt_with_unknown_occlusion"] == 1
    assert loss.statistics["matched_gt_with_unknown_occlusion"] == 1
    assert loss.statistics["gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["matched_gt_by_occlusion"] == [1, 1, 1]


def test_validity_mask_preserves_weights_and_statistics_for_known_occlusion():
    loss = make_loss()
    loss.config.update(joint_weight=0.5)
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 64.0, 64.0], [0.0, 0.0, 8.0, 8.0]]])
    batch = {"batch_idx": torch.tensor([0, 0, 0]), "occlusion": torch.tensor([0.0, 1.0, 2.0])}
    assigned = torch.tensor([[0, 1, 2]])
    fg = torch.ones(1, 3, dtype=torch.bool)
    legacy_weights = loss.positive_weights(batch, gt, assigned, fg)
    legacy_statistics = {key: value.copy() if isinstance(value, list) else value for key, value in loss.statistics.items()}
    loss.reset_statistics()
    explicit_weights = loss.positive_weights(dict(batch, occ_valid=torch.ones(3, dtype=torch.bool)), gt, assigned, fg)

    assert torch.equal(explicit_weights, legacy_weights)
    assert loss.statistics == legacy_statistics
    assert loss.statistics["gt_with_unknown_occlusion"] == 0
    assert loss.statistics["matched_gt_with_unknown_occlusion"] == 0


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
    assert batch["occ_valid"].dtype == torch.bool
    assert batch["occ_valid"].tolist() == [True, True]
    assert batch["cls"].flatten().tolist() == [0.0, 3.0]
    assert torch.allclose(batch["bboxes"][:, 0], torch.tensor([0.8, 0.3]), atol=1e-6)


@pytest.mark.parametrize("unknown_degree", [-1, None])
def test_unknown_occlusion_preserves_instance_order_flip_and_collation(tmp_path, unknown_degree):
    for kind in ["images", "labels", "occlusion"]:
        (tmp_path / kind / "train").mkdir(parents=True)
    for stem in ["a", "b"]:
        Image.new("RGB", (64, 64)).save(tmp_path / f"images/train/{stem}.jpg")
        (tmp_path / f"labels/train/{stem}.txt").write_text("0 .2 .4 .1 .2\n3 .7 .6 .2 .1\n")
        # JSON order differs from TXT order: alignment must use class and box, including unknown targets.
        objects = [
            {"yolo": [3, 0.7, 0.6, 0.2, 0.1], "occlusion": unknown_degree, "occ_valid": False},
            {"yolo": [0, 0.2, 0.4, 0.1, 0.2], "occlusion": 2},
        ]
        (tmp_path / f"occlusion/train/{stem}.json").write_text(json.dumps({"objects": objects}))
    hyp = get_cfg()
    hyp.fliplr = 1.0
    hyp.flipud = 0.0
    dataset = OcclusionDataset(
        img_path=str(tmp_path / "images/train"),
        imgsz=64,
        batch_size=2,
        augment=True,
        safe_augment=True,
        hyp=hyp,
        data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
    )
    samples = [dataset[0], dataset[1]]
    for sample in samples:
        assert sample["cls"].flatten().tolist() == [0.0, 3.0]
        assert sample["occlusion"].tolist() == [2.0, -1.0]
        assert sample["occ_valid"].dtype == torch.bool
        assert sample["occ_valid"].tolist() == [True, False]
        expected_boxes = torch.tensor([[0.8, 0.4, 0.1, 0.2], [0.3, 0.6, 0.2, 0.1]])
        assert torch.allclose(sample["bboxes"], expected_boxes, atol=1e-6)

    batch = OcclusionDataset.collate_fn(samples)
    assert batch["occlusion"].tolist() == [2.0, -1.0, 2.0, -1.0]
    assert batch["occ_valid"].dtype == torch.bool
    assert batch["occ_valid"].tolist() == [True, False, True, False]
    assert batch["cls"].flatten().tolist() == [0.0, 3.0, 0.0, 3.0]
    assert batch["batch_idx"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert torch.allclose(batch["bboxes"], expected_boxes.repeat(2, 1), atol=1e-6)


@pytest.mark.parametrize("invalid_degree", [-1, None, 3, "1"])
@pytest.mark.parametrize("validity_field", [{}, {"occ_valid": True}])
def test_invalid_occlusion_without_explicit_unknown_flag_fails(tmp_path, invalid_degree, validity_field):
    for kind in ["images", "labels", "occlusion"]:
        (tmp_path / kind / "train").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "images/train/a.jpg")
    (tmp_path / "labels/train/a.txt").write_text("0 .2 .4 .1 .2\n")
    obj = {"yolo": [0, 0.2, 0.4, 0.1, 0.2], "occlusion": invalid_degree, **validity_field}
    (tmp_path / "occlusion/train/a.json").write_text(json.dumps({"objects": [obj]}))

    with pytest.raises(ValueError, match="Invalid occlusion metadata"):
        OcclusionDataset(
            img_path=str(tmp_path / "images/train"),
            imgsz=64,
            batch_size=1,
            augment=False,
            hyp=get_cfg(),
            data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
        )
