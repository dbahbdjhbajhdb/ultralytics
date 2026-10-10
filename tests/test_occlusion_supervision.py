"""User-runnable regression checks for direct GT-region occlusion supervision."""

from io import BytesIO

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.occlusion_supervision import OcclusionSupervisedModel
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.occlusion_aux import OcclusionAuxDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.occlusion_supervision_loss import OcclusionSupervisionLoss
from ultralytics.utils.torch_utils import initialize_weights


class TinyDetectionModel(nn.Module):
    """Supply the standard loss contract without constructing an entire YOLO backbone."""

    def __init__(self, coefficient=0.1, channels=(2, 3, 4)):
        super().__init__()
        source = Detect(nc=2, ch=channels)
        source.stride = torch.tensor([8.0, 16.0, 32.0])
        self.model = nn.ModuleList([OcclusionAuxDetect(source, hidden=8, roi_size=3)])
        self.args = get_cfg()
        self.occlusion_aux_config = {"coefficient": coefficient, "progress": 1.0}


class NoForegroundAssigner(nn.Module):
    """Represent valid detection GT for which TAL finds no foreground anchor."""

    def __init__(self):
        super().__init__()
        self.seen_gt = 0

    def forward(self, scores, boxes, anchors, gt_labels, gt_boxes, mask_gt):
        self.seen_gt = int(mask_gt.sum())
        batch_size, anchor_count, _ = scores.shape
        return (
            torch.zeros((batch_size, anchor_count), dtype=torch.long, device=scores.device),
            torch.zeros_like(boxes),
            torch.zeros_like(scores),
            torch.zeros((batch_size, anchor_count), dtype=torch.bool, device=scores.device),
            torch.zeros((batch_size, anchor_count), dtype=torch.long, device=scores.device),
        )


def feature_pyramid(batch_size=2, channels=(2, 3, 4), requires_grad=False):
    return tuple(
        torch.randn(batch_size, channel_count, height, width, requires_grad=requires_grad)
        for channel_count, (height, width) in zip(channels, [(8, 8), (4, 4), (2, 2)])
    )


def detection_batch():
    return {
        "batch_idx": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        "bboxes": torch.tensor([[0.6, 0.6, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]] * 2),
        "occlusion": torch.tensor([0.0, 1.0, 2.0, -1.0]),
        "occ_valid": torch.tensor([True, True, True, False]),
    }


def predictions(model, features):
    return {
        "detect": [torch.zeros(f.shape[0], model.model[-1].no, *f.shape[2:], requires_grad=True) for f in features],
        "occlusion_features": features,
    }


def test_region_sampling_uses_normalized_position_and_correct_image_at_every_scale():
    channels = (2, 3, 4)
    head = OcclusionAuxDetect(Detect(nc=2, ch=channels), hidden=8, roi_size=3)
    features = []
    offsets = []
    for level, (channel_count, (height, width)) in enumerate(zip(channels, [(12, 10), (6, 5), (3, 3)])):
        y = (torch.arange(height).float() + 0.5) / height
        x = (torch.arange(width).float() + 0.5) / width
        spatial = x[None, :] + 2 * y[:, None]
        values = torch.stack(
            [
                torch.stack([spatial + 10 * image + 100 * level + channel for channel in range(channel_count)])
                for image in range(2)
            ]
        )
        features.append(values)
        offsets.extend(100 * level + channel for channel in range(channel_count))
    boxes = torch.tensor([[0.6, 0.55, 0.1, 0.15], [0.4, 0.45, 0.1, 0.15], [0.4, 0.45, 0.1, 0.15]])
    pooled = head.pool_regions(tuple(features), boxes, torch.tensor([1, 0, 1]))
    expected = torch.tensor([11.7, 1.3, 11.3])[:, None] + torch.tensor(offsets)[None, :]

    assert pooled.shape == (3, sum(channels))
    assert pooled.dtype == torch.float32
    assert torch.allclose(pooled, expected, atol=1e-4)


def test_region_sampling_uses_box_extent_instead_of_only_the_center():
    head = OcclusionAuxDetect(Detect(nc=2, ch=(1, 1, 1)), hidden=8, roi_size=3)
    x = (torch.arange(64).float() + 0.5) / 64
    field = (x - 0.5).square()[None, None, None, :].expand(1, 1, 64, 64)
    features = (field, F.avg_pool2d(field, 2), F.avg_pool2d(field, 4))
    boxes = torch.tensor([[0.5, 0.5, 0.05, 0.2], [0.5, 0.5, 0.8, 0.2]])
    pooled = head.pool_regions(features, boxes, torch.tensor([0, 0]))

    assert (pooled[1] > pooled[0] + 0.01).all()


def test_all_known_gt_receive_auxiliary_supervision_even_without_tal_foreground(monkeypatch):
    model = TinyDetectionModel()
    loss = OcclusionSupervisionLoss(model)
    loss.assigner = NoForegroundAssigner()
    head = model.model[-1]
    captured = {}
    classify = head.classify_regions

    def capture(features, boxes, batch_idx):
        captured["boxes"] = boxes.detach().clone()
        captured["batch_idx"] = batch_idx.detach().clone()
        return classify(features, boxes, batch_idx)

    monkeypatch.setattr(head, "classify_regions", capture)
    batch = detection_batch()
    value, items = loss(predictions(model, feature_pyramid()), batch)

    assert loss.assigner.seen_gt == 4  # The unknown object still participates in ordinary detection.
    assert torch.equal(captured["boxes"], batch["bboxes"][:3])
    assert torch.equal(captured["batch_idx"], batch["batch_idx"][:3].long())
    assert value.shape == items.shape == (4,)
    assert torch.isfinite(value).all() and value[3] > 0
    assert loss.statistics["supervised_gt"] == 3
    assert loss.statistics["gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["evaluated_gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["unknown_gt"] == 1


@pytest.mark.parametrize("unknown_degree", [-1.0, 2.0])
def test_unknown_flag_excludes_auxiliary_target_without_changing_other_targets(unknown_degree):
    model = TinyDetectionModel()
    loss = OcclusionSupervisionLoss(model)
    features = feature_pyramid()
    batch = detection_batch()
    batch["occlusion"][-1] = unknown_degree
    expected_batch = {key: value[:3] for key, value in batch.items()}
    expected = loss._auxiliary_loss(features, expected_batch)
    loss.reset_statistics()
    actual = loss._auxiliary_loss(features, batch)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert batch["bboxes"].shape[0] == 4 and batch["cls"].shape[0] == 4
    assert loss.statistics["supervised_gt"] == 3
    assert loss.statistics["gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["unknown_gt"] == 1


@pytest.mark.parametrize("degree", [-1.0, 3.0, 0.5, float("nan")])
def test_invalid_degree_marked_known_is_rejected(degree):
    loss = OcclusionSupervisionLoss(TinyDetectionModel())
    batch = detection_batch()
    batch["occlusion"][0] = degree
    with pytest.raises(ValueError):
        loss._auxiliary_loss(feature_pyramid(), batch)


@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_all_unknown_auxiliary_loss_is_finite_connected_zero(empty):
    model = TinyDetectionModel()
    loss = OcclusionSupervisionLoss(model)
    features = feature_pyramid(requires_grad=True)
    batch = detection_batch()
    if empty:
        batch = {key: value[:0] for key, value in batch.items()}
    else:
        batch["occ_valid"].fill_(False)
        batch["occlusion"].fill_(-1)
    value = loss._auxiliary_loss(features, batch)
    assert torch.isfinite(value) and value.item() == 0.0
    value.backward()

    assert loss.statistics["supervised_gt"] == 0
    for feature in features:
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad) == 0
    for parameter in model.model[-1].aux_classifier.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


def test_unknown_half_precision_targets_do_not_overflow_zero_loss():
    loss = OcclusionSupervisionLoss(TinyDetectionModel())
    features = tuple(
        torch.full_like(feature, 60000.0, dtype=torch.float16, requires_grad=True) for feature in feature_pyramid()
    )
    batch = detection_batch()
    batch["occ_valid"].fill_(False)
    batch["occlusion"].fill_(-1)
    value = loss._auxiliary_loss(features, batch)

    assert value.dtype == torch.float32
    assert torch.isfinite(value) and value.item() == 0.0
    value.backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)


def test_auxiliary_gradient_reaches_the_shared_feature_extractor():
    model = TinyDetectionModel(channels=(4, 4, 4))
    loss = OcclusionSupervisionLoss(model)
    shared = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    feature = shared(torch.randn(2, 3, 8, 8))
    features = (feature, F.avg_pool2d(feature, 2), F.avg_pool2d(feature, 4))
    for item in features:
        item.retain_grad()
    value = loss._auxiliary_loss(features, detection_batch())
    value.backward()

    assert shared.weight.grad is not None and torch.isfinite(shared.weight.grad).all()
    assert shared.weight.grad.abs().sum() > 0
    assert all(item.grad is not None and item.grad.abs().sum() > 0 for item in features)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.model[-1].aux_classifier.parameters()
    )


def test_zero_coefficient_preserves_detection_losses_and_skips_roi(monkeypatch):
    model = TinyDetectionModel(coefficient=0.0)
    loss = OcclusionSupervisionLoss(model)
    baseline = v8DetectionLoss(model)
    batch = detection_batch()
    preds = predictions(model, feature_pyramid())

    def unexpected_auxiliary_call(*args, **kwargs):
        raise AssertionError("Zero auxiliary coefficient must skip GT-region classification")

    monkeypatch.setattr(model.model[-1], "classify_regions", unexpected_auxiliary_call)
    expected, expected_items = baseline(preds["detect"], batch)
    actual, actual_items = loss(preds, batch)

    assert torch.equal(actual[:3], expected)
    assert torch.equal(actual_items[:3], expected_items)
    assert actual[3].item() == actual_items[3].item() == 0.0
    assert loss.statistics["supervised_gt"] == 0
    assert loss.statistics["gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["evaluated_gt_by_occlusion"] == [0, 0, 0]
    assert loss.statistics["ce_sum"] == 0.0


def test_warmup_records_predictions_without_adding_an_optimization_loss():
    model = TinyDetectionModel()
    loss = OcclusionSupervisionLoss(model)
    loss.set_progress(0.0)
    value, items = loss(predictions(model, feature_pyramid()), detection_batch())

    assert value[3].item() == items[3].item() == 0.0
    assert loss.statistics["supervised_gt"] == 3
    assert loss.statistics["evaluated_gt_by_occlusion"] == [1, 1, 1]
    assert loss.statistics["ce_sum"] > 0
    assert loss.statistics["weighted_aux_sum"] == 0.0


def test_training_forward_keeps_original_features_and_standard_detection_predictions():
    source = Detect(nc=2, ch=(2, 3, 4)).train()
    head = OcclusionAuxDetect(source, hidden=8, roi_size=3).train()
    features = feature_pyramid()
    expected = source(list(features))
    result = head(list(features))

    assert set(result) == {"detect", "occlusion_features"}
    assert all(actual is original for actual, original in zip(result["occlusion_features"], features))
    assert all(torch.equal(actual, original) for actual, original in zip(result["detect"], expected))


def test_evaluation_preserves_standard_detect_output_and_never_samples_roi(monkeypatch):
    source = Detect(nc=2, ch=(2, 3, 4)).eval()
    initialize_weights(source)  # Match the BN and activation settings applied by DetectionModel.
    source.stride = torch.tensor([8.0, 16.0, 32.0])
    head = OcclusionAuxDetect(source, hidden=8, roi_size=3).eval()
    features = feature_pyramid()

    def unexpected_auxiliary_call(*args, **kwargs):
        raise AssertionError("Inference must not require GT boxes or auxiliary predictions")

    monkeypatch.setattr(head, "pool_regions", unexpected_auxiliary_call)
    monkeypatch.setattr(head, "classify_regions", unexpected_auxiliary_call)
    with torch.no_grad():
        expected, expected_raw = source(list(features))
        actual, actual_raw = head(list(features))

    assert head.no == source.no
    assert actual.shape == expected.shape and actual.shape[1] == 4 + source.nc
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_raw, expected_raw))


def test_supervised_model_keeps_standard_initialization_and_checkpoint_class_paths():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = DetectionModel("yolo11n.yaml", nc=2, verbose=False)
        baseline_next_random = torch.rand(4)
        torch.manual_seed(42)
        supervised = OcclusionSupervisedModel("yolo11n.yaml", nc=2, verbose=False)
        supervised_next_random = torch.rand(4)
    baseline_state = baseline.state_dict()
    supervised_state = supervised.state_dict()
    assert all(torch.equal(value, supervised_state[key]) for key, value in baseline_state.items())
    assert torch.equal(baseline_next_random, supervised_next_random)
    assert isinstance(supervised.model[-1], OcclusionAuxDetect)
    supervised_modules = dict(supervised.named_modules())
    for name, module in baseline.named_modules():
        copied_module = supervised_modules[name]
        if isinstance(module, nn.BatchNorm2d):
            # These behavior-changing settings are absent from state_dict().
            assert copied_module.eps == module.eps
            assert copied_module.momentum == module.momentum
            assert copied_module.affine == module.affine
            assert copied_module.track_running_stats == module.track_running_stats
        if hasattr(module, "inplace"):
            assert copied_module.inplace == module.inplace
    supervised_parameters = dict(supervised.named_parameters())
    assert all(
        parameter.requires_grad == supervised_parameters[name].requires_grad
        for name, parameter in baseline.named_parameters()
    )

    checkpoint = BytesIO()
    torch.save({"model": supervised}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    assert isinstance(restored, OcclusionSupervisedModel)
    assert isinstance(restored.model[-1], OcclusionAuxDetect)
    assert restored.__class__.__module__ == "ultralytics.models.yolo.detect.occlusion_supervision"
    assert restored.model[-1].__class__.__module__ == "ultralytics.nn.modules.occlusion_aux"
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in supervised_state.items())
