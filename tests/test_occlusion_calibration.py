"""Checks for held-out monotonic confidence calibration in the occlusion pilot."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def calibration_module():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location(
        "occlusion_calibration_eval", Path(__file__).parents[1] / "examples/VisDrone-Occlusion/evaluate_occlusion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detection_sample(probabilities, positives, image="sample", image_sha256=None):
    """Create spatially distinct detections with known TP/FP labels."""
    scores = torch.tensor(probabilities, dtype=torch.float64)
    x = torch.arange(len(scores), dtype=torch.float32) * 20
    boxes = torch.stack((x, torch.zeros_like(x), x + 10, torch.ones_like(x) * 10), dim=1)
    positive_ids = torch.as_tensor(positives, dtype=torch.long)
    sample = {
        "image": image,
        "scores": scores,
        "boxes": boxes,
        "classes": torch.zeros(len(scores), dtype=torch.long),
        "keep": torch.arange(len(scores)),
        "eligible": torch.ones(len(scores), dtype=torch.bool),
        "gt": boxes[positive_ids],
        "gcls": torch.zeros(len(positive_ids), dtype=torch.long),
        "occ": torch.zeros(len(positive_ids), dtype=torch.long),
        "area": torch.full((len(positive_ids),), 100.0),
        "trace": [],
        "candidate_ids": torch.arange(len(scores)),
    }
    if image_sha256 is not None:
        sample["image_sha256"] = image_sha256
    return sample


def test_collect_rejects_empty_split(calibration_module, tmp_path):
    with pytest.raises(ValueError, match="No supported images"):
        calibration_module.collect(None, tmp_path, "cal", 320, 0.0001, 0.5, 0.5)


@pytest.mark.parametrize("suffix", [".png", ".JPEG"])
def test_collect_reads_supported_extensions(calibration_module, tmp_path, suffix):
    import numpy as np

    images = tmp_path / "images/cal"
    metadata = tmp_path / "metadata/cal"
    images.mkdir(parents=True)
    metadata.mkdir(parents=True)
    image = images / ("sample" + suffix)
    assert calibration_module.cv2.imwrite(str(image), np.zeros((64, 64, 3), dtype=np.uint8))
    (metadata / "sample.json").write_text(json.dumps({"objects": []}))
    prediction = torch.tensor([160.0, 160.0, 16.0, 16.0, 0.5]).reshape(1, 5, 1)
    samples = calibration_module.collect(lambda _: prediction, tmp_path, "cal", 320, 0.0001, 0.5, 0.5)
    assert len(samples) == 1 and samples[0]["image"] == "sample"
    assert len(samples[0]["image_sha256"]) == 64


@pytest.mark.parametrize("method", ["temperature", "platt"])
@pytest.mark.parametrize("kind", ["no_images", "no_detections", "only_negative", "only_positive"])
def test_missing_class_does_not_fit(calibration_module, method, kind):
    samples = {
        "no_images": [],
        "no_detections": [detection_sample([], [])],
        "only_negative": [detection_sample([0.1, 0.9], [])],
        "only_positive": [detection_sample([0.1, 0.9], [0, 1])],
    }[kind]
    result = calibration_module.fit_calibration(samples, 0.5, method)
    assert result["status"] == "not_fitted_degenerate_labels"
    p = torch.tensor([0.0, 0.2, 0.8, 1.0])
    assert torch.equal(calibration_module.apply_calibration(p, result), p)


@pytest.mark.parametrize("score,positive_count", [(0.8, 5), (0.2, 15)])
def test_platt_bias_corrects_confidence(calibration_module, score, positive_count):
    sample = detection_sample([score] * 20, range(positive_count))
    result = calibration_module.fit_platt([sample], 0.5)
    q = calibration_module.apply_calibration(sample["scores"], result)
    assert result["status"] == "fitted" and result["slope"] > 0
    assert torch.isfinite(q).all() and torch.all((q > 0) & (q < 1))
    assert abs(float(q.mean()) - positive_count / 20) < 0.025
    assert result["cal_brier_after"] < result["cal_brier_before"]
    assert result["bias"] < 0 if score == 0.8 else result["bias"] > 0


def test_mapping_is_monotonic_and_finite_at_probability_boundaries(calibration_module):
    sample = detection_sample([0.0, 0.1, 0.3, 0.55, 0.8, 1.0], [2, 4, 5])
    result = calibration_module.fit_platt([sample], 0.5)
    grid = torch.linspace(0, 1, 1001, dtype=torch.float64)
    q = calibration_module.apply_calibration(grid, result)
    assert result["slope"] > 0 and torch.isfinite(q).all()
    assert torch.all(q[1:] >= q[:-1]) and torch.all((q >= 0) & (q <= 1))
    central = calibration_module.apply_calibration(torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64), result)
    assert torch.all(central[1:] > central[:-1])


def test_ece_assigns_all_boundaries_once(calibration_module):
    p = torch.arange(11, dtype=torch.float64) / 10
    assert calibration_module.ece(p, torch.zeros_like(p)) == pytest.approx(float(p.mean()), abs=1e-7)
    assert calibration_module.ece(p, torch.ones_like(p)) == pytest.approx(float((1 - p).mean()), abs=1e-7)
    assert calibration_module.ece(torch.empty(0), torch.empty(0)) == 0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_fail_before_fitting(calibration_module, score):
    with pytest.raises(ValueError, match="finite probabilities"):
        calibration_module.fit_platt([detection_sample([score, 0.5], [1])], 0.5)


def test_no_calibration_and_legacy_summary(calibration_module):
    sample = detection_sample([0.9, 0.7, 0.6], [0, 2])
    identity = calibration_module.fit_calibration([], 0.5, "none")
    assert identity["status"] == "not_requested"
    assert torch.equal(calibration_module.apply_calibration(sample["scores"], identity), sample["scores"])
    before, traces = calibration_module.summarize([sample], 0.5, 0.0, 1.0)
    # Deliberate sigmoid saturation tests that ties cannot change matching or retained candidate IDs.
    after, calibrated_traces = calibration_module.summarize(
        [sample], 0.5, 0.0, calibration={"method": "platt", "slope": 1.0, "bias": 100.0}
    )
    assert before["tp"] == after["tp"] == 2 and before["fp"] == after["fp"] == 1
    assert traces == calibrated_traces
    empty, _ = calibration_module.summarize([], 0.5, 0.0, calibration=identity)
    assert empty["gt"] == 0 and empty["brier"] is None


def test_sigmoid_saturation_does_not_change_matching(calibration_module):
    sample = detection_sample([0.8, 0.9], [0, 1])
    sample["boxes"] = torch.tensor([[3.0, 0.0, 13.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
    sample["gt"] = torch.tensor([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
    mapping = {"method": "platt", "slope": 1.0, "bias": 100.0}
    q = calibration_module.apply_calibration(sample["scores"], mapping)
    assert q[0] == q[1]  # Finite precision loses the original ordering if these are sorted again.
    tied_labels, _ = calibration_module.match(sample["boxes"], sample["classes"], q, sample["gt"], sample["gcls"], 0.5)
    assert int(tied_labels.sum()) == 1
    metrics, _ = calibration_module.summarize([sample], 0.5, 0.0, calibration=mapping)
    assert metrics["tp"] == 2 and metrics["fp"] == 0


@pytest.mark.parametrize("same_name,same_content", [(True, False), (False, True)])
def test_calibration_evaluation_overlap_is_rejected(calibration_module, same_name, same_content):
    cal = [detection_sample([0.5], [0], "cal_image", "cal_hash")]
    evaluation = [
        detection_sample(
            [0.5], [0], "cal_image" if same_name else "eval_image", "cal_hash" if same_content else "eval_hash"
        )
    ]
    with pytest.raises(ValueError, match="split overlap"):
        calibration_module.validate_calibration_splits(cal, evaluation)


def test_main_fits_only_calibration_split_and_saves_parameters(calibration_module, monkeypatch, tmp_path):
    cal = [detection_sample([0.8, 0.8, 0.8, 0.8], [0], "cal_image", "cal_hash")]
    evaluation = [detection_sample([0.2, 0.3], [0, 1], "test_image", "test_hash")]
    seen = []
    actual_fit = calibration_module.fit_calibration

    def fit(samples, eta, method):
        assert samples is cal
        seen.append(method)
        return actual_fit(samples, eta, method)

    class FakeModel:
        def cpu(self):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(calibration_module, "YOLO", lambda _: type("FakeYOLO", (), {"model": FakeModel()})())
    monkeypatch.setattr(
        calibration_module, "collect", lambda model, data, split, *args: cal if split == "cal" else evaluation
    )
    monkeypatch.setattr(calibration_module, "fit_calibration", fit)
    output = tmp_path / "report"
    monkeypatch.setattr(
        calibration_module.sys,
        "argv",
        ["evaluate_occlusion.py", "--weights", "fake.pt", "--eval-split", "test", "--output", str(output)],
    )
    calibration_module.main()
    result = json.loads((output / "calibration.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert seen == ["platt"]
    assert result["fit_split"] == "cal" and result["evaluation_split"] == "test"
    assert result["fit_images"] == ["cal_image"] and result["n"] == 4 and result["positives"] == 1
    assert report["calibrated"]["gt"] == 2 and report["calibration"] == result
