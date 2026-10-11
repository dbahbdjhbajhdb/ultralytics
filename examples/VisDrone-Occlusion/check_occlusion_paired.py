#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Run bounded, synthetic CPU checks locally before a long paired training run.

No dataset, CUDA device, network download or pretrained checkpoint is required.
Torch and Ultralytics imports occur after argument parsing, so --help is light.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))

    import torch

    from ultralytics.data.occlusion_pairing import make_occlusion_pairs
    from ultralytics.models.yolo.detect.occlusion_paired import PairedOcclusionModel
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.occlusion_paired_loss import PairedOcclusionLoss, sample_pair_regions

    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = DetectionModel("yolo11n.yaml", nc=10, verbose=False).eval()
    torch.manual_seed(42)
    disabled = PairedOcclusionModel(
        "yolo11n.yaml", nc=10, verbose=False, pair_config={"compensation": False}
    ).eval()
    reference_state, disabled_state = baseline.state_dict(), disabled.state_dict()
    assert reference_state.keys() == disabled_state.keys(), "Disabled compensation changed detector state keys"
    assert all(
        torch.equal(value, disabled_state[name]) for name, value in reference_state.items()
    ), "Disabled compensation changed initialized detector tensors"
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        reference_prediction = baseline(image)[0]
        disabled_prediction = disabled(image)[0]
    assert torch.equal(reference_prediction, disabled_prediction), "Disabled compensation changed predictions"
    print("PASS: disabled compensation preserves initialization and predictions exactly")

    torch.manual_seed(42)
    paired = PairedOcclusionModel("yolo11n.yaml", nc=10, verbose=False).eval()
    with torch.no_grad():
        initialized_prediction = paired(image)[0]
    assert torch.equal(
        reference_prediction, initialized_prediction
    ), "Zero initialized compensation changed original predictions"
    head = paired.model[-1]
    assert int(torch.count_nonzero(head.local_compensation.output.weight)) == 0
    print("PASS: enabled compensation starts with exact detector identity")

    batch = {
        "img": image.clone(),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.08, 0.08]]),
        "batch_idx": torch.zeros(3),
        "cls": torch.tensor([[0.0], [1.0], [2.0]]),
        "occlusion": torch.tensor([0, 2, -1]),
        "occ_valid": torch.tensor([True, True, False]),
    }
    original_image = batch["img"].clone()
    original_metadata = {key: value.clone() for key, value in batch.items() if key != "img"}
    config = {"pair_probability": 1.0, "coverage_min": 0.2, "coverage_max": 0.3, "selection": "grade"}
    torch.manual_seed(42)
    augmented = make_occlusion_pairs(batch, config, progress=1.0)
    assert augmented["pair_indices"].tolist() == [0], "Expected one valid grade 0 pair"
    assert torch.equal(batch["img"], original_image), "Pairing changed the caller's original image"
    assert not torch.equal(augmented["img"], original_image), "Pairing did not change the student view"
    assert all(torch.equal(augmented[key], value) for key, value in original_metadata.items())
    left, top, right, bottom = augmented["pair_rectangles"][0].tolist()
    coverage = (right - left) * (bottom - top) / (32 * 32)
    assert 0.2 - 1e-6 <= coverage <= 0.3 + 1e-6
    assert 16 <= left < right <= 48 and 16 <= top < bottom <= 48
    no_pair = make_occlusion_pairs(batch, config, progress=0.0)
    assert len(no_pair["pair_indices"]) == 0 and torch.equal(no_pair["img"], original_image)

    teacher_prediction = torch.zeros(1, 14, 1)
    teacher_prediction[0, :4, 0] = torch.tensor([32.0, 32.0, 32.0, 32.0])
    teacher_prediction[0, 4, 0] = 0.9
    torch.manual_seed(42)
    reliable = make_occlusion_pairs(batch, config, progress=1.0, teacher_predictions=teacher_prediction)
    assert reliable["pair_indices"].tolist() == [0], "Correct teacher prediction was rejected"
    teacher_prediction[0, 4, 0] = 0.0
    teacher_prediction[0, 5, 0] = 0.9
    unreliable = make_occlusion_pairs(batch, config, progress=1.0, teacher_predictions=teacher_prediction)
    assert len(unreliable["pair_indices"]) == 0, "Wrong-class teacher prediction was accepted"
    print("PASS: paired images preserve GT grades and boxes; reliable teacher gating works")

    channels = (head.p2_channels, *head.input_channels)
    sizes = (16, 8, 4, 2)
    features = [torch.randn(1, channel, size, size, requires_grad=True) for channel, size in zip(channels, sizes)]
    head.train()
    predictions = head(features)
    assert torch.equal(predictions["pair_features"][0], features[1])
    indices = augmented["pair_indices"]
    teacher_feature = torch.randn(1, head.input_channels[0], 8, 8, requires_grad=True)
    selected_boxes = augmented["bboxes"][indices]
    selected_images = augmented["batch_idx"][indices]
    teacher_regions = sample_pair_regions(teacher_feature, selected_boxes, selected_images, roi_size=head.roi_size)
    head_regions = head.pool_regions((teacher_feature,), selected_boxes, selected_images)
    assert torch.equal(teacher_regions, head_regions), "Trainer and head ROI sampling conventions differ"
    assert tuple(teacher_regions.shape) == (1, head.input_channels[0], head.roi_size, head.roi_size)
    augmented["pair_targets"] = teacher_regions
    paired.args = deepcopy(DEFAULT_CFG)
    criterion = PairedOcclusionLoss(paired)
    criterion.set_progress(1.0)
    losses, items = criterion(predictions, augmented)
    assert tuple(losses.shape) == (4,) and tuple(items.shape) == (4,)
    assert torch.isfinite(losses).all() and float(items[3]) > 0
    # Backpropagate only the paired term to check that it reaches the detector
    # and compensation, while the supplied teacher graph remains detached.
    losses[3].backward()
    assert teacher_feature.grad is None, "Pair objective propagated into its teacher target"
    assert features[1].grad is not None and bool(features[1].grad.abs().sum() > 0)
    output_gradient = head.local_compensation.output.weight.grad
    assert output_gradient is not None and bool(output_gradient.abs().sum() > 0)
    assert criterion.statistics["paired_gt"] == 1
    assert criterion.statistics["unknown_gt"] == 1
    print("PASS: finite detection and paired loss; student and compensation gradients; detached teacher")
    print("CHECK_PASSED: synthetic CPU checks passed; dataset correctness and training gains require your experiment")


if __name__ == "__main__":
    main()
