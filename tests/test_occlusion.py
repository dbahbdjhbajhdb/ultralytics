"""Regression checks for opt-in occlusion metadata and loss weighting."""

import importlib.util
import json
from pathlib import Path
import torch
import pytest
from PIL import Image
from ultralytics.data.occlusion import OcclusionDataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.occlusion_loss import WeightedDetectionLoss

SPEC = importlib.util.spec_from_file_location(
    "extended_yolo_import", Path(__file__).parents[1] / "examples/VisDrone-Occlusion/import_extended_yolo.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("label_format,row", [("yolo-occ", "0 .5 .5 .2 .2 2"), ("yolo-trunc-occ", "0 .5 .5 .2 .2 1 2")])
def test_metadata_survives_loading(tmp_path, label_format, row):
    source = tmp_path / "source"
    dest = tmp_path / "converted"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(source / "images/train/a.jpg")
    (source / "labels/train/a.txt").write_text(row + "\n")
    MODULE.convert(source, dest, label_format)
    assert (source / "labels/train/a.txt").read_text() == row + "\n"
    assert len((dest / "labels/train/a.txt").read_text().split()) == 5
    dataset = OcclusionDataset(
        img_path=str(dest / "images/train"),
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=get_cfg(),
        data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
    )
    batch = dataset.collate_fn([dataset[0]])
    assert batch["occlusion"].tolist() == [2.0] and batch["cls"].flatten().tolist() == [0.0]
    (dest / "metadata/train/a.json").write_text(json.dumps({"objects": []}))
    with pytest.raises(ValueError, match="metadata mismatch"):
        OcclusionDataset(
            img_path=str(dest / "images/train"),
            imgsz=64,
            batch_size=1,
            augment=False,
            hyp=get_cfg(),
            data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
        )


@pytest.mark.parametrize("empty", [False, True])
def test_weight_zero_matches_baseline_and_backward(monkeypatch, empty):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = DetectionModel("yolo11n.yaml", nc=10, verbose=False)
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.3, 0.3]]),
        "occlusion": torch.tensor([2.0, 0.0]),
    }
    if empty:
        for key in ["batch_idx", "cls", "bboxes", "occlusion"]:
            batch[key] = batch[key][:0]
    preds = model(batch["img"])
    monkeypatch.setenv("SIZE_WEIGHT", "0")
    monkeypatch.setenv("OCC_WEIGHT", "0")
    original, _ = v8DetectionLoss(model)(preds, batch)
    weighted, _ = WeightedDetectionLoss(model)(preds, batch)
    assert torch.allclose(original, weighted, atol=1e-6)
    monkeypatch.setenv("SIZE_WEIGHT", "1")
    monkeypatch.setenv("OCC_WEIGHT", "1")
    loss, _ = WeightedDetectionLoss(model)(preds, batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("flat", [False, True])
def test_user_schema_hash_and_row_validation(tmp_path, flat):
    import hashlib

    source = tmp_path / "source"
    output = tmp_path / "output"
    for folder in ["images/train", "labels/train", "metadata/train"]:
        (source / folder).mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(source / "images/train/a.jpg")
    label = source / "labels/train/a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    payload = {
        "schema": "visdrone-yolo-occlusion-v1",
        "image_name": "a.jpg",
        "image_size_wh": [64, 64],
        "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
        "objects": [
            {
                "label_row": 1,
                "class_id": 0,
                "bbox_xywh_normalized": [0.5, 0.5, 0.2, 0.2],
                "occlusion": 2,
                "occ_valid": True,
                "truncation": 0,
            }
        ],
        "ignored_annotations": [],
    }
    meta = source / "metadata/train/a.json"
    meta.write_text(json.dumps(payload))
    if flat:
        (source / "images/train/a.jpg").rename(source / "images/a.jpg")
        label.rename(source / "labels/a.txt")
        label = source / "labels/a.txt"
        (source / "occlusion").mkdir()
        meta.rename(source / "occlusion/a.json")
    spec = importlib.util.spec_from_file_location(
        "sidecar_import", Path(__file__).parents[1] / "examples/VisDrone-Occlusion/import_sidecar_yolo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.convert(source, output) == {"train": 1}
    dataset = OcclusionDataset(
        img_path=str(output / "images/train"),
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=get_cfg(),
        data={"names": [str(i) for i in range(10)], "nc": 10, "channels": 3},
    )
    assert dataset[0]["occlusion"].tolist() == [2.0]
    label.write_text("0 0.5 0.5 0.3 0.3\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        module.convert(source, tmp_path / "bad")


def test_nms_trace_and_temperature_fit():
    spec = importlib.util.spec_from_file_location(
        "occlusion_eval", Path(__file__).parents[1] / "examples/VisDrone-Occlusion/evaluate_occlusion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [30.0, 30.0, 40.0, 40.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    classes = torch.tensor([0, 0, 0, 0])
    keep, trace = module.greedy_trace(boxes, scores, classes, 0.5)
    assert keep.tolist() == [0, 2, 3] and trace[0]["removed"] == 1 and trace[0]["suppressor"] == 0
    samples = [
        dict(
            boxes=boxes,
            scores=scores,
            classes=classes,
            keep=keep,
            eligible=torch.ones(3, dtype=torch.bool),
            gt=boxes[[0, 2]],
            gcls=torch.tensor([0, 0]),
        )
    ]
    result = module.fit_temperature(samples, 0.5)
    assert result["status"] == "fitted" and result["positives"] == 2 and 0.05 <= result["temperature"] <= 20
    calibrated = (module.logit(scores) / result["temperature"]).sigmoid()
    assert calibrated.argsort(descending=True).tolist() == scores.argsort(descending=True).tolist()
