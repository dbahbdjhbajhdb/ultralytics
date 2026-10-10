# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validate the occlusion CLI contract without downloading models or training."""

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "examples/VisDrone-Occlusion/train_visdrone.py"


def load_cli():
    """Import the guarded entry point with pytest's unrelated argv present."""
    spec = importlib.util.spec_from_file_location("occlusion_train_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_change_runtime_environment(monkeypatch):
    for key in ("WEIGHTED_LOSS", "SIZE_WEIGHT", "OCC_WEIGHT", "YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    load_cli()
    assert dict(os.environ) == before


def test_cli_configuration_preserves_training_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("OCC_EXPERIMENT_ROOT", str(tmp_path))
    cli = load_cli()
    args = cli.build_parser().parse_args(
        [
            "--data",
            "C:/date/VisDrone_occ_split/dataset.yaml",
            "--device",
            "0",
            "--batch",
            "4",
            "--workers",
            "0",
            "--no-amp",
            "--no-deterministic",
            "--seed",
            "17",
            "--lr0",
            "0.0005",
            "--optimizer",
            "SGD",
            "--patience",
            "0",
            "--max-det",
            "700",
            "--weighted",
            "--size-weight",
            "0.25",
            "--occ-weight",
            "0.75",
            "--weight-cap",
            "2",
            "--weight-warmup",
            "3",
            "--weight-ramp",
            "6",
            "--safe-augment",
            "--threads",
            "1",
        ]
    )
    overrides, config = cli.build_configuration(args)
    assert overrides["project"] == str(tmp_path / "runs")
    assert overrides["batch"] == 4 and overrides["workers"] == 0 and overrides["amp"] is False
    assert overrides["device"] == "0" and overrides["seed"] == 17
    assert overrides["deterministic"] is False
    assert overrides["lr0"] == 0.0005 and overrides["optimizer"] == "SGD"
    assert overrides["patience"] == 0 and overrides["max_det"] == 700
    assert config == {
        "enabled": True,
        "size_weight": 0.25,
        "occ_weight": 0.75,
        "joint_weight": 0.0,
        "size_reference": 32.0,
        "cap": 2.0,
        "warmup_epochs": 3,
        "ramp_epochs": 6,
        "safe_augment": True,
    }


def test_baseline_defaults_and_explicit_augmentation_disable():
    cli = load_cli()
    args = cli.build_parser().parse_args([])
    overrides, config = cli.build_configuration(args)
    assert overrides["epochs"] == 100 and overrides["model"] == "yolo11n.pt"
    assert overrides["amp"] is True and config["safe_augment"] is True
    assert overrides["deterministic"] is True
    assert config["enabled"] is False and config["cap"] == 2.0
    args = cli.build_parser().parse_args(["--amp", "--no-safe-augment", "--batch", "-1"])
    overrides, config = cli.build_configuration(args)
    assert config["safe_augment"] is False and overrides["batch"] == -1


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--size-weight", "nan"),
        ("--occ-weight", "-1"),
        ("--joint-weight", "nan"),
        ("--joint-weight", "-1"),
        ("--size-reference", "0"),
        ("--weight-cap", "0.5"),
        ("--weight-warmup", "-1"),
        ("--weight-ramp", "-1"),
        ("--lr0", "0"),
        ("--batch", "0"),
        ("--threads", "0"),
        ("--max-det", "0"),
    ],
)
def test_invalid_training_configuration_is_rejected(flag, value):
    cli = load_cli()
    args = cli.build_parser().parse_args([flag, value])
    with pytest.raises(ValueError):
        cli.build_configuration(args)


def test_attention_yaml_keeps_explicit_pretrained_and_joint_weight():
    cli = load_cli()
    args = cli.build_parser().parse_args(
        [
            "--model",
            "ultralytics/cfg/models/11/yolo11n-occ-goa.yaml",
            "--pretrained",
            "yolo11n.pt",
            "--weighted",
            "--joint-weight",
            "0.5",
            "--size-reference",
            "48",
        ]
    )
    overrides, config = cli.build_configuration(args)
    assert overrides["model"].endswith("yolo11n-occ-goa.yaml")
    assert overrides["pretrained"] == "yolo11n.pt"
    assert config["joint_weight"] == 0.5 and config["size_reference"] == 48.0


def test_checkpoint_model_rejects_ignored_pretrained_override():
    cli = load_cli()
    args = cli.build_parser().parse_args(["--model", "yolo11n.pt", "--pretrained", "other.pt"])
    with pytest.raises(ValueError, match="initializes model YAMLs"):
        cli.build_configuration(args)


def test_existing_entry_enables_scratch_auxiliary_supervision_and_paired_baseline():
    cli = load_cli()
    parser = cli.build_parser()
    supervised, config = cli.build_configuration(parser.parse_args(["--occlusion-supervision"]))
    baseline, baseline_config = cli.build_configuration(
        parser.parse_args(["--occlusion-supervision", "--aux-weight", "0"])
    )
    assert supervised["model"] == "yolo11n.yaml" and supervised["pretrained"] is False
    assert supervised["deterministic"] is False and supervised["close_mosaic"] == 0
    assert config == {"coefficient": 0.1, "hidden": 64, "roi_size": 3, "warmup_epochs": 5, "ramp_epochs": 10}
    assert baseline_config == dict(config, coefficient=0.0)
    assert {key: value for key, value in supervised.items() if key != "name"} == {
        key: value for key, value in baseline.items() if key != "name"
    }


def test_existing_entry_does_not_mix_weighting_with_supervision():
    cli = load_cli()
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--occlusion-supervision", "--weighted"])


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--aux-weight", "nan"),
        ("--aux-weight", "-1"),
        ("--aux-hidden", "0"),
        ("--aux-roi-size", "0"),
        ("--aux-warmup", "-1"),
        ("--aux-ramp", "-1"),
        ("--batch", "-1"),
        ("--model", "yolo11n-occ-goa.yaml"),
    ],
)
def test_existing_entry_rejects_incompatible_auxiliary_configuration(flag, value):
    cli = load_cli()
    with pytest.raises(ValueError):
        cli.build_configuration(cli.build_parser().parse_args(["--occlusion-supervision", flag, value]))


def test_existing_entry_preserves_explicit_supervision_pretraining_and_schedule():
    cli = load_cli()
    args = cli.build_parser().parse_args(
        [
            "--occlusion-supervision",
            "--pretrained",
            "yolo11n.pt",
            "--aux-weight",
            "0.05",
            "--aux-warmup",
            "2",
            "--aux-ramp",
            "4",
            "--deterministic",
        ]
    )
    overrides, config = cli.build_configuration(args)
    assert overrides["pretrained"] == "yolo11n.pt" and overrides["deterministic"] is True
    assert config["coefficient"] == 0.05 and config["warmup_epochs"] == 2 and config["ramp_epochs"] == 4
