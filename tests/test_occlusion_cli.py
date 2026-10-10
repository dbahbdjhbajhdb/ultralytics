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
    assert overrides["lr0"] == 0.0005 and overrides["optimizer"] == "SGD"
    assert overrides["patience"] == 0 and overrides["max_det"] == 700
    assert config == {
        "enabled": True,
        "size_weight": 0.25,
        "occ_weight": 0.75,
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
    assert config["enabled"] is False and config["cap"] == 2.0
    args = cli.build_parser().parse_args(["--amp", "--no-safe-augment", "--batch", "-1"])
    overrides, config = cli.build_configuration(args)
    assert config["safe_augment"] is False and overrides["batch"] == -1


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--size-weight", "nan"),
        ("--occ-weight", "-1"),
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
