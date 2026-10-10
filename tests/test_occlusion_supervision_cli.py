# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Check the auxiliary CLI contract without importing torch or starting a model."""

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "examples/VisDrone-Occlusion/train_occlusion_supervised.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("occlusion_supervised_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_has_no_environment_side_effects():
    before = dict(os.environ)
    load_cli()
    assert dict(os.environ) == before


def test_scratch_defaults_and_zero_auxiliary_baseline_share_training_controls():
    cli = load_cli()
    parser = cli.build_parser()
    supervised, config = cli.build_configuration(parser.parse_args([]))
    baseline, baseline_config = cli.build_configuration(parser.parse_args(["--aux-weight", "0"]))
    assert supervised["model"] == "yolo11n.yaml" and supervised["pretrained"] is False
    assert supervised["deterministic"] is False
    assert config["coefficient"] == 0.1 and baseline_config["coefficient"] == 0
    assert config["warmup_epochs"] == 5 and config["ramp_epochs"] == 10
    assert {key: value for key, value in supervised.items() if key != "name"} == {
        key: value for key, value in baseline.items() if key != "name"
    }


def test_explicit_pretraining_preserves_path_and_rejects_goa_configuration():
    cli = load_cli()
    args = cli.build_parser().parse_args(["--pretrained", "yolo11n.pt"])
    overrides, _ = cli.build_configuration(args)
    assert overrides["pretrained"] == "yolo11n.pt"
    with pytest.raises(ValueError, match="without GOA"):
        cli.build_configuration(cli.build_parser().parse_args(["--model", "yolo11n-occ-goa.yaml"]))


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--aux-weight", "nan"),
        ("--aux-weight", "-1"),
        ("--aux-hidden", "0"),
        ("--aux-roi-size", "0"),
        ("--aux-warmup", "-1"),
        ("--aux-ramp", "-1"),
        ("--lr0", "0"),
        ("--batch", "0"),
        ("--batch", "-1"),
    ],
)
def test_invalid_auxiliary_configuration_is_rejected(flag, value):
    cli = load_cli()
    with pytest.raises(ValueError):
        cli.build_configuration(cli.build_parser().parse_args([flag, value]))
