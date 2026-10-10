"""User-runnable checks for grouped orthogonal attention; no training or dataset download."""

import pytest
import torch

from ultralytics.nn.modules import C3k2, C3k2GOA, GroupedOrthogonalAttention


def test_spatial_basis_is_orthonormal():
    attention = GroupedOrthogonalAttention(64)
    basis = attention.dct_basis.flatten(1)
    assert torch.allclose(basis @ basis.T, torch.eye(8), atol=1e-5)
    assert "dct_basis" not in attention.state_dict()


@pytest.mark.parametrize("shape", [(2, 64, 12, 9), (1, 24, 4, 5)])
def test_initial_attention_is_identity_and_gate_can_learn(shape):
    attention = GroupedOrthogonalAttention(shape[1])
    features = torch.randn(shape, requires_grad=True)
    output = attention(features)
    assert torch.equal(output, features)
    output.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert attention.projection[-1].weight.grad.abs().sum() > 0


def test_c3k2_transfer_preserves_features_and_split_path():
    original = C3k2(128, 64, n=1).eval()
    modified = C3k2GOA(128, 64, n=1).eval()
    missing, unexpected = modified.load_state_dict(original.state_dict(), strict=False)
    assert missing and all(key.startswith("attention.") for key in missing)
    assert not unexpected
    features = torch.randn(1, 128, 12, 12)
    with torch.no_grad():
        assert torch.allclose(modified(features), original(features), atol=1e-6)
        assert torch.allclose(modified.forward_split(features), modified(features), atol=1e-6)


def test_attention_model_builds_with_correct_detection_scales():
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo11n-occ-goa.yaml", nc=10, verbose=False)
    assert isinstance(model.model[16], C3k2GOA) and isinstance(model.model[19], C3k2GOA)
    assert model.model[-1].f == [16, 19, 22]
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


@pytest.mark.parametrize("options", [{"groups": 0}, {"num_filters": 50}, {"residual_scale": float("nan")}])
def test_invalid_attention_options_fail(options):
    with pytest.raises(ValueError):
        GroupedOrthogonalAttention(64, **options)
