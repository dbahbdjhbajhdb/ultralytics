import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/ultralytics-config")
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.occlusion_loss import WeightedDetectionLoss

torch.set_num_threads(2)
torch.manual_seed(42)
model = DetectionModel("yolo11n.yaml", nc=10, verbose=False)
model.args = DEFAULT_CFG
model.train()
batch = {
    "img": torch.rand(2, 3, 64, 64),
    "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    "cls": torch.tensor([[0.0], [3.0], [0.0]]),
    "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1], [0.5, 0.5, 0.4, 0.4]]),
    "occlusion": torch.tensor([0.0, 2.0, 1.0]),
}
preds = model(batch["img"])
os.environ["SIZE_WEIGHT"] = "0"
os.environ["OCC_WEIGHT"] = "0"
a, _ = v8DetectionLoss(model)(preds, batch)
b, _ = WeightedDetectionLoss(model)(preds, batch)
assert torch.allclose(a, b, atol=1e-6), (a, b)
os.environ["SIZE_WEIGHT"] = "1"
os.environ["OCC_WEIGHT"] = "1"
c, _ = WeightedDetectionLoss(model)(preds, batch)
assert torch.isfinite(c).all()
c.sum().backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
loss = WeightedDetectionLoss(model)
gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 64.0, 64.0]]])
w = loss.positive_weights(
    {"batch_idx": torch.tensor([0, 0]), "occlusion": torch.tensor([2.0, 0.0])},
    gt,
    torch.tensor([[0, 1]]),
    torch.tensor([[True, True]]),
)
assert w[0, 0] > w[0, 1] and torch.allclose(w.mean(), torch.tensor(1.0))
print("PASS: zero-weight baseline equivalence, finite backward gradients, hard-target ordering and normalization")
