import os
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))
os.environ.setdefault("YOLO_CONFIG_DIR", str(repository_root / "runs/yolo_config"))
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
model.occlusion_config = {"size_weight": 0.0, "occ_weight": 0.0, "cap": 2.0}
a, _ = v8DetectionLoss(model)(preds, batch)
b, _ = WeightedDetectionLoss(model)(preds, batch)
assert torch.allclose(a, b, atol=1e-6), (a, b)
model.occlusion_config = {"size_weight": 0.5, "occ_weight": 0.5, "cap": 2.0}
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
assert w[0, 0] > w[0, 1] and (w >= 1).all() and (w <= 2).all()
print("PASS: zero-weight baseline equivalence, finite backward gradients, hard-target ordering and bounded weights")
