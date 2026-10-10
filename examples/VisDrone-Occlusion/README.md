# VisDrone：遮挡加权损失 + 分组正交注意力

本版按用户要求只修改代码，训练和效果验证由用户执行。新增注意力与联合加权尚未运行训练、前向、评估或 pytest；只做源码、语法、格式和模型配置的静态检查。普通 YOLO11 模型仍可作为对照，NMS 保持原实现。

## 直接运行你的数据

在仓库根目录、已激活且支持 CUDA 的 Python 环境中执行：

```powershell
git fetch origin
git switch feat/visdrone-occlusion-training
git pull --ff-only
python -m pip install -e .
```

已提供 `examples/VisDrone-Occlusion/visdrone_occ.yaml`，其中 `path: C:/date/VisDrone_occ` 对应你给出的路径。源目录保持以下结构，**不要将 occlusion 改名为 labels，也不要移动原 labels**：

```text
C:/date/VisDrone_occ/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt              # 标准五列：class cx cy w h
  labels/val/*.txt
  occlusion/train/*.json          # 与图片、txt 同名
  occlusion/val/*.json
```

Dataset 同时读取五列框标签和 JSON 遮挡标签。支持上传示例的 `visdrone-yolo-occlusion-v1`，也支持 `metadata/{split}` 目录。加载时核对类别、框坐标和标签 SHA256（若提供）；错误对应会报错。遮挡 0/1/2 是等级编码，不是精确遮挡比例。

JSON 中明确标记 `occ_valid: false` 的目标（例如原始重复框的遮挡 0/1 冲突，经转换记为 `occlusion: -1`）保留为遮挡未知。检测框、类别、普通检测损失和小目标权重全部保留；仅该目标的遮挡与小目标×遮挡加权项为零，其余有效遮挡标签继续正常加权。读取器输出独立的 `occ_valid` 布尔掩码，加载日志报告未知目标和图片数量；`weight_diagnostics.jsonl` 单独记录 `gt_with_unknown_occlusion`、`matched_gt_with_unknown_occlusion`，不会把未知目标算作无遮挡。未明确标为无效的非法等级仍报错。无需修改源 JSON/TXT 或删除缓存。

上述未知掩码支持当前训练入口。可选的 `import_sidecar_yolo.py` 严格导入工具仍拒绝未知标注，独立评估脚本尚未接入此掩码；直接训练无需运行它们。

GPU 环境可自行检查：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

先由你运行完整方案：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --data examples/VisDrone-Occlusion/visdrone_occ.yaml --model ultralytics/cfg/models/11/yolo11n-occ-goa.yaml --pretrained yolo11n.pt --weighted --size-weight 0.5 --occ-weight 0.5 --joint-weight 0.5 --size-reference 32 --weight-cap 2 --weight-warmup 5 --weight-ramp 10 --epochs 100 --imgsz 960 --batch 4 --workers 0 --device 0 --amp --seed 42 --patience 0 --name goa_size_occ_s42
```

`--pretrained yolo11n.pt` 很重要：新模型从 YAML 建结构，通过该参数加载原始 YOLO11n 中形状匹配的预训练层，而非仅构建随机初始化网络。日志中会显示 transferred weights。新增注意力参数自行学习，VisDrone 的 10 类输出也会按类别数量建立。首次使用检查点可能需要联网下载。

如果显存不足，先减小 batch；降低分辨率时所有对照组同步修改。Windows 建议从 `--workers 0` 开始，再按本机情况增大。以上 100 轮是运行配置，不代表已确定的最优预算。当前支持单 GPU，尚未提供自动 resume。

若参照原来 `epochs=200 imgsz=640 batch=4 device=0 workers=0 deterministic=False` 的设置，使用下面的一行。`--no-deterministic` 对应 `deterministic=False`，默认仍为 True；普通 `yolo detect train` 不会启用本项目的遮挡加权训练器。

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --data examples/VisDrone-Occlusion/visdrone_occ.yaml --model ultralytics/cfg/models/11/yolo11n-occ-goa.yaml --pretrained yolo11n.pt --weighted --size-weight 0.5 --occ-weight 0.5 --joint-weight 0.5 --weight-cap 2 --weight-warmup 5 --weight-ramp 10 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 0 --no-deterministic --amp --seed 42 --patience 0 --name goa_size_occ_200
```

若你希望先自行做三轮检查，将上述命令改为 `--epochs 3 --imgsz 640 --batch 2 --weight-warmup 0 --weight-ramp 3`，换一个运行名。短训练必须缩短预热，否则前三轮默认权重均为 1。

## 本版改动

### 损失权重

代码：`ultralytics/utils/occlusion_loss.py`。对每个 GT 计算：

```text
s = size_reference / (sqrt(input_pixel_area) + size_reference)
o = occlusion / 2
w = clip(1 + r(epoch) × (αs + βo + γso), 1, cap)
```

仅对遮挡已知的目标使用上述 `o`。遮挡未知目标保持 `occlusion=-1`、`occ_valid=false`，在权重公式中屏蔽 β、γ 项，得到 `w = clip(1 + r(epoch) × αs, 1, cap)`，不删除检测目标。

α 为 `--size-weight`，β 为 `--occ-weight`，γ 为 `--joint-weight`。目标越小、遮挡等级越高，权重越高；γ 项专门增加同时小且被遮挡目标的权重。`joint-weight` 默认 0 以保持旧配置兼容，上述完整方案显式设为 0.5。`size-reference` 默认 32，是模型输入图上的像素尺度。

默认前 5 轮不加权，再用 10 轮渐增至完整权重。权重通过 TaskAlignedAssigner 的匹配 GT 索引作用于正类别 BCE、bbox 回归和 DFL；背景类别 BCE 不加权。保留原损失分母，避免整批难例的权重被均值归一化抵消。未获得正样本匹配的 GT 无法直接受益。

### 正交注意力

代码：`ultralytics/nn/modules/orthogonal_attention.py`。模型配置：`ultralytics/cfg/models/11/yolo11n-occ-goa.yaml`。

在 YOLO11 颈部融合后的第 16 层 P3/8、第 19 层 P4/16 使用 `C3k2GOA`。P3 保留较高分辨率细节，P4 提供更大范围的融合特征；Detect 输入仍是 `[16,19,22]`，原 C3k2 参数名称和层编号保留，便于预训练权重迁移。

模块把特征池化到 7×7，用 8 个固定归一化二维 DCT 空间基提取不同投影，再由 8 组通道 MLP 产生权重。空间基满足离散内积正交；没有把可学习通道或 MLP 权重称为正交。门控为 `1 + 0.5 × tanh(logits)`，末端投影零初始化，因此初始不改变原特征，随后学习增强或抑制通道。训练和推理均只输入图像特征，推理不读取遮挡真值。

这是本仓库的实验性分组正交空间描述注意力，不是对某篇特定“分组正交注意力”论文的复现。检测收益、速度及导出兼容性均待你验证。

## 对照组怎么运行

保持相同数据、预训练检查点、种子、分辨率、轮数和增强，至少比较 baseline、loss-only、GOA-only、GOA+loss。完整方案见前面的命令；其余三组如下：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --data examples/VisDrone-Occlusion/visdrone_occ.yaml --model yolo11n.pt --epochs 100 --imgsz 960 --batch 4 --workers 0 --device 0 --amp --seed 42 --patience 0 --name baseline_s42
python examples/VisDrone-Occlusion/train_visdrone.py --data examples/VisDrone-Occlusion/visdrone_occ.yaml --model yolo11n.pt --weighted --size-weight 0.5 --occ-weight 0.5 --joint-weight 0.5 --size-reference 32 --weight-cap 2 --weight-warmup 5 --weight-ramp 10 --epochs 100 --imgsz 960 --batch 4 --workers 0 --device 0 --amp --seed 42 --patience 0 --name loss_only_s42
python examples/VisDrone-Occlusion/train_visdrone.py --data examples/VisDrone-Occlusion/visdrone_occ.yaml --model ultralytics/cfg/models/11/yolo11n-occ-goa.yaml --pretrained yolo11n.pt --epochs 100 --imgsz 960 --batch 4 --workers 0 --device 0 --amp --seed 42 --patience 0 --name goa_only_s42
```

损失内部可再分 size-only（β=γ=0）、occ-only（α=γ=0）、size+occ（γ=0）、size+occ+joint（γ=0.5）。重复种子时修改 `--seed` 和 `--name`。参数只在开发验证集选择。

所有组使用相同 letterbox、HSV、翻转，保持遮挡标签对应。暂未接入改变可见性的 Mosaic、裁剪、MixUp；`--no-safe-augment` 可改为仅 letterbox。这是受控实验基线，不等同于充分调参的标准 YOLO 结果。

## 输出与预测

默认输出在仓库 `runs/<name>/`。同名目录已有结果时自动生成新目录，以日志中的 save_dir 为准：

- `weights/best.pt`、`weights/last.pt`：模型权重。
- `results.csv`：逐轮训练与验证指标。
- `run_parameters.json`、`args.yaml`：运行参数。
- `occlusion_config.json`：真实损失系数、增强、注意力层号及预训练来源。
- `weight_diagnostics.jsonl`：加权组的进度、有效权重、各遮挡等级匹配 GT 数量。

使用训练后的 checkpoint 即可预测，无需重新指定模型 YAML：

```powershell
yolo predict model=runs/goa_size_occ_s42/weights/best.pt source="C:/date/VisDrone_occ/images/val" imgsz=960 device=0 conf=0.25 max_det=1000
yolo val model=runs/goa_size_occ_s42/weights/best.pt data=examples/VisDrone-Occlusion/visdrone_occ.yaml imgsz=960 device=0 batch=4 max_det=1000
```

普通 `yolo val` 是框级验证，不读取遮挡 JSON 做分层；custom trainer 才启用遮挡加权损失。完整 VisDrone 结论仍需官方协议。

## 可选：数据校验导入与独立校准

无需为了上述训练改名目录。若要更严格地核对 JSON 行号/图像尺寸，或准备独立校准集，可使用已有导入工具，始终保留源数据：

```powershell
python examples/VisDrone-Occlusion/import_sidecar_yolo.py --source "C:/date/VisDrone_occ" --output "C:/date/VisDrone_occ_staged"
python examples/VisDrone-Occlusion/split_validation.py --source "C:/date/VisDrone_occ_staged" --output "C:/date/VisDrone_occ_split" --seed 42
```

该工具保留训练集，把原验证集按图随机分为开发验证20%、cal40%、内部test40%；输出必须为新目录。正式实验应按来源/序列检查相邻帧泄漏，当前未按序列分组。需要使用此划分时，**所有训练组**的 `--data` 改为 `C:/date/VisDrone_occ_split/dataset.yaml`。

在选定模型后，独立校准/遮挡诊断命令如下；该入口是 CPU 原型，大数据集可能慢：

```powershell
python examples/VisDrone-Occlusion/evaluate_occlusion.py --weights runs/goa_size_occ_s42/weights/best.pt --data "C:/date/VisDrone_occ_split" --eval-split test --imgsz 960 --calibration platt --output runs/goa_size_occ_s42/calibration_test
```

校准需使用对应 split 数据训练的 checkpoint。可设 `--calibration none|temperature|platt` 做对照。Platt 只在 cal 拟合，推理不读遮挡真值；全局单调校准不能改善检测排序或保证减少误报。该诊断报告含遮挡0/1/2、原图面积小于32²的小目标召回和候选池 ECE/Brier；不是官方 AP，也不包含标准验证的 max_det 截断，不能直接横比。

## 验证状态

当前版新增 GOA、联合权重及预训练入口，仅静态检查。旧提交 d1c653f6 的54项测试和CPU试训不覆盖当前新增注意力，不能作为本版运行或效果证据。

如你希望自行检查模块、权重迁移、损失与 CLI，可运行以下可选检查。本轮未执行这些测试：

```powershell
python -m pip install pytest
python -m pytest tests/test_orthogonal_attention.py tests/test_occlusion.py tests/test_occlusion_training.py tests/test_occlusion_cli.py tests/test_occlusion_calibration.py -q
```

未实施 P2、NWD、蒸馏或学习式权重控制器。本轮不同时叠加这些结构，以便你验证损失与注意力各自的作用。
