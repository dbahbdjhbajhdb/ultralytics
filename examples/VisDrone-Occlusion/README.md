# VisDrone：小目标与遮挡感知训练实验

当前主线是**按目标大小和遮挡等级渐进加权训练，再单独检验置信度校准**。各组保留相同的原始 NMS，便于比较训练带来的变化。这是可运行的实验原型，尚未证明性能提升；普通 `yolo train` 不受影响。

## 数据与目录

你的源目录应保持以下结构，**不要把 occlusion 改名为 labels**：

```text
C:/date/VisDrone_occ/
  images/{train,val}/              # 图片
  labels/{train,val}/              # 标准五列 YOLO .txt
  occlusion/{train,val}/           # 同名 JSON，逐目标遮挡等级
```

上传的 JSON 属于 `visdrone-yolo-occlusion-v1`：`label_row` 为一基行号，`class_id` 与 `bbox_xywh_normalized` 对应标签行，`occlusion` 为 0/1/2。它们是有序等级，不是精确遮挡面积比例。完整导入时还会检查行数、坐标、图像尺寸和标签 SHA256（若提供）。单个 JSON 合格不代表本机所有文件已通过校验。

另外支持原始 VisDrone 8 列标注，以及显式指定列顺序的六列 `yolo-occ`、七列 `yolo-trunc-occ`；不自动猜测额外列。普通五列标签自身不能恢复丢失的遮挡信息。

## Windows 从数据检查到试训

以下命令在本仓库根目录、已激活的 Python 环境中执行。先取得实验分支并安装本仓库：

```powershell
git fetch origin
git switch feat/visdrone-occlusion-training
git pull --ff-only
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

CUDA 12.9 的版本号本身不能确认环境可用，上面的实际 CUDA 运算是检查依据。本云环境只有 CPU，无法访问你的 C 盘，也未验证本机 GPU。

保留源文件，导入到新目录；再将原验证集切成开发验证、校准和内部留出三部分：

```powershell
python examples/VisDrone-Occlusion/import_sidecar_yolo.py --source "C:/date/VisDrone_occ" --output "C:/date/VisDrone_occ_staged"
python examples/VisDrone-Occlusion/split_validation.py --source "C:/date/VisDrone_occ_staged" --output "C:/date/VisDrone_occ_split" --seed 42
```

导入成功会生成 `import_report.json` 和 `dataset.yaml`。切分输出必须是新目录，保留训练集；默认将原验证集按图随机分为约 20% 开发验证、40% 校准、40% 内部留出。正式实验要检查相邻视频帧/同源序列的泄漏；该工具尚未按序列分组。内部留出结果不能称为官方测试服务器结果。

```text
VisDrone_occ_split/
  dataset.yaml
  images/{train,val,cal,test}/
  labels/{train,val,cal,test}/
  metadata/{train,val,cal,test}/    # 遮挡 JSON，已同步导入
  split_manifest.json
```

Windows 复制图片，Linux 建符号链接。五列标签和 JSON 始终一起使用。

先跑相同设置的两组三轮训练，检查本机全链路。加权组显式缩短预热，确保这三轮实际启用权重：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --data "C:/date/VisDrone_occ_split/dataset.yaml" --model yolo11n.pt --device 0 --epochs 3 --imgsz 640 --batch 2 --workers 0 --amp --seed 42 --name smoke_baseline
python examples/VisDrone-Occlusion/train_visdrone.py --data "C:/date/VisDrone_occ_split/dataset.yaml" --model yolo11n.pt --device 0 --epochs 3 --imgsz 640 --batch 2 --workers 0 --amp --seed 42 --weighted --weight-warmup 0 --weight-ramp 3 --name smoke_weighted
```

首次运行会下载 `yolo11n.pt`。确认 `runs/smoke_weighted/weights/{best,last}.pt` 正常保存，并查看 `weight_diagnostics.jsonl` 中的 `progress` 和 `mean_positive_weight`。加权模块只接入这套自定义训练入口，直接使用普通 `yolo train` 不会启用它。

## 正式训练与消融

试训通过后，可先用 960 输入、batch 4、100 轮做正式基线与加权对照。显存不足时减小 batch，所有对照组同步设置；640/960/1280 分辨率应另做消融。显卡与显存未知，下列参数是起点：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --data "C:/date/VisDrone_occ_split/dataset.yaml" --model yolo11n.pt --device 0 --epochs 100 --imgsz 960 --batch 4 --workers 4 --amp --seed 42 --name baseline_s42
python examples/VisDrone-Occlusion/train_visdrone.py --data "C:/date/VisDrone_occ_split/dataset.yaml" --model yolo11n.pt --device 0 --epochs 100 --imgsz 960 --batch 4 --workers 4 --amp --seed 42 --weighted --size-weight 0.5 --occ-weight 0.5 --weight-cap 2 --weight-warmup 5 --weight-ramp 10 --name size_occ_s42
```

Windows 多进程入口有 `main` 保护；若遇 DataLoader 问题，先设 `--workers 0`。CPU 用 `--device cpu --no-amp`。当前支持单 GPU；尚未支持实验配置在 DDP 中传播，也未提供自动 resume。默认 AdamW、lr0=0.001、patience=50，具体值记录于运行文件；可用 `--patience 0` 关闭早停以执行固定轮数。

第一阶段固定模型、初始化、种子、分辨率、增强、训练预算和 NMS，比较四组：

| 组别 | 训练参数 |
|---|---|
| baseline | 不加 `--weighted` |
| size-only | `--weighted --size-weight 0.5 --occ-weight 0` |
| occ-only | `--weighted --size-weight 0 --occ-weight 0.5` |
| size+occ | `--weighted --size-weight 0.5 --occ-weight 0.5` |

各组至少复现多个种子（如 42、43、44），参数只在开发验证集选择。上述 100 轮用于建立起点，不保证充分收敛。

## 遮挡信息怎样参与训练

`OcclusionDataset` 同时加载五列框标签和同名 JSON，通过类别与框坐标核对逐目标遮挡等级；batch 中增加 `occlusion` 张量。`WeightedDetectionLoss` 使用 TaskAlignedAssigner 返回的 GT 索引，把 GT 权重传递给它匹配的正样本 anchor：

```text
w = clip(1 + r(epoch) × [α × 32/(sqrt(area)+32) + β × occ/2], 1, cap)
```

`area` 是缩放后模型输入上的目标像素面积；默认 α=β=0.5、cap=2。默认前 5 轮 r=0，再用 10 轮逐步增至 1。更小的框、较高的遮挡等级得到更高权重。权重用于正类别 BCE、框回归与 DFL，背景类别 BCE 保持原权重；保留原损失分母，不按 batch 权重均值重新归一化，避免一批都是难例时加权被抵消。零系数已验证与原损失一致。

这是依赖样本属性和训练进度的显式规则，尚未实现学习式权重控制器。没有获得正样本匹配的 GT 仍无法直接受益，因此日志同时统计各遮挡等级的 GT 和已匹配 GT 数量。

两组默认均使用 letterbox、HSV 和翻转增强，保持遮挡标签与框行的对应；暂未接入会裁剪/混合目标、改变可见性的 Mosaic、裁剪和 MixUp。`--no-safe-augment` 切换为仅 letterbox。通用 `args.yaml` 的部分增强参数仍显示框架默认值，应以 `occlusion_config.json` 记录的真实 transforms 为准。

每次训练保存 `args.yaml`、`run_parameters.json`、`occlusion_config.json`；加权组额外保存逐轮 `weight_diagnostics.jsonl`。默认输出在仓库 `runs/<name>/`，可通过环境变量 `OCC_EXPERIMENT_ROOT` 改变输出根目录。已有同名运行不会被覆盖，以日志中的实际目录为准。

## 独立校准与评估

先选定 checkpoint，再在 `cal` 拟合，在内部 `test` 评估。示例是 CPU 诊断入口，大数据集可能较慢：

```powershell
python examples/VisDrone-Occlusion/evaluate_occlusion.py --weights runs/size_occ_s42/weights/best.pt --data "C:/date/VisDrone_occ_split" --eval-split test --imgsz 960 --calibration platt --output runs/size_occ_s42/calibration_test
```

分别设置 `--calibration none`、`temperature`、`platt` 并使用不同输出目录做对照。Platt 为 `sigmoid(a × logit(p) + b)`，a>0；温度为原有标量控制。只用校准集 NMS 后候选与 GT 匹配产生的 TP/FP 拟合，推理分数不读取真实遮挡标签。跨校准与评估集合存在同名或相同图片内容时拒绝运行；无正例或无负例时标记 `not_fitted_degenerate_labels` 并保留原始分数。

输出 `report.json`、`calibration.json`、`candidate_cache.pt` 和 `suppression_traces.json`。报告包含遮挡 0/1/2 分层、原图面积小于 32² 的小目标组、precision/recall、ECE/Brier。NMS 和匹配排序始终使用原始分数。全局单调校准不改善候选排序，也不能修复定位或恢复被 NMS 删除的框；它改变固定置信度阈值下的取舍，不能改善相同召回率下的误检数量。**校准不保证误检减少**，还需比较 PR 曲线、固定误报预算下的召回和校准误差，不能只看阈值 0.01 下的召回。

训练验证 `max_det` 默认 1000；该诊断入口不截断候选，且采用 top-class 候选、简化 ignore 区域规则，不是官方 AP，不能与训练验证指标直接横比。正式检测结论需要官方 VisDrone 协议。暂未实现全量 GPU 候选导出、正式分层 AP 或自动阈值搜索。

## 已验证的范围

官方验证集取独立的 8 张训练、4 张校准、4 张验证图像，训练 704 GT、校准 273 GT、验证 309 GT。该子集只用于工程检查；验证集用于 checkpoint 选择，不能称为未经使用的最终测试集。

本轮相同预训练、种子 42、320 输入、三轮训练完成 baseline 与渐进加权两组，权重诊断记录：预热 r=0 时平均权重 1，r=0.5 时约 1.24，r=1 时约 1.48。模型、训练日志和校准报告均能保存。两组标准验证 AP 仍为 0，不能证明加权有效。新模型在 0.01 阈值下原始 TP0/FP0，Platt 后 TP0/FP109，验证侧 ECE/Brier 变差。旧十轮弱模型的温度/Platt 对照同样没有改善误报；不能称为误检抑制成功。

```powershell
python -m pip install pytest
python -m pytest tests/test_occlusion.py tests/test_occlusion_training.py tests/test_occlusion_cli.py tests/test_occlusion_calibration.py -q
```

测试覆盖元数据对齐和哈希、零权重等价、空目标与梯度、难例权重不抵消、预热进度、翻转对应、CLI 配置、校准退化情况与数据隔离。上述功能测试与选取的框架回归检查共 54 项通过。注意力模块、P2、NWD、蒸馏、遮挡预测辅助头与 NMS 替换尚未实现，后续应在上述四组对照完成后分别验证。
