# VisDrone：遮挡元数据、加权损失与置信度校准实验

这是可运行的实验原型，不是已验证的性能改进。新增模块采用自定义 trainer 接口接入，普通 `yolo train` 的行为不变。所有转换都保留输入文件，输出标准五列 YOLO 标签与逐实例 JSON 元数据。

## 支持的数据格式

1. 原始 VisDrone：`images/*.jpg` 与 `annotations/*.txt`，原始标注至少 8 列，最后两列为 truncation、occlusion。
2. 扩展 YOLO：显式指定 `yolo-occ`（`class cx cy w h occ`）或 `yolo-trunc-occ`（`class cx cy w h trunc occ`）。不自动猜测列顺序。
3. 五列 YOLO + JSON：支持 `visdrone-yolo-occlusion-v1`，包含 `class_id`、`bbox_xywh_normalized`、`occlusion`、`occ_valid` 和一基 `label_row`。JSON 放在 `metadata/{split}/`、`occlusion/{split}/` 或同名标签旁；也支持平铺的 `images/`、`labels/`、`occlusion/`。平铺训练目录用 `--split train`，平铺验证目录用 `--split val`；无需改名或移动原始目录。导入时校验标签 SHA256（若有）、目标行数、框坐标和图像尺寸。上传的示例属于此格式。

准备后的目录：

```text
VisDrone_occ_staged/
  dataset.yaml
  images/{train,val,cal,test}/
  labels/{train,val,cal,test}/       # 每行只保留标准五列
  metadata/{train,val,cal,test}/    # 同名 JSON
```

Windows 图片使用复制，Linux 使用符号链接。源数据必须保留；没有遮挡标签的普通 YOLO 数据不能凭空恢复 occlusion。ignore 区域若此前丢失也无法恢复。

## 准备数据

已有五列标签与 JSON（按实际路径替换，下例可在 Windows 上执行）：

```bash
python examples/VisDrone-Occlusion/import_sidecar_yolo.py --source "C:/date/VisDrone_occ" --output "C:/date/VisDrone_occ_staged"
```

六列扩展标签：

```bash
python examples/VisDrone-Occlusion/import_extended_yolo.py --source /data/VisDrone_occ --output /data/VisDrone_occ_staged --label-format yolo-occ
```

原始标注分别转换官方 train/val 到同一输出目录：

```bash
python examples/VisDrone-Occlusion/prepare_visdrone.py --source /data/VisDrone2019-DET-train --output /data/VisDrone_occ_staged --split train
python examples/VisDrone-Occlusion/prepare_visdrone.py --source /data/VisDrone2019-DET-val --output /data/VisDrone_occ_staged --split val
```

正式实验要区分模型选择、校准和最终评估。该工具将原始验证集分为开发验证 20%、校准 40%、内部留出 40%，保持训练集；输出必须是新目录：

```bash
python examples/VisDrone-Occlusion/split_validation.py --source /data/VisDrone_occ_staged --output /data/VisDrone_occ_split
```

此内部留出集不是 VisDrone 官方测试服务器结果。若存在视频相邻帧/来源相关性，应在正式实验前按来源分组切分；当前工具按图像随机切分。

## 训练及消融

安装本仓库（可编辑安装）和 pytest。首次运行 `yolo11n.pt` 会下载预训练权重。两组使用同一权重、种子、数据、分辨率、训练轮数及增强设置：

```bash
python examples/VisDrone-Occlusion/train_visdrone.py --data /data/VisDrone_occ_split/dataset.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --device 0 --name baseline
python examples/VisDrone-Occlusion/train_visdrone.py --data /data/VisDrone_occ_split/dataset.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --device 0 --weighted --size-weight 1 --occ-weight 1 --name size_occ
```

CPU 用 `--device cpu`。默认输出在仓库 `runs/visdrone_occ/` 下；环境变量 `OCC_EXPERIMENT_ROOT` 可指定实验输出根目录。相同运行名若已有结果，Ultralytics 会生成新目录；以输出日志中的实际 save_dir 为准。

对 GT：`w = clip(1 + α × 32/(sqrt(area)+32) + β × occ/2, 1, 3)`。area 是 letterbox 后输入图上的像素面积。通过 TaskAlignedAssigner 的 GT 索引把权重映射到正样本 anchor，并按正样本 anchor 平均值归一化。权重应用到正类别 BCE 项、框回归和 DFL，负类别 BCE 保持原始权重，避免整体抬高全部背景损失。这里“动态”指依赖样本大小/遮挡的权重，不是已经实现的学习式控制器。

分别设置 `--size-weight 1 --occ-weight 0`、`--size-weight 0 --occ-weight 1` 做单因素消融；两者为零时已测试与原始损失一致。运行配置保存到 `occlusion_config.json`，环境变量是当前 trainer 的参数传递方式；恢复训练应显式使用同样权重设置，当前脚本不支持自动 resume。

为确保遮挡信息和 GT 对应，当前两组都只用 letterbox，关闭 Mosaic、裁剪、混合和翻转增强。训练 args.yaml 中通用增强参数仍可能显示默认值，但实际 dataset 覆盖了 transforms，`occlusion_config.json` 记录真实设置。正式增加增强时必须同步过滤/重排元数据，并考虑增强改变可见性的问题。

## NMS 诊断与校准

```bash
python examples/VisDrone-Occlusion/evaluate_occlusion.py --weights runs/visdrone_occ/runs/size_occ/weights/best.pt --data /data/VisDrone_occ_split --eval-split test --imgsz 640 --output /data/results/size_occ
```

输出 report.json、原始候选缓存 candidate_cache.pt、抑制轨迹 suppression_traces.json。遮挡 0/1/2 分层；原图面积小于 32² 为小目标诊断组。评估固定 score threshold 前后的候选覆盖、NMS 删除及一对一召回；不使用 max_det 截断，避免混入 NMS 损失。诊断采用 top-class candidate，与 Ultralytics 某些多标签验证设置不同，不能与标准 AP 直接比较。

校准只在 `cal` 图像中对 NMS 后的候选拟合正标量温度；不读取真实遮挡标签来修正推理分数。标签由同类、IoU≥匹配阈值的一对一匹配产生。正例或负例缺失时返回 `not_fitted_degenerate_labels`，不会伪造成功校准。

标量温度保持排序，因此不改变已发生的 NMS 删除；改变固定分数阈值下的取舍。校准不是误检减少保证。这里校准是后处理模块，而非端到端新网络；当前尚未实现分组正交注意力，避免同时改网络与损失而混淆归因。report 的 ECE/Brier 是候选池条件下的二分类校准诊断，不是涵盖漏检的检测校准全指标。

忽略区域只采用“未匹配预测与 ignore 区域 IoA>0.5 时排除”的原型规则。正式 VisDrone 结论需采用官方协议。与模型选择共用的验证结果不是未经使用的测试结果；请使用 `--eval-split test` 做最终报告。

## 已完成的流程验证

从官方验证集取独立的 8 张训练、4 张校准、4 张评估图像，原始遮挡信息保留；训练侧 704 个目标，校准 273 个，评估 309 个。该划分仅用于工程流程验证。

已完成原始/加权训练，随后使用相同预训练初始化、10 轮、320 输入重新跑两组。独立校准集的加权模型获得有效正负样本并完成温度拟合。检测质量仍很低；标准验证 AP 为零，不能据此支持论文的性能或 NMS 主要致因结论。小子集的校准拟合结果不能证明泛化。在默认诊断报告阈值 0.01 下，加权组温度校准后仅得到 TP=17、FP=4394（GT=309），评估侧 ECE/Brier 也未改善；不能把这一结果称为误检抑制有效。

```bash
pytest tests/test_occlusion.py -q
```

测试覆盖标准/扩展元数据对齐、哈希拒绝、零权重等价、空目标、梯度、抑制轨迹及温度拟合。真实实验还需完整数据、训练预算、多种子、官方评估、置信区间、阈值敏感性和进一步抑制方法对照。
