# VisDrone：GT 区域遮挡辅助监督

本入口在标准 YOLO11n 上增加训练期的遮挡分类任务。每个遮挡标签有效的 GT 都可以提供辅助监督，不要求这个 GT 先被检测分配器匹配到正样本。普通检测损失保持原有形式，模型配置使用 `yolo11n.yaml`。

新监督损失已接入项目原来的 `train_visdrone.py`：增加 `--occlusion-supervision` 即使用 `OcclusionSupervisedTrainer`、训练期辅助头和 `OcclusionSupervisionLoss`。该模式使用标准模型，不与原来的 GOA 模型或 `--weighted` 同时启用。独立入口 `train_occlusion_supervised.py` 仍保留，使用同一套监督损失。

## 先更新代码

你下载的 `cc2d8ee9` ZIP，以及之前只替换 `occlusion.py`、`occlusion_loss.py` 的修复，都不包含新的辅助任务和已接入监督损失的原训练入口。请先安装本方案对应的修改；文件下载和替换说明以交付时提供的固定提交为准。保留你自己建立的 `visdrone2019-occ.yaml` 和原数据。

这些改动本轮只进行静态检查，没有运行模型、训练、验证或 pytest。下列命令由你在自己的 GPU 环境执行，效果需要实验确认。

提供 `update_occlusion_supervision.py` 用于旧 ZIP：下载交付的固定提交版本，在外层解压目录或内层项目目录运行 `python update_occlusion_supervision.py --root . --ref <交付提交的完整 SHA>`。脚本自动定位一层内的项目，先下载并核对所有文件的 SHA256 和 Python 语法，再备份并写入。更新包含原训练入口和新版遮挡读取器，不需要重新转换数据；自己的数据 YAML 和已有训练结果不会被覆盖。外层目录完成更新后，训练命令仍需在打印的 `ROOT` 内层目录执行。

## 数据保持原目录

```text
C:/date/VisDrone_occ/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  occlusion/train/*.json
  occlusion/val/*.json
```

TXT 仍是标准五列 `class cx cy w h`，JSON 提供同一目标的遮挡等级。图片、TXT、JSON 使用相同文件名主体。遮挡等级 `0/1/2` 分别代表无遮挡、部分遮挡、重度遮挡，是离散标签，不是连续遮挡比例。

`occ_valid: false` 的冲突或未知记录不参与遮挡分类，但对应的检测框和类别仍保留。有效记录仍需是数值 `0/1/2`；未知标签不会被改成等级 0。数据读取器会检查 JSON 与 TXT 的类别、框坐标以及可选 SHA256。

本例使用你建立的配置文件：

```text
ultralytics/cfg/datasets/visdrone2019-occ.yaml
```

配置中的 `path` 指向实际数据集根目录，`train: images/train`、`val: images/val`。不要将 `occlusion` 重命名为 `labels`。

## 在 PyCharm 终端训练

选择你的 `cuda129` Python 环境，在包含 `pyproject.toml`、`examples`、`ultralytics` 的内层项目目录执行。按你当前 ZIP 的位置，先进入：

```powershell
cd "C:\Users\123\Downloads\ultralytics-cc2d8ee923fee079909c9e3a104267af221cf06d\ultralytics-cc2d8ee923fee079909c9e3a104267af221cf06d"
```

辅助监督组，从随机初始化训练：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --occlusion-supervision --model yolo11n.yaml --data ultralytics/cfg/datasets/visdrone2019-occ.yaml --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 0 --aux-weight 0.1 --aux-warmup 5 --aux-ramp 10 --aux-hidden 64 --aux-roi-size 3 --amp --seed 42 --optimizer AdamW --lr0 0.001 --patience 0 --name occ_supervised_scratch_200
```

配对基线使用同一入口并保留 `--occlusion-supervision`，只将辅助损失系数设为 0：

```powershell
python examples/VisDrone-Occlusion/train_visdrone.py --occlusion-supervision --model yolo11n.yaml --data ultralytics/cfg/datasets/visdrone2019-occ.yaml --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 0 --aux-weight 0 --aux-warmup 5 --aux-ramp 10 --aux-hidden 64 --aux-roi-size 3 --amp --seed 42 --optimizer AdamW --lr0 0.001 --patience 0 --name occ_supervised_baseline_200
```

两组保持同样的数据、模型配置、初始化来源、种子、增强、优化器、轮数和输入分辨率。不要将辅助组的结果直接与使用不同增强或预训练设置的旧 GOA 结果比较。

以上命令不指定 `--pretrained`，检测网络从随机初始化训练。如果改用预训练，在**两组命令中都增加** `--pretrained yolo11n.pt`，并换新的运行名。日志中的权重迁移数量只说明成功加载了形状匹配的检测参数，新的辅助分类头仍需学习。

默认 `deterministic=False`，Windows 使用 `workers=0`。AMP 默认启用；若需要关闭，在两组中都改用 `--no-amp`。显存不足时先减小 batch，所有对照组同步修改。

## 遮挡标签如何产生梯度

1. 从 Detect 接收的 P3、P4、P5 特征中，按当前输入图上的归一化 GT 框获取区域特征。
2. 对三个尺度的 GT 区域分别采样、池化；默认区域采样尺寸为 `3×3`。
3. 拼接三个尺度的区域描述，用小型 MLP 预测遮挡等级 `0/1/2`。
4. 对所有有效遮挡 GT 计算交叉熵，并将梯度传回区域特征及共享检测网络。

```text
L_total = L_detection + lambda × r(epoch) × CE_occlusion
```

`--aux-weight` 设置 lambda，默认 `0.1`。`--aux-warmup 5` 表示前 5 轮辅助贡献为 0，随后通过 `--aux-ramp 10` 在 10 轮内逐步增加到完整系数。检测任务一直正常训练。

辅助分支使用 GT 区域，因此没有依赖 TAL 的正样本匹配门槛；普通检测损失中的分配过程仍照常进行。这里“直接监督所有有效 GT”指遮挡分类任务，并不意味着每个目标都已被正确检测。

辅助系数设为 0，或仍在预热阶段时，辅助项不提供梯度。短训练若只用于你自己的流程检查，可同步缩短预热和渐增轮数；不能据此前几轮的零辅助损失判定 JSON 没有读取。

## 日志和输出

默认保存到 `runs/<name>/`，已有同名目录时自动追加编号；以启动日志中的实际目录为准。

- `weights/best.pt`、`weights/last.pt`：完整 checkpoint。
- `results.csv`：逐轮训练与验证指标。
- `occlusion_supervision_diagnostics.jsonl`：遮挡辅助任务的原始交叉熵、有效样本数量、各等级准确率及系数进度。

诊断字段包括 `gt_by_occlusion`、`unknown_gt`、`supervised_gt`、`raw_mean_ce`、`accuracy_by_occlusion`、`mean_weighted_aux_loss` 和 `progress`。关闭辅助时不执行 ROI 分类，`supervised_gt=0`、原始 CE 与各等级准确率为 null；这时 `gt_by_occlusion` 仍反映读取到的有效标签分布。预热阶段系数尚未启用，但辅助组仍计算原始 CE 与准确率供检查。

训练日志的第四项 `occ_loss` 是**加入总损失的加权贡献**，包含 lambda 和进度系数，并非原始交叉熵。检查原始交叉熵和样本数量时读取诊断 JSONL。未出现的等级没有可解释的等级准确率，不应当作为准确率为 0 的类别来分析。

验证和推理只执行检测任务，不计算遮挡辅助分支。因此验证中的 `occ_loss=0` 是设计行为，不表示训练没有使用遮挡标签。训练期遮挡分类准确率用于检查辅助任务是否学习，不能替代验证集上的检测精度和召回。

## 保存的模型如何使用

Detect 的原输出形式保留，推理输入仍是图片，无需 GT 框或遮挡标签。预测命令可在**包含新 custom 类的这份仓库环境**中执行：

```powershell
python -c "from ultralytics import YOLO; YOLO('runs/occ_supervised_scratch_200/weights/best.pt').predict(source='C:/date/VisDrone_occ/images/val', imgsz=640, device=0, conf=0.25, max_det=1000)"
```

在项目根目录用这条 Python 命令，可优先导入当前仓库代码。完整 checkpoint 包含本方案的自定义 model 和辅助头。即使推理不计算辅助任务，加载 checkpoint 也需要这些类所在的代码。不能直接承诺它可在未修改的 pip 版 Ultralytics 中加载。导出以及自动恢复训练也需要另行验证。

## 实验解释和限制

辅助监督的目标是让共享特征包含更多与遮挡程度相关的信息；能否提高遮挡小目标检测的召回，仍需与配对基线比较。优先报告整体检测指标，再按遮挡等级、目标尺度统计检测召回，检查误检和推理成本。辅助分类损失下降本身不是检测效果提升的证据。

遮挡等级可能与目标尺度、类别和场景相关。GT ROI 特征也可能包含邻近目标和背景，尤其在密集场景中；这既提供上下文，也可能产生干扰。标签冲突、少数等级样本不足和类别分布不均均应记录，不通过猜测填补未知遮挡标签。

为保持遮挡标签与目标对应，默认仍使用受控的 letterbox、HSV 和翻转增强，不加入改变目标可见性的 Mosaic、裁剪和 MixUp。两组增强保持一致后，得到的是这个受控设置下的结论；如后续扩大增强范围，需要同步调整标签处理和基线。
