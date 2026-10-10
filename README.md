# Derm7pt 多模态分类基础项目

根据 [交接文档](derm7pt_ai_handoff.md) 建立。研究机制尚未确定，当前交付是可训练、可恢复、可导出病例级预测的公共 baseline，实验结果尚未产生。

## 当前实现

| 编号 | 图像输入 | 融合 |
| --- | --- | --- |
| B01 | 临床 | 单编码器 |
| B02 | 皮肤镜 | 单编码器 |
| B04 | 临床和皮肤镜 | 两个独立编码器，pooled mean |
| B05 | 临床和皮肤镜 | concat 后投影 |
| B06 | 临床和皮肤镜 | 逐通道 gated fusion |

研究骨干使用 `torchvision` ConvNeXt-Base；预训练标识为 `ConvNeXt_Base_Weights.IMAGENET1K_V1`。单模态不构造另一分支。五类顺序固定为 BCC、NEV、MEL、MISC、SK。默认使用 CE、固定末轮权重、EMA、argmax、无 TTA；没有元数据输入、属性监督或创新机制。`tiny_cnn` 专用于合成流程测试，正式配置禁止使用。

模型层预留七项分类属性头、masked CE 和 CB Focal 的实现及契约测试；首版配置入口暂时禁止开启，避免把未经实验确认的选项混进基线。已接入开发五折与 pooled OOF；空间交叉注意力、源 V2 迁移、refit、B03 概率融合和候选研究机制列为后续阶段。

## 目录

```text
configs/                 官方留出/开发五折协议、基线运行计划
data/manifests/          规范化病例、五类字典、数据审计
src/data/                官方零基索引、输入/监督隔离、图像增强
src/models/              编码器、融合注册表、诊断及预留属性头
src/evaluation/          固定五类病例级指标和预测校验
src/train.py             唯一训练入口
src/train_utils.py       EMA、梯度累积、AMP、完整epoch断点状态
src/infer.py             唯一推理入口与最终test保护
src/provenance.py        源码、配置、数据、权重身份及环境记录
scripts/                 数据准备、审计、计划、汇总、合成smoke
tests/                   索引、数据泄漏、模型、恢复、产物身份契约
docs/                    协议、源码核验、开发决策、验证与证据
outputs/                 运行计划、smoke及未来研究产物
```

## 环境

在项目根目录执行。建议为新课题建独立环境，不修改 ISIC2025 环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch 与 torchvision 需安装匹配版本。CPU 验证使用了 PyTorch 2.8.0+cpu / torchvision 0.23.0+cpu；准确环境见 `requirements.cpu-tested.lock.txt` 和 `docs/validation.md`。GPU 环境以实际驱动与 PyTorch CUDA build 为准。首次正式使用 ConvNeXt 预训练权重会下载，离线环境需先准备缓存；流程测试不下载预训练权重。

## 数据准备

配置保持 `dataset.root: null`，不擅自将旧项目数据认定为新课题正式来源。已对 `/mnt/d/DeepLearning/code/ISIC2025/data/derm7pt` 作只读候选审计，产物在 `data/manifests/`；病例数为 1011，官方 train/valid/test 为 413/203/395。数据使用边界和问题见 [data/README.md](data/README.md)。

```bash
python scripts/prepare_derm7pt.py --config configs/base.yaml --root /path/to/derm7pt
python scripts/audit_dataset.py --manifest data/manifests/samples.csv --root /path/to/derm7pt
```

如发布包存在唯一的文件名大小写差异，可显式使用 `--normalize-path-case` 生成可移植清单；保存原始引用与修复记录，原 CSV、图像和官方划分保持原样。正式运行必须先解决审计报告中的证据缺口。诊断、七点清单和总分仅作监督，病例标识仅用于对齐；不会伪造 age、skin_tone 或 patient_id。

## 检查与运行

```bash
python -m pytest -q
python scripts/smoke_test.py --device cpu --output-dir outputs/smoke_cpu
python scripts/smoke_test.py --protocol dev_cv5 --device cpu --output-dir outputs/smoke_cv42_cpu
python scripts/smoke_test.py --device cuda --output-dir outputs/smoke_gpu
python scripts/run_experiments.py --suite configs/experiments/baselines.yaml --dry-run
python scripts/run_experiments.py --suite configs/experiments/baselines.yaml --preflight-only
```

默认计划是 5 个图像基线 × 3 个种子（101、3407、777），共 15 次训练。draft 允许生成计划，实际训练会一次提示缺失的 `dataset.root`、`dataset.clinical_substitution_policy`、`train.epochs`；这些不是已经确定的研究参数。224 输入、batch 16、学习率与 EMA decay 是有出处的开发起点，训练预算要通过开发数据确定。

服务器可用 `bash scripts/train_server.sh --data-root /data/derm7pt` 启动独立开发作业：默认五个基线 × 三个种子、25 epochs、图像 CE 和末轮 EMA，训练 train、评价 valid，test 保持封存。先执行 `--preflight-only` 核对环境与数据；依赖、SSH 后台运行、单基线和恢复命令见 [服务器训练说明](docs/server_training.md)。默认 15 次运行按每次 6 GiB 预留约 90 GiB，这是保守预算，尚未进行真实训练验证。

本轮先用 **seed 42 做开发五折**，在服务器执行：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42
```

此协议默认 B01/B02/B04 × 五折，共15次；划分种子固定20261004，每次25轮、batch1/梯度累计16。仅使用官方train＋valid的616例，不改原索引，test封存。入口自动生成独立folds.csv、训练各折、校验预测并输出pooled OOF和汇总；先追加 `--preflight-only` 检查，追加 `--dry-run` 仅看计划。B05/B06可通过 `--only B01 B02 B04 B05 B06` 显式加入，扩展为25次。

将核验后的决定写入自己的完整配置，再开始单项开发运行：

```bash
python -m src.train --config configs/local_development.yaml --seed 101 --preflight-only
python -m src.train --config configs/local_development.yaml --seed 101
python -m src.train --config outputs/runs/<run_id>/resolved_config.yaml --seed 101 --resume outputs/runs/<run_id>/checkpoints/final_regular.pt
python -m src.infer --config outputs/runs/<run_id>/resolved_config.yaml --checkpoint outputs/runs/<run_id>/checkpoints/final_ema.pt
python scripts/collect_results.py --suite-id derm7pt_baseline_v1
```

`configs/local_development.yaml` 是需自行填写的本地配置路径示例，不是已有配置。官方留出协议训练 train、评价 valid；开发五折按固定 heldout 折训练和验证。最终 test 要在训练之前冻结完整配置，使用同一 `stage: frozen` 配置训练及推理，并显式指定 `--partition test --final-evaluation`。不能通过训练后修改 stage 将开发 checkpoint 变成正式 checkpoint；配置变动会改变身份。当前数据审计仍有临床替代状态等待核验项，正式入口会拒绝未就绪数据。

每次训练保存 Regular 和 EMA、optimizer/scheduler/scaler、Python/NumPy/Torch/CUDA/loader RNG、epoch、history、完整配置及源码/数据哈希。同一Python/依赖环境和设备下支持 epoch 边界恢复；没有承诺 batch 中间恢复。已有运行须显式 `--resume` 才能校验后复用或恢复，不自动覆盖。CV fold参与配置、路径与checkpoint身份，禁止跨折复用。汇总检查产物哈希与病例对齐；缺少完整种子或CV折时拒绝完整汇总。

项目使用 Git 跟踪源码、配置、测试和文档；本地生成的数据清单、原始数据、模型权重及训练输出不纳入仓库。克隆后需按上文重新准备数据。运行同时记录 Git 状态及源码 SHA256。开发进展及实际验证范围见 [docs/validation.md](docs/validation.md)。
