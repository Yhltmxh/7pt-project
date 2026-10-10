# Linux 服务器一键启动开发基线

## 本轮：seed 42 开发五折

在已配置好 CUDA/PyTorch 的服务器项目根目录执行，替换数据路径：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42 --job-dir outputs/server/cv42
```

此模式默认 B01、B02、B04，每个模型训练五折，共15次。训练 seed 为42，固定划分 seed 为20261004；原始官方train＋valid的616例构成开发池，每折训练492或493例、验证123或124例。官方test的395例不分折、不训练、不推理。原始官方索引和清单的official_split字段保留。

默认每次25轮、末轮EMA、CE、图像输入、batch1/梯度累积16，无元数据、属性辅助或TTA。25轮是当前开发预算，不是已经证明适合该数据的正式预算。类别分层按病例/已核验病灶整体分折；患者身份尚未核验，不声明患者互斥。开发池BCC、SK各26例，每个验证折只有5或6例，需同时查看逐类表现。

入口自动准备 `<job-dir>/data/folds.csv` 和 `folds_meta.json`，保存原始行号、固定seed、逐折数量与SHA256；正式开始训练前重算划分并检查所有开发病例恰好一次、无test、组内一致和跨折精确重复。每折的训练病例、类别数量、folds/split哈希进入provenance；fold同时进入配置、目录、checkpoint和预测身份。所有模型、所有后续训练种子共用同一固定划分。

先查看计划或预检：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42 --dry-run
bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42 --preflight-only
```

开始前确认完整15次运行的存储预算，按每次6 GiB保守预留约90 GiB。若显式加入B05/B06，使用 `--only B01 B02 B04 B05 B06`，计划扩展为25次、保守预算150 GiB。入口只使用已有GPU环境，预检的小运算探针不能代替完整ConvNeXt的显存实测。

SSH后台一行启动，先执行 `mkdir -p outputs/server`：

```bash
nohup bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42 --python "$(command -v python)" --job-dir outputs/server/cv42 > outputs/server/cv42.launch.log 2>&1 < /dev/null &
```

查看日志和恢复：

```bash
tail -f outputs/server/cv42.launch.log
bash scripts/train_server.sh --resume outputs/server/cv42
```

恢复命令只加载原作业参数。不要修改seed、划分、fold文件、源码或旧配置后继续原作业；有冲突会拒绝。恢复到最后完整epoch，已完成的折经过产物校验后复用，缺checkpoint的失败折保留原目录并另开作业。仅计划或预检作业可用resume首次继续启动。

每个模型、每个种子必须完成全部五折才能生成pooled OOF：按病例ID拼接其heldout预测，验证616例恰好各一次，再计算五类Macro F1等指标。逐折结果为 `runs.csv`，pooled单种子结果为 `seed_metrics.csv`，跨种子汇总为 `aggregate.csv`，病例级OOF保存到 `oof/`；真实位置由job.json、summary.json记录。单种子跨种子标准差为null，折间标准差单独报告。结果标记 `dev_cv_oof`、`formal=false`，用于开发筛选，不混入官方valid或最终test结果。

当前已实现并进行工程验证；真实训练由用户在服务器启动。本机合成smoke只验证流程，不产生医学实验结论。

本次基础配置参考ISIC2025后新增Albumentations/tqdm依赖，并更新默认训练增强。服务器升级及直接Python启动示例：

```bash
git pull --ff-only
conda activate dlenv
python -m pip install "albumentations==2.0.8" "tqdm==4.67.1"
python -m pip check
python scripts/train_server.py --data-root /data/derm7pt --protocol dev_cv5 --seeds 42 --job-dir outputs/server/cv42_isic_aug
```

数据路径需替换为实际目录。新配置使用新作业目录；已开始的旧作业保持原源码、环境及配置，不在运行中升级或用新源码resume旧checkpoint。增强、DataLoader及参数对照见 [基础配置参考](isic2025_reference.md)。终端显示批次进度条，nohup重定向日志时自动关闭动态进度条，保留每轮学习率、loss、训练/验证Accuracy、验证Macro F1及AMP跳步统计。

## 官方留出入口

在项目根目录执行：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt
```

`/data/derm7pt` 是示例路径，替换为服务器上的数据目录；该目录必须包含 `meta/meta.csv`、三个官方分区索引和 `images/`。入口会准备独立数据清单、展开计划并预检，再按顺序训练 B01、B02、B04、B05、B06，每个基线使用 101、3407、777 三个种子，共 15 次运行。

默认预算为 **25 epochs 的开发训练**，不是已冻结的论文实验。输入仅为图像，使用 CE、末轮 EMA 和官方 train/valid；不会启动 test 评价。五类顺序为 BCC、NEV、MEL、MISC、SK。临床替代策略采用 `retain_official`，保留发布包内容与 unknown 状态；该选择不证明病例都具有真实完整双模态。唯一的路径大小写差异会显式规范化，原始引用、修复事件和哈希保留，原始 CSV、图像及官方分区不改。

## Python 和 GPU 环境

服务器需已有可用的 Linux Python 3.10 或更高版本、匹配的 PyTorch/torchvision 和实际 CUDA 运行环境。已有Conda时，在项目根目录执行：

```bash
bash scripts/setup_conda.sh
conda activate dlenv
```

脚本默认创建名为dlenv的独立Python3.13.9环境，安装PyTorch2.9.0/torchvision0.24.0的CUDA13.0 wheel，再安装requirements.txt中的其他依赖；使用constraints避免第二次依赖解析切换torch版本或CPU/GPU构建。该组GPU版本来自本机Windows dlenv的只读实时核验（2026-10-10）；Linux wheel配对与源依据[PyTorch官方安装表](https://pytorch.org/get-started/previous-versions/#v290)。Conda负责环境与Python，PyTorch在该环境中用pip安装。

创建Python/pip时使用 `--override-channels --channel https://conda.anaconda.org/conda-forge`，仅搜索conda-forge，不回退到默认的pkgs/main、pkgs/r，也不修改全局`.condarc`或自动接受条款。仅添加 `-c conda-forge` 仍会搜索默认源，参数含义见[Conda官方说明](https://docs.conda.io/projects/conda/en/latest/commands/create.html#channel-customization)。

旧脚本在创建环境前报 `CondaToSNonInteractiveError` 时，在项目目录更新并重试：

```bash
git pull --ff-only
bash scripts/setup_conda.sh
conda activate dlenv
```

如果dlenv已经创建，按下方说明显式使用 `--reuse`。

其他构建、查看计划和失败后重试：

```bash
bash scripts/setup_conda.sh --name dlenv-cu126 --variant cu126
bash scripts/setup_conda.sh --name dlenv-cpu --variant cpu
bash scripts/setup_conda.sh --dry-run
bash scripts/setup_conda.sh --name dlenv --variant cu130 --reuse
```

支持cu126/cu128/cu130/cpu；GPU构建需匹配服务器实际驱动，脚本不安装驱动。CPU选项使用已验证的Python3.12、torch2.8.0+cpu/torchvision0.23.0+cpu，供流程检查。已有环境默认拒绝修改，`--reuse`明确安装到已有兼容Python环境（GPU为3.13，CPU为3.12）；不要在训练运行期间更改该环境依赖。脚本通过 `conda run` 使用指定环境，不要求安装前activate；实际版本、依赖清单和脚本/requirements SHA256记录到 `outputs/conda_setup.*/`。安装完成的小Conv2d反向与torchvision原生NMS检查不代替完整模型显存验证。

也可使用venv建立独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 版 PyTorch 的安装来源与构建需和服务器实际驱动匹配；启动脚本不会自动安装或切换 GPU PyTorch。执行预检前确认所选解释器加载的是预期环境：

```bash
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); print("CUDA available:", torch.cuda.is_available())'
```

默认调用当前环境的 `python3`，默认设备为 `cuda`。可用 `--python /path/to/python` 或 `DERM7PT_PYTHON` 指定解释器；`--data-root` 或 `DERM7PT_ROOT` 指定数据根目录。显式命令参数优先于环境变量。例如：

```bash
DERM7PT_PYTHON="$PWD/.venv/bin/python" DERM7PT_ROOT=/data/derm7pt bash scripts/train_server.sh --preflight-only
```

若报 `No module named 'yaml'`，当前训练解释器没有PyYAML；requirements.txt已声明该依赖，通常需要完成安装或选中正确环境。启动日志会打印实际Python路径。先在项目目录执行，并将数据路径替换为服务器实际路径：

```bash
conda activate dlenv
python -m pip install "PyYAML>=6,<7"
python -c 'import sys, yaml; print(sys.executable); print("PyYAML:", yaml.__version__)'
bash scripts/train_server.sh --python "$(command -v python)" --data-root /data/derm7pt --protocol dev_cv5 --seeds 42
```

使用 `python -m pip` 可保证安装到训练所选解释器。若安装脚本此前未成功完成，执行 `bash scripts/setup_conda.sh --reuse` 补全项目依赖并完成设备验证；单独补PyYAML只解决当前缺包错误。遇到其他缺包或CUDA错误时按实际报错修复，不宣称安装已完成。

## 先看计划，再做预检

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --dry-run
bash scripts/train_server.sh --data-root /data/derm7pt --preflight-only
```

`--dry-run` 展开参数和运行清单；`--preflight-only` 准备并核对数据、环境、设备、缓存和存储，均不开始训练。ConvNeXt 预训练权重在真正训练时会按需下载；预检只报告缓存情况，不自动下载。离线运行添加 `--offline`，所需缓存缺失时拒绝启动。应在服务器当前 Python 使用的缓存目录提前准备相同预训练权重。

计划按每次运行预留 **6 GiB** 估算存储，默认 15 次约 **90 GiB**。这是保守预算，不是实测文件占用；还需为原始图像、预训练缓存、日志和其他进程留空间。双分支 ConvNeXt-Base 的显存需求须在实际服务器验证，批量大小与梯度累积可显式调整。

## SSH 后台运行与查看日志

激活所需环境后，先创建日志父目录：

```bash
mkdir -p outputs/server
```

以下是一行后台启动命令；断开 SSH 后仍保留进程和日志：

```bash
nohup bash scripts/train_server.sh --data-root /data/derm7pt --python "$(command -v python)" --job-dir outputs/server/baseline_dev25 > outputs/server/baseline_dev25.launch.log 2>&1 < /dev/null &
```

查看启动、预检与训练进度：

```bash
tail -f outputs/server/baseline_dev25.launch.log
```

`--job-dir` 为每次作业指定独立目录，保存该作业的数据清单、suite、日志和 runs。已有作业不会被覆盖，也不会修改项目原有的 `data/manifests/`。需要更换参数时使用新的作业目录。

作业内的持续日志为 `<job-dir>/train.log`，可直接查看：

```bash
tail -f outputs/server/baseline_dev25/train.log
```

作业目录保存 `job.json`、`suite.yaml`、`data/`、`runs/`，并按执行阶段生成 `preflight.json`、`status.json` 和 `summary.json`。套件计划与汇总保存在该作业独有的 `outputs/suites/<suite_id>/`；其准确位置记录于 `job.json` 和完成后的 `summary.json`。

## 单基线与调整开发预算

先只跑双模态 mean 的一个种子：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --only B04 --seeds 101 --job-dir outputs/server/b04_seed101
```

显式选择两个基线、两个种子和预算：

```bash
bash scripts/train_server.sh --data-root /data/derm7pt --only B04 B05 --seeds 101 3407 --epochs 25 --warmup-epochs 5 --batch-size 1 --gradient-accumulation 16 --num-workers 4 --device cuda --job-dir outputs/server/mean_concat_dev25
```

此例每次迭代处理 1 个样本，累计 16 次梯度后更新权重；完整累计区间的有效 batch 为 16，最后不足区间按实际样本处理。`--device cpu` 可明确使用 CPU，`--device auto` 由实际可用设备决定；调整设备和预算属于新的运行身份。

| 参数 | 用途 |
| --- | --- |
| `--only B04 B05` | 限定基线；支持 B01、B02、B04、B05、B06 |
| `--protocol dev_cv5` | train＋valid开发五折，默认B01/B02/B04及seed42 |
| `--split-seed 20261004` | 固定划分种子，独立于训练种子；更改时另开作业 |
| `--seeds 101 3407` | 限定种子列表 |
| `--epochs 25`、`--warmup-epochs 5` | 设定开发轮数和 warmup |
| `--batch-size 1`、`--gradient-accumulation 16` | 控制单批显存与梯度累计 |
| `--num-workers 4` | 图像加载 worker 数 |
| `--device cuda` | 选择 cuda、cpu 或 auto |
| `--job-dir outputs/server/name` | 新作业的独立产物目录 |
| `--offline` | 禁止下载，缺少预训练缓存时拒绝启动 |

## 中断后恢复

```bash
bash scripts/train_server.sh --resume outputs/server/baseline_dev25
```

恢复沿用作业保存的配置、清单与身份，仅支持 **epoch 边界**。已经开始训练的中断运行必须已有 `checkpoints/final_regular.pt`；该文件包含模型、EMA、优化器、调度器、scaler 和随机状态。EMA 权重供推理使用，不能代替恢复 checkpoint。

源码、配置、数据、类别顺序、种子或身份哈希有冲突时，恢复会在改写已有作业前拒绝。不要在恢复命令中加入新的训练参数，也不要通过修改旧配置绕过身份检查。尚未产生 checkpoint 的失败运行不能无痕重跑，应保留原作业并使用新的 `--job-dir`。仅执行计划或预检、尚未开始训练的作业可用相同 `--resume` 命令继续首次启动；这类作业没有需要恢复的训练状态。

作业数据准备阶段若留下不完整的 `data/`，入口同样保留它并拒绝覆盖，应使用新作业目录。已完成项会校验后复用；当前整作业恢复仍会执行原设备和磁盘预检，需要原训练环境可用。

目前尚未完成真实数据完整训练。启动入口的静态检查、契约测试、合成 smoke 和预检只能验证各自覆盖的工程行为；真实 GPU 训练是否可完成、实际显存／耗时及模型性能仍需服务器实际运行确认。正式评价继续遵循 [开发协议](protocol.md)，默认一键入口产物均属于开发阶段。
