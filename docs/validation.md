# 开发验收记录

日期：2026-10-04。以下是本轮实际执行的验证，不包含真实数据训练性能。

| 检查 | 结果 | 证据/边界 |
| --- | --- | --- |
| 完整契约测试 | 66 passed | outputs/validation/pytest.xml |
| WSL CPU全流程 | 通过 | outputs/smoke_cpu/report.json；合成图像及tiny encoder |
| Windows CUDA全流程 | 通过 | outputs/smoke_gpu/report.json；合成图像、RTX5060 Laptop |
| 实际ConvNeXt-Base临床分支 | CPU前向/反向通过 | 输入[1,3,64,64]，feature1024、token[1,4,1024]、诊断[1,5]，全部编码器梯度有限 |
| 实际双ConvNeXt-Base mean | CUDA前向/反向通过 | outputs/validation/convnext_gpu.json；两路[1,3,64,64]，logits[1,5]，全部编码器梯度有限；未加载预训练权重 |
| 候选真实数据batch | 通过 | clinical/dermoscopic[2,3,224,224]，mask[2,2]，diagnosis int64 |
| 草稿suite dry run | 15项计划通过 | B01/B02/B04/B05/B06，各101/3407/777；未训练 |
| 候选数据开发preflight | 通过 | outputs/validation/preflight.json；仅内存设置root/budget，未改配置 |
| 未就绪数据正式preflight | 正确拒绝 | 替代身份未解决，formal_ready=false |
| train/infer/prepare/audit/runner/collector --help | 通过 | 确认实际CLI入口 |

CPU验证环境：临时隔离目录 `/tmp/derm7pt-baseline-venv`，Python3.12.3、torch2.8.0+cpu、torchvision0.23.0+cpu。完整版本记录见 requirements.cpu-tested.lock.txt 和 smoke report；基础项目安装说明见README。

GPU验证调用既有 Windows dlenv 的 Python3.13.9、torch2.9.0+cu130、torchvision0.24.0+cu130；未安装、升级或改动该环境。WSL默认Python起初不含PyTorch，因此CPU依赖只安装在新建的临时venv。两个环境的恢复验证分别在各自环境内执行，不宣称跨环境得到相同权重。

合成全流程实际训练B01、B02、B04，各2轮。核对Regular/EMA、病例级NPZ、固定五类指标、重新加载EMA后的逐项相同预测、epoch1中断再恢复与不中断epoch2的逐张量相同权重、EMA更新次数和history。包含最后一个不满批次及不满梯度累积窗口。错误配置恢复在写入前被拒绝，旧status/config保持原字节；开发test推理被拒绝。

GPU初次检查发现FP16高初始缩放导致梯度溢出，已经修复：初始scale1024，unscale后检查梯度，溢出由GradScaler跳过optimizer，同时不更新EMA；记录跳步次数；整轮没有有效更新时拒绝完成。专门测试注入一次有限loss但无限gradient，确认跳步后能恢复、EMA只更新一次、权重保持有限。正式训练仍需按目标GPU、224图像和batch预算实测资源。

数据候选审计：1011病例，413/203/395；2022图像引用，2013唯一文件，全部解码。无缺失、解码失败或跨分区精确重复。case816唯一目录大小写错误仅在规范化清单中显式修复，源引用和哈希保留，原始CSV/图像未变。9例双模态引用同文件，替代身份仍unknown，正式准备状态待核验；近重复搜索与患者身份未验证。

尚未执行：实际Derm7pt训练、预训练权重下载及缓存校验、224/batch16下双ConvNeXt完整训练的显存验证、完整三种子套件、官方test评价、CV/refit、研究机制和论文性能分析。当前产物只能证明工程流程已跑通。

## 2026-10-10：服务器一键入口

新增 `scripts/train_server.sh` / `train_server.py`，串联独立作业配置、环境运算探针、全套存储预算、数据准备、已有 runner 的预检和训练、产物校验与结果汇总。未创建第二套训练器。默认 development、25 epochs、batch 1、gradient accumulation 16、CUDA；五个基线和三个配对种子构成15项计划。这些是可调整的开发起点，不能作为已冻结的实验设置。

| 检查 | 结果 | 证据/边界 |
| --- | --- | --- |
| 完整测试 | 94 passed | outputs/validation/pytest_server.xml；含原有66项和新增28项 |
| 独立 CPU smoke | 通过 | outputs/smoke_server_cpu/report.json；合成图像、tiny encoder |
| 服务器一键恢复链路 | 通过 | tests/test_server_launcher.py；合成20例、B04/tiny、2轮，一轮后中断再恢复，真实 runner/train/collector |
| 已完成运行复用 | 通过 | 同一合成服务器作业再次启动；校验后不重训，Regular SHA256保持相同 |
| 失败和身份漂移 | 正确拒绝 | tests/test_server_resume.py；无checkpoint、配置/种子/源码/原始数据变化、完整产物篡改；mock trainer，原产物字节保留 |
| dry run 无ML依赖 | 通过 | 子进程禁止导入 torch/torchvision/Pillow/sklearn，数据路径不存在仍可生成计划 |
| 设备运算与缓存 | CPU探针通过 | 小Conv2d前向/反向/optimizer，RNG保持；缓存缺失/损坏/有效哈希及独立磁盘预算使用测试夹具，未下载真实权重 |
| 官方草稿suite计划 | 15项通过 | outputs/validation/baselines_server_dry_run.json；未训练 |
| 服务器完整计划 | 15项通过 | outputs/server/server_full_plan_validation/job.json 所指plan；默认开发参数，未读数据、未训练 |
| 候选真实数据服务器预检 | 通过 | outputs/server/server_cpu_preflight_validation/preflight.json；1011病例及413/203/395官方分区，B04/seed101/CPU，仅检查；formal_ready=false |
| Bash入口 | 通过 | 语法、--help、带空格解释器及原退出码传递 |

本轮CPU环境为隔离的 `/tmp/derm7pt-server-venv`，按照已有 `requirements.cpu-tested.lock.txt` 安装。未改动 ISIC2025 项目或 Windows dlenv 环境。服务器套件恢复验证与单项恢复均限同环境下的 epoch 边界；没有承诺中间 batch 或跨环境精确恢复。

真实候选数据预检通过服务器Bash入口执行，显式指定旧项目中的候选数据根目录，只读解码和哈希，清单写入本作业的data目录。原始文件和原有data/manifests未改动；运行目录没有训练checkpoint或病例预测。ConvNeXt真实缓存当前缺失，预检记录available=false并保留训练时下载步骤，未报告真实缓存校验已完成。

尚未验证服务器CUDA完整基线训练、真实权重下载成功、双ConvNeXt在224输入下的目标batch容量、真实耗时或模型性能。真实权重缓存API已在已安装torchvision中核对，完整SHA256及URL哈希前缀的运行校验由入口执行。当前小运算探针不代表完整模型显存验证。

## 2026-10-10：seed 42 开发五折

本轮接通 `dev_cv5`：官方 train＋valid 构成开发池，训练种子42，划分种子20261004；默认 B01/B02/B04 各五折，共15次。训练、推理仍复用唯一入口，fold参与配置、目录、checkpoint、预测和恢复身份。每个模型全部五折完成后，按病例拼接 heldout 预测并重算 pooled OOF 指标；不以逐折指标均值代替 pooled 指标。

| 检查 | 结果 | 证据/边界 |
| --- | --- | --- |
| 完整测试 | 179 passed | outputs/validation/pytest_cv42.xml、pytest_cv42.log；含原官方留出契约和CV新增检查 |
| 官方留出 CPU smoke | 通过 | outputs/smoke_holdout_after_cv/report.json；B01/B02/B04、合成图像、tiny encoder、2轮 |
| seed42 五折 CPU smoke | 15次通过 | outputs/smoke_cv42_cpu/report.json；合成30例，其中25例开发池、5例封存test；B01/B02/B04、tiny encoder、2轮 |
| 五折服务器入口集成 | 通过 | tests/test_cv_server.py；真实server→runner→trainer→collector，B04五折合成数据；epoch1中断后恢复，OOF及完整折复用 |
| CV恢复与产物身份 | 通过 | 同环境恢复与不中断结果逐tensor/history一致；拒绝跨fold、划分/来源漂移、缺折、重复病例或不完整注册 |
| 末轮EMA中断恢复 | 通过 | tests/test_cv_training.py；Regular末轮已保存但EMA缺失或旧epoch时补齐，预算已满不重训；一致EMA保持原SHA256 |
| 预测及OOF原子写入 | 通过 | 临时ZIP写入后注入KeyboardInterrupt，不发布损坏最终NPZ；完全一致NPZ缺metrics可补，已有异变拒绝覆盖 |
| 两种协议 dry run | 各15项通过 | outputs/validation/baselines_after_cv_dry_run.json、cv42_dry_run.json；未训练 |
| 服务器五折最终计划 | 15项通过 | outputs/server/cv42_plan_verified/job.json及其plan；B01/B02/B04、seed42、25轮；未访问数据或导入ML |
| 真实候选数据固定划分 | 616例通过 | outputs/validation/cv42_data_check/data/folds_meta.json；valid数量124/123/123/123/123，全部开发病例恰好一次，无test；跨折精确图像重复检查通过 |
| 真实候选数据fold0预检 | 通过 | outputs/validation/cv42_data_check/preflight.json；CPU、B04配置，train492/valid124；只读原始数据，未构造模型、下载权重或训练 |
| 最终证据与来源哈希 | 已记录 | outputs/validation/cv42_validation.json；当前源码、配置、清单/折分、合成checkpoint及OOF文件SHA256 |

本轮使用 `/tmp/derm7pt-server-venv` 的CPU依赖环境。候选数据清单复制到本项目的验证目录，原始索引、图像和原有清单保持原样；患者身份及临床替代状态仍待核验，不声明患者级互斥或正式数据就绪。合成smoke验证的是流程与恢复，不是医学性能证据。

真实训练由用户在服务器启动；本轮没有执行真实ConvNeXt五折训练、服务器CUDA训练、真实预训练下载、官方test推理或refit。模型显存、耗时、收敛与性能仍待服务器实测。25轮保持开发预算；单种子不提供跨种子方差，五折标准差只作为开发描述统计单独报告。

## 2026-10-10：Git初始化

使用main分支建立本地仓库，提交源码、配置、测试和文档。原始数据、本地生成的data/manifests、权重及outputs忽略，仅保留清单目录占位文件；已验证暂存范围及暂存字节与工作文件一致。`.gitattributes` 固定源码、脚本和文本配置的LF换行，README同步仓库与数据准备说明。

提交前重新执行完整测试179 passed（outputs/validation/pytest_git_init.xml）、CPU合成smoke通过（outputs/smoke_git_init_cpu/report.json）、官方基线dry run共15项通过（outputs/validation/baselines_git_init_dry_run.json），`git diff --cached --check`通过。没有启动真实数据训练；未改训练或推理实现。

## 2026-10-10：Conda依赖安装脚本

新增 `scripts/setup_conda.sh`，环境名按用户要求默认为dlenv，GPU配置采用只读实时核验的本机版本：Python3.13.9、torch2.9.0+cu130、torchvision0.24.0+cu130。cu126/cu128为可选GPU构建；CPU选项沿用已验证的Python3.12、torch2.8.0+cpu/torchvision0.23.0+cpu。匹配的wheel对来自PyTorch官方源，第二步requirements解析使用constraints固定版本及构建；实际安装成功时记录软件版本与脚本/requirements SHA256。

| 检查 | 结果 | 边界 |
| --- | --- | --- |
| 完整测试 | 191 passed | outputs/validation/pytest_conda_setup.xml；含12项安装入口检查 |
| Conda入口边界 | 通过 | tests/test_conda_setup.py；mock Conda管理和pip安装，验证默认dlenv/cu130、非法配置拒绝、旧环境保留、显式reuse及Python不匹配时停止；CPU探针调用真实已有隔离环境 |
| Bash语法/help/dry-run | 通过 | 未依赖实际Conda，未创建环境或下载 |
| CPU smoke和基线计划 | 通过 | outputs/smoke_conda_setup_cpu/report.json、outputs/validation/baselines_conda_setup_dry_run.json；合成训练及15项草稿计划 |
| 本机GPU版本 | 实时确认 | Windows dlenv Python3.13.9、torch2.9.0+cu130、torchvision0.24.0+cu130、CUDA13.0；RTX5060 Laptop，驱动581.57 |
| 本机GPU小运算探针 | 通过 | outputs/validation/conda_local_gpu_ops.w9_blx5n/environment.json；与安装脚本相同的Conv2d前向/反向、有限梯度及CUDA原生NMS |

本机原Windows dlenv没有pytest，因此原样执行完整安装后验证片段在pytest导入处拒绝；未向该环境安装软件。GPU运算复查只省略该导入，执行代码SHA256及边界在上述报告中单独记录。安装脚本先安装项目requirements（包含pytest），再执行完整导入和运算检查。

本轮没有实际执行Conda环境创建、网络wheel安装或服务器Linux CUDA检查，本机既有Windows环境未改动；Conda/pip路由测试不证明真实下载安装成功。服务器执行脚本后的验证与environment.json才记录实际安装结果。GPU小探针不代表完整ConvNeXt显存、训练稳定性或医学性能验证。

## 2026-10-10：Conda默认源条款报错修复

针对服务器反馈的 `CondaToSNonInteractiveError`，安装脚本创建环境时改为仅使用conda-forge，显式传入 `--override-channels` 和完整channel URL。dry-run与实际创建共用同一命令；不回退到Anaconda默认源，不更改全局源配置，也不执行条款接受命令。Python、PyTorch/CUDA版本及环境名保持原定配置。

完整测试192 passed（`outputs/validation/pytest_conda_tos.xml`）；13项安装入口检查包含模拟默认源未接受条款时的创建行为，实际CPU运算探针继续通过。CPU合成smoke通过（`outputs/smoke_conda_tos_cpu/report.json`）；官方基线dry-run生成15项计划且未启用test推理（`outputs/validation/baselines_conda_tos_dry_run.json`）；Bash语法及Git diff检查通过。

上述验证使用已有隔离CPU环境和mock Conda/pip，不代表已在服务器完成真实Conda下载安装。服务器执行更新后的脚本会生成实际版本及脚本/requirements SHA256报告；本轮没有启动真实训练或修改其他项目环境。

## 2026-10-10：缺少PyYAML时的训练入口诊断

针对服务器反馈的 `No module named 'yaml'`，核验requirements.txt已包含PyYAML；目前不能从单条报错判定服务器是未完成安装还是选错解释器。训练入口增加实际Python路径日志；缺少模块时输出环境选择提示，缺少yaml时给出针对该解释器的PyYAML安装命令。服务器说明补充激活dlenv、使用python -m pip安装及显式指定训练解释器的步骤。未自动安装依赖或更改训练配置。

完整测试193 passed（`outputs/validation/pytest_missing_yaml.xml`），新增缺包诊断及创建作业前拒绝的回归检查。另用实际空venv通过Bash入口复现缺少yaml，确认退出码2、解释器路径和可执行修复命令，且未创建作业（`outputs/validation/missing_yaml_probe.json`，含入口源码SHA256）。CPU合成smoke通过（`outputs/smoke_missing_yaml_cpu/report.json`）；官方基线dry-run生成15项计划，无test推理（`outputs/validation/baselines_missing_yaml_dry_run.json`）。

验证没有访问用户服务器、向其他项目或既有Windows环境安装软件，也没有启动真实训练。服务器实际解释器、安装完成状态及CUDA可用性须通过服务器上的安装验证和启动日志确认。

## 2026-10-10：参考ISIC2025更新基础配置

只读核验用户指定GitHub仓库的默认分支commit d902742cc9c378c747ff2a38da915308c5ea6f83，以及本机旧项目1d11163版本。来源文件SHA256、远程/本机版本差异和实际配置保存于docs/isic2025_reference.json。旧项目所读文件哈希在完成检查后保持一致。

默认增强接入Albumentations2.0.8，参考旧项目的翻转、Affine、RandomResizedCrop、亮度/对比度、HSV和模糊/噪声；双模态共享几何和外观参数，不宣称图像配准。不同输入尺寸先统一画布，是明确记录的Derm7pt适配。训练每次为Albumentations设置由checkpointed Python RNG派生的seed，persistent_workers保持false。新增显式加载配置、终端tqdm和每轮准确率/学习率/AMP更新统计，环境记录新增依赖版本。原AdamW、warmup/cosine、EMA及裁剪值已核对，保留当前五折25轮/CE/末轮EMA协议。

| 检查 | 结果 | 证据及范围 |
| --- | --- | --- |
| 完整测试 | 207 passed | outputs/validation/pytest_isic_reference_final.xml；概率/范围和未知配置拒绝、共享增强、不同尺寸及单模态、验证确定性 |
| epoch恢复 | 0/2 workers通过 | tests/test_cv_training.py；真实tiny训练，Regular/EMA/history逐项一致，测试各自同一worker数 |
| 终端与日志 | 通过 | tests/test_server_launcher.py；动态进度条可见且不写入服务器逐轮日志，后台动态条关闭 |
| 服务器恢复链路 | 通过 | 真实server→runner→tiny trainer→collector；五折中断恢复、OOF和完成折复用 |
| CPU合成smoke | 通过 | outputs/smoke_isic_reference_final_cpu/report.json；B01/B02/B04和精确恢复 |
| seed42五折合成smoke | 15次通过 | outputs/smoke_isic_reference_final_cv42/report.json；B01/B02/B04、30例合成数据、tiny编码器、每次2轮，含中断恢复/OOF/完成项复用 |
| 依赖与计划 | 通过 | pip check；baselines_isic_reference_final_dry_run.json、cv42_isic_reference_final_dry_run.json各15项，未使用test |

CPU验证使用既有隔离venv，并在其中补装Albumentations2.0.8/tqdm4.67.1；完整实际依赖更新到requirements.cpu-tested.lock.txt。2-worker测试通过，但Python3.12报告多线程进程fork的DeprecationWarning；未将其解释为已验证任意平台/worker数。未修改本机旧项目或Windows dlenv，未执行服务器CUDA、真实ConvNeXt训练、真实权重下载或官方test评价。本次增强是开发设置，合成分数不能作为效果证据。

本次最终源码/配置、参考版本、依赖锁及验证产物哈希索引为outputs/validation/isic_reference_validation.json。
