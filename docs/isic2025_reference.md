# ISIC2025基础配置迁移

按用户指定的[ISIC2025仓库](https://github.com/Yhltmxh/ISIC2025)，只读核验GitHub默认分支commit `d902742cc9c378c747ff2a38da915308c5ea6f83` 的config/dataset/train/train_utils，以及本机旧项目commit `1d11163e77d29d4fb915cb521277d0526a1b238c` 的当前文件。GitHub快照下载到临时目录，未修改本机ISIC2025项目。文件SHA256及读取范围保存于isic2025_reference.json。

GitHub版本与本机后续版本不同：GitHub默认seed29、50轮、早停、max_grad_norm=0；本机已有较新的训练协议和requirements（Albumentations2.0.8、tqdm4.67.1）。本次不混用旧实验结果，也不把旧任务设置认定为Derm7pt最佳参数。

## 当前统一参数

| 配置 | 当前Derm7pt开发设置 | 来源/处理 |
| --- | --- | --- |
| 输入 | RGB，224×224，ImageNet标准化 | 参考源dataset/config；Pillow读取后交给Albumentations |
| 水平/垂直翻转 | 各p=0.5 | 同源配置 |
| Affine | translate_percent=0.05、scale=[0.9,1.1]、rotate=30、p=0.6 | 同源参数；OpenCV线性插值、常数0填充 |
| RandomResizedCrop | scale=[0.8,1.0]、ratio=[0.9,1.1]、p=0.6 | 同源参数 |
| 亮度/对比度 | limit=0.2/0.2、p=0.4 | 同源参数 |
| HSV | hue/saturation/value limit=5/10/5、p=0.2 | 同源参数 |
| 模糊/噪声 | OneOf p=0.15；GaussianBlur blur_limit=3；GaussNoise std_range=[0.01,0.05] | 同源参数；组内默认等权 |
| 双模态增强 | 几何和外观均共享随机参数 | 同源additional_targets约定；不证明空间配准 |
| 验证/推理 | 固定resize、normalize、tensor，无TTA | 保留当前协议 |
| 优化器 | AdamW，backbone LR=5e-5、head LR=5e-4、weight_decay=5e-3 | 与源config一致，原已实现 |
| 调度 | 5轮线性warmup＋cosine、最低LR=1e-7 | 与源train公式一致，原已实现 |
| EMA | decay=0.999，仅使用有效optimizer更新 | 与源基础值一致，保留现有AMP溢出保护 |
| 梯度裁剪 | max_grad_norm=1.0 | 保留当前协议及本机后续设置，不采用GitHub旧版0 |
| 训练预算 | seed42、5折、25轮；batch1/accumulation16 | 保留已约定的Derm7pt开发预算；非实测显存/效果 |
| 加载 | workers4、pin_memory=true（CPU实际关闭）、prefetch_factor2、persistent_workers=false | 参考源加载设置；workers0时prefetch为None |
| 日志 | tqdm终端批次进度＋逐轮LR/loss/Accuracy/Macro F1/AMP统计 | 参考源train/train_utils；后台关闭动态条 |

## Derm7pt适配及可复现边界

临床图和皮肤镜图可能尺寸不同，因此先分别缩放到共同输入尺寸，再共享Affine/crop参数；最后按源流程resize/normalize/tensor。这项顺序适配与旧项目原始管线不同，不声称像素级复现其结果。Albumentations/OpenCV resize与旧Pillow resize也不默认逐像素相同。

训练每次调用从Python RNG取一个32位seed并传给Albumentations，复用现有worker初始化和checkpoint RNG恢复。这样避免遗漏Albumentations2.x内部独立随机状态；persistent_workers固定false。验证不使用随机操作。0与2个worker的恢复须各自在同一环境验证，不承诺跨worker数量、跨硬件或跨依赖版本一致。

增强参数纳入严格配置校验和配置哈希；未知字段、概率越界、逆序范围及不兼容共享设置均拒绝。Albumentations版本固定2.0.8，其他运行库版本记录到环境信息。原flips/none模式作为显式对照保留；使用时appearance_coupling必须为none。

五类顺序、官方索引、test封存、图像CE、无metadata/属性辅助、末轮EMA及唯一训练/推理入口保持本课题协议。不引入旧项目11类、MONET、CB-Focal、早停、旧checkpoint或决策调整。所有B01/B02/B04五折共享本次增强与优化设置；新源码/配置拒绝复用旧作业，需新建目录。

以上是工程适配和开发设置，真实Derm7pt收敛、增强收益、显存及耗时尚未实测。只在开发五折上评估设置，不使用test选择增强。
