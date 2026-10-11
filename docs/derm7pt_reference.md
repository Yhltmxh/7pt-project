# 旧 Derm7pt 项目与 seed 42 双模态开发对照

核验来源为 https://github.com/Yhltmxh/Derm7pt，commit `f62b13abeb7a0b1697dc50c8532032fa2025c514`。静态读取模型、CV训练器、增强/训练工具、索引加载器和训练日志；文件 SHA256 在 `derm7pt_reference.json`。没有执行旧项目、加载旧权重或重算旧结果。

旧训练器实际配置为 ConvNeXt-Base 双分支、cross_attn、7项辅助任务、元数据、CB Focal、Mixup alpha=0.2，以及联合阶段的 CutMix/SupCon；分类 MLP dropout=0.5。CV训练器采用分阶段训练、早停、综合分数选择最佳轮，并在验证分区拟合阈值。旧 config.py 的 label_smoothing=0.05 不等于实际用于该 CB Focal 训练，不直接复制为已验证配置。

旧 log_cv.md 记载 argmax 五折 F1 均值0.6749±0.0515；该数字不是本项目 pooled OOF，也没有独立重算。更重要的是，旧 data_utils.py 的 load_indexes 对官方 CSV indexes 减1，和本项目保留官方零基原行号的规则不同。旧类顺序也不同（BCC/MEL/NEV/OTHER/SK）。因此旧日志仅作训练经验参考，不能用作本项目已对齐的目标分数，不迁移旧索引、标签编号、阈值或 checkpoint。

本轮仅 B04、training seed42、固定 split_seed20261004 五折，共5次，仍然图片输入、CE、mean融合、固定25轮、末轮EMA、argmax、无TTA、无属性/元数据。新 profile `b04_regularized` 在原配置上启用：

| 参数 | 原首轮 | 新开发候选 |
| --- | --- | --- |
| 分类头 dropout | 0 | 0.5 |
| 双模态 Mixup alpha | 0 | 0.2 |
| EMA decay | 0.999 | 0.99 |
| batch / accumulation | 实际16 / 1 | 16 / 1 |
| 逐轮 Regular 验证 | 无 | Accuracy、Macro-F1、Balanced Accuracy、硬标签CE |
| 逐轮训练集固定预处理评价 | 无 | Regular的Accuracy、Macro-F1、硬标签CE |

骨干/分类头 LR 仍为5e-5/5e-4，weight decay仍为0.005，增强及25轮训练预算保持不变。采用Mixup与dropout的目的为限制对训练样本的硬拟合；降低EMA decay用于排查此前末轮EMA明显落后Regular的现象。它是组合开发优化，不是能分别归因的单因素消融，也没有承诺0.99或dropout=0.5必定更好。不迁移元数据、属性任务、复杂融合、CutMix、SupCon、早停或阈值调整。

Mixup两幅图使用同一病例排列和同一lambda，只混合输入图像，两个诊断标签分别算CE后加权。modality mask和病例ID保持语义；targets/ids不进入forward。Mixup只在训练开启，不用于验证/推理。随机数来自已保存的NumPy/Torch RNG，epoch恢复必须重放相同混合轨迹。

使用Mixup时train_accuracy是两种标签命中率的lambda加权值，不能与旧的硬标签训练Accuracy直接解释为过拟合改善。判断泛化差距用同一Regular权重、同一固定预处理下的 `train_regular_eval_*` 和 `valid_regular_*`；EMA验证对应原 `valid_*` 列，`valid_loss` 是硬标签CE。所有逐轮观察仅用于开发诊断，最终产物仍取固定末轮，不根据曲线峰值偷换主结果。

训练完成后与首轮B04相同病例OOF比较主EMA F1、Regular F1、逐类Recall，特别关注BCC/SK/MISC；不以训练Accuracy下降单独判定成功。只有完成新训练才能判断改善程度。数据formal_ready仍为false，test继续封存；新源码/配置身份禁止续用旧checkpoint。
