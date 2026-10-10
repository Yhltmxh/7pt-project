# 首版开发协议

诊断类别顺序：BCC、NEV、MEL、MISC、SK。原 CSV 的零基行号先关联官方 indexes，再生成清单；筛选和重排不得重新编号。按病例/病灶审计，不将 case_num、case_id 宣称为患者身份。

支持 official_holdout 和开发 dev_cv5。官方留出用 train 训练、valid 观察；开发五折只在官方 train＋valid 内重新分折，原始 official_split 和行号始终保留，test 封存。所有对照共用图像预处理、增强、训练预算、CE 和 EMA。图像尺寸224、同步几何翻转与 ImageNet normalization 是草稿起点。两幅图未配准，共享增强不证明真实空间对应。无外观随机增强，因此 appearance_coupling 为 none。

当前首轮五折使用 training seed 42、固定 split_seed 20261004，B01/B02/B04 共15次运行。训练种子与划分种子独立；补训练种子时不重划折。病例/已核验病灶整体分折，尚未核验患者身份，不宣称患者互斥。616例开发池按五类分层，每折训练492或493例、验证123或124例。folds.csv 保留 sample_id、lesion_id、source_row_index、dev_fold，不给test分折；启动时检查完整覆盖、组内一致、跨折精确重复和固定seed重算一致。

Dataset 返回独立 inputs、targets、ids。训练器只调用 model(**inputs)。七项属性保留三类或二类字典；missing=-1且mask=false，与absent类别分开。所有属性监督和元数据在首版实验配置中关闭。

主指标为固定五类 argmax Macro F1。另报 Accuracy、五类平均Recall（Balanced Accuracy）和逐类Precision/Recall/F1/support。缺少正例或负例时该类 AUROC 为 null，宏 OVR AUROC也为null，不改变宏平均分母。默认有效官方分区五类齐全；极小测试样本的缺类策略在指标文件中明示。

正式模型在训练前将完整配置冻结，末轮 EMA、无 TTA、argmax。开发到正式阶段的配置变化创建新身份和新运行，不能篡改旧运行。正式数据必须通过技术审计并完成替代状态等证据核对。官方索引、病例标签、路径与哈希在 preflight 重新检查。套件运行数与存储估计先打印，runner不自动评价test。

dev_cv5 只允许 development，不能直接改为 frozen，也不能用CV checkpoint推理test。每折训练统计仅从该折训练病例计算，默认图片CE、无元数据、无属性辅助、末轮EMA、argmax和无TTA。先确定共同训练预算；五折分数用于选机制或预算时属于开发证据，不宣称独立无偏的最终性能。

每个训练种子先核验五折0..4各自完整，按病例ID拼接仅该折heldout预测，每个开发病例恰好一次，再计算 pooled OOF Macro F1。每折指标作为补充；折间标准差与跨种子标准差分开记录，单种子跨种子标准差为null。官方 valid、dev_cv_valid、dev_cv_oof 分别标记，不混合排序。源码、配置、folds文件、split、原始数据和checkpoint身份任何变化均拒绝复用。

断点恢复限 epoch 边界，Regular checkpoint 保存模型、EMA、optimizer、scheduler、AMP scaler及所有使用的随机状态。拒绝不匹配的源码、配置、类别顺序、数据和seed。拒绝发生在对旧运行写入之前。EMA文件用于推理，不用于恢复优化器。

当前尚未实现：refit、配对bootstrap、TTA、模型集成、决策调整、空间attention、源V2复现与候选机制。不要将基础设施测试分数写入研究结果。
