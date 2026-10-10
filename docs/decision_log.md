# 开发决策

- 2026-10-04：以本项目交接文档为契约，在独立目录实现；旧ISIC2025仅只读。新目录原仅含交接文档，没有Git和AGENTS。
- 五类和七项属性的字典重新核对作者代码，未知诊断立即报错。索引按原始CSV行位置关联。
- 首版无metadata和属性辅助，使用CE及简单融合；研究机制未定，不预设有效性。
- 基础协议只支持official_holdout。未实现选项在配置校验阶段拒绝，不用空脚本假装支持。
- 预训练使用具体weights enum；预训练未下载的smoke明确weights=None。tiny仅合成测试，冻结配置禁止tiny。
- 源码未直接复制，参考来源、当前Git与SHA256存入source_audit。源码指纹与交接文档核对一致，但旧工作区有未提交修改，不认定其为干净发布版本。
- 根目录和训练epoch保留待定；数据候选只读审计不等于确认正式数据来源。
- 保存Regular的完整训练状态，EMA另存推理权重；阶段恢复先验身份后写文件。复用不完整/不同配置运行被拒绝。
- 临时WSL工具缺少codex-linux-sandbox，标准授权exec入口可运行。此次未修改Codex设置、WSL配置或旧项目环境。
- 2026-10-10：服务器一键入口默认development，五个基线×三种子、25轮、batch1/梯度累积16；这些参数提供可直接运行的开发预算，正式预算仍需独立冻结。每个作业生成独立清单和suite，保留官方分区及原始数据。
- 显式 `--resume` 先校验计划和身份，完成项校验产物后复用，中断项通过Regular epoch checkpoint恢复。无checkpoint失败不删除重跑；保留原目录并另开作业。未知替代证据不会因一键运行自动成为formal_ready。
- 2026-10-10：用户指定首轮 training seed42开发五折，并明确在服务器自行启动；先B01/B02/B04共15次。split_seed保持20261004，五折只覆盖官方train＋valid，原索引保留、test封存。配置、目录、checkpoint和预测身份都包含fold；单种子指标从五折pooled OOF计算，折间标准差不充当跨种子标准差。本机仅工程测试和只读数据核验。
- 2026-10-10：按用户要求参考GitHub ISIC2025基础配置，核验远程commit d902742和本机只读版本1d11163。默认增强改为Albumentations2.0.8共享几何/外观策略，补充加载选项、tqdm与准确率日志；增强强度和概率全部配置化。训练仍为seed42开发五折、25轮、图像CE和末轮EMA，不采用GitHub旧版seed29/50轮/早停/关闭裁剪，不迁移旧任务标签、metadata或MONET。两幅图先分别缩放到共同尺寸，增强seed每次从已checkpoint的Python RNG派生；来源哈希、版本差异及适配边界见isic2025_reference.md/json。
