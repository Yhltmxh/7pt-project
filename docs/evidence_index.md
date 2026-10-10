# 证据入口

| 内容 | 证据 | 范围 |
| --- | --- | --- |
| 新课题边界 | derm7pt_ai_handoff.md | 本地交接契约 |
| 五分类/属性标签及官方索引 | src/data/labels.py、data/manifests/label_map.json、作者dataset.py | 字典静态核验 |
| 数据完整性 | data/manifests/dataset_audit.json、samples.csv | 候选本地包；非正式性能 |
| 源项目核验 | docs/source_audit.json | 当前只读源码与Git事实 |
| 模型和工程契约 | tests/、docs/validation.md | 小样本流程验证 |
| CPU/GPU全流程 | outputs/smoke_cpu/report.json、outputs/smoke_gpu/report.json | 合成数据；不能作为模型性能 |
| 实验数量/配置 | outputs/suites/derm7pt_baseline_v1/plan.json | 草稿计划；未启动真实训练 |
| 服务器一键入口与恢复 | docs/server_training.md、outputs/validation/pytest_server.xml、outputs/smoke_server_cpu/report.json | 开发入口；恢复链路合成验证，未训练真实套件 |
| seed42开发五折与OOF | configs/experiments/dev_cv_seed42.yaml、outputs/validation/cv42_validation.json、outputs/smoke_cv42_cpu/report.json | 15项合成训练及恢复；真实候选数据仅分折检查和fold0预检，未开始真实训练 |
| 研究性能或创新有效性 | 待验证 | 尚无真实数据训练结果 |

作者数据实现：https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py

作者分区示例：https://github.com/jeremykawahara/derm7pt/blob/master/minimal_example.py
