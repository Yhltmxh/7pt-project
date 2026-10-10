# Derm7pt 数据清单

原始数据根目录由 `dataset.root` 或 `--root` 显式传入，必须包含 `meta/` 和 `images/`。当前配置不自动选用旧项目中的数据。原图不复制、不修改，也不为缺失模态复制另一幅图。

```bash
python scripts/prepare_derm7pt.py --root /path/to/derm7pt --output-dir data/manifests
python scripts/audit_dataset.py --manifest data/manifests/samples.csv --root /path/to/derm7pt
```

如原始 CSV 的大小写拼写与磁盘不同，可在准备时显式追加 `--normalize-path-case`。仅在每段路径均有严格唯一的大小写无关匹配时规范化读取路径；歧义立即报错。`source_clinical_path`、`source_dermoscopic_path` 保留 CSV 原始字符串，审计记录 raw、resolved 和原图 SHA256，重审计再次核对原始引用和唯一匹配。该修复只改变清单读取路径，不改变原图、CSV、官方分区、标签或病例集合。

准备程序保存 `samples.csv`、`label_map.json`、`metadata_vocab.json` 和 `dataset_audit.json`。官方分区索引必须为原始 `meta.csv` 的零基位置；索引互斥、完整覆盖、分区数量 413/203/395 均需通过。五类顺序固定为 BCC、NEV、MEL、MISC、SK。七项属性分别进行多分类，缺失用 `-1` 和 false mask。未知诊断或未知非空属性立即报错。[标签来源](https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py)。

`metadata_vocab.json` 仅从官方 train 拟合 sex、location 词表，提供 `<unknown>` 与 `<missing>` 编码，记录 `input_enabled=false`、`fit_partition=train`。elevation 保留在审计中，首版不启用。验证和测试分区不参与词表拟合，当前模型输入仍只有图像。

图像路径相对数据根目录，统一含 `images/` 前缀。审计包括图像解码、SHA256、跨分区同路径与同哈希检查；近重复检查尚未实现，报告明确记录这一限制。跨分区精确重复使审计失败，必须逐项核实后制定统一处理协议。

`has_clinical`、`has_dermoscopic` 表示发布图像文件是否存在；不能证明原始真实模态完整性。`is_clinical_substituted` 默认 unknown，不能仅由两幅图同哈希推断。`patient_id` 保持空值，`case_num` 只用于官方病例／病灶标识；不宣称患者互斥。真实配对敏感性子集应另行审核并在训练前冻结。

准备和审计不要求 PyTorch；解码要求 Pillow。`--skip-decode` 可用于未配置图像依赖时的部分检查，但返回状态码 2，`passed` 为 false。`--allow-nonofficial-counts` 只用于测试夹具或明确修改后的数据协议。

Dataset 返回 `(inputs, targets, ids)`；模型输入仅含所选图像张量和 `modality_mask`。元数据、七点真值和标识不会进入模型输入。训练双图默认共享水平／垂直翻转，评价仅 resize 和 ImageNet 标准化；配置通过 `geometry_coupling` 显式选择 shared 或 independent。这不代表两幅图已空间配准。

本次只读核验的数据候选来自 `/mnt/d/DeepLearning/code/ISIC2025/data/derm7pt`。该数据包 `README.txt` 标注可用于学术研究、图像不可再分发、Dr. Giuseppe Argenziano 保留图像权利，并列出 CC BY-NC-ND 4.0 许可。此处记录包内说明，未对其作额外许可解释。[作者数据入口](http://derm.cs.sfu.ca/)、[作者代码](https://github.com/jeremykawahara/derm7pt)。

2026-10-04 的候选数据审计覆盖 1011 例、2022 个图像引用和 2013 个唯一文件。没有缺失或解码失败，也没有跨分区同路径／同哈希重复。病例 `derm7pt_case_565`、`derm7pt_case_800`、`derm7pt_case_824`、`derm7pt_case_827`、`derm7pt_case_833`、`derm7pt_case_834`、`derm7pt_case_839`、`derm7pt_case_844`、`derm7pt_case_851` 的两种模态指向同一文件；不能直接认定它们就是论文描述的临床替代病例。

唯一读取路径修复位于 `source_row_index=815`、`derm7pt_case_816`、官方 train 的皮肤镜图：原始引用 `FCl/Fcl068.jpg`，读取路径由 `images/FCl/Fcl068.jpg` 规范化为磁盘实际 `images/FCL/Fcl068.jpg`。原始引用、文件 SHA256 和修复事件均保留。规范化前的失败审计保存在 `data/manifests/dataset_audit_before_path_normalization.json`。

规范化后的技术检查 `passed=true`，但 1011 例替代状态均保留 unknown，因此 `status=needs_review`、`formal_ready=false`。正式数据根目录和临床替代处理协议仍待确定，配置未自动填入候选根目录。词表已仅从官方 train 拟合，模型输入仍未启用元数据。
