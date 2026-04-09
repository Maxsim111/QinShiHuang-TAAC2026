## DeepFM PyTorch Baseline for TAAC2026

该目录已经从原始 TensorFlow/Criteo 示例重构为面向 TAAC2026 样例数据的 PyTorch DeepFM 基线。

### 1. 当前目标

- 只使用 PyTorch
- 只在 `RTX 5090 D` 上训练
- 从 TAAC2026 `parquet` 原始数据中抽取静态特征、序列统计特征和时间特征
- 先做单目标二分类基线，默认目标行为为 `action_type = 1`

### 2. 代码结构

```text
DeepFM/
├── configs/
│   └── default.yaml
├── layer.py
├── model.py
├── train.py
└── utils.py
```

### 3. 数据流

训练会读取：

- `user_id`
- `item_id`
- `user_feature`
- `item_feature`
- `seq_feature`
- `timestamp`
- `label`

处理方式：

- `user_id` / `item_id` 作为主稀疏特征
- `user_feature` / `item_feature` 中的 `int_value` 作为稀疏特征，数组和浮点值转为统计型稠密特征
- `seq_feature` 不直接保留原始变长结构，而是转为长度、统计量、与当前 `item_id` 的重合、时间间隔等压缩特征
- `timestamp` 派生小时、星期、时间桶等特征
- `label` 默认映射为 `action_type == 1` 的二分类标签

### 4. 数据切分

采用时间切分：

- `train = 70%`
- `valid = 30%`

所有编码器和归一化参数仅在训练集上拟合，验证集只复用训练集拟合结果，未见类别统一映射到 `UNK`。

### 5. 运行方式

先创建环境：

```bash
conda env create -f environment.taac2026-torch.yml
conda activate taac2026-torch
```

只做特征生成：

```bash
python DeepFM/train.py --prepare-only
```

完整训练：

```bash
python DeepFM/train.py
```

强制重新生成特征缓存：

```bash
python DeepFM/train.py --force-prepare
```

### 6. 输出目录

所有训练产物都写到仓库外的 `local_workspace/`：

- `local_workspace/outputs/taac2026/deepfm/features/`
- `local_workspace/logs/taac2026/deepfm/`
- `local_workspace/checkpoints/taac2026/deepfm/`

### 7. 设备约束

训练脚本会强制：

- 设置 `CUDA_VISIBLE_DEVICES=0`
- 校验可见 GPU 只有 1 张
- 校验该卡名称包含 `5090`

如果设备不满足约束，训练会直接报错退出，不会自动回退到 `2080 Ti`。
