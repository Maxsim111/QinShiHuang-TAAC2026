# QinShiHuang-TAAC2026

2026 腾讯广告算法大赛 Team 秦始皇项目仓库。

## 项目结构

```text
.
├── DeepInterestNetwork/
├── DeepFM/
└── environment.taac2026-torch.yml
```

## DeepFM

`DeepFM/` 已经从原始 TensorFlow 示例重构为 TAAC2026 的 PyTorch DeepFM 训练工程，默认读取本地样例数据并将训练产物写入仓库外的 `local_workspace/`。

当前默认能力：

- 仅使用 PyTorch，不再依赖 TensorFlow
- 默认训练目标为 `action_type = 1` 的单目标二分类
- 原始数据读取路径：
  `local_workspace/datasets/TAAC2026/data_sample_1000/raw/sample_data.parquet`
- 训练数据切分方式：
  按 `timestamp` 做 `70/30` 的 `train/val` 时间切分
- 训练设备约束：
  只允许使用 `RTX 5090 D`

## 说明

- `DeepInterestNetwork/` 保留原仓库已有内容。
- `DeepFM/` 作为当前推荐基线开发目录。
- `environment.taac2026-torch.yml` 提供比赛环境依赖定义。

## 快速开始

创建环境：

```bash
conda env create -f environment.taac2026-torch.yml
conda activate taac2026-torch
```

仅生成特征：

```bash
python DeepFM/train.py --prepare-only
```

开始训练：

```bash
python DeepFM/train.py
```
