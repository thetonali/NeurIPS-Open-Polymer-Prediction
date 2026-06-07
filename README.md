# NeurIPS Open Polymer Prediction 2025

这是一个面向 Kaggle `NeurIPS - Open Polymer Prediction 2025` 的清晰 baseline 项目。目标是从聚合物 `SMILES` 预测五个性质：

- `Tg`: glass transition temperature
- `FFV`: fractional free volume
- `Tc`: thermal conductivity
- `Density`: polymer density
- `Rg`: radius of gyration

项目默认使用 `SMILES` 文本特征 + 可选 RDKit 分子描述符 + 每个任务单独建模。代码会自动处理训练集中目标列缺失的情况。

## 目录结构

```text
.
├── config/
│   └── default.json
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── train_supplement/
│   │       ├── dataset1.csv
│   │       ├── dataset2.csv
│   │       ├── dataset3.csv
│   │       └── dataset4.csv
│   ├── processed/
│   └── submissions/
├── models/
├── notebooks/
│   └── kaggle_inference.py
├── scripts/
│   ├── make_submission.py
│   └── train_cv.py
├── src/
│   └── polymer_prediction/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── modeling.py
│       └── utils.py
├── .gitignore
└── requirements.txt
```

## 第一步：下载数据

从 Kaggle 比赛页面下载数据后，按下面方式放置：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
data/raw/train_supplement/dataset1.csv
data/raw/train_supplement/dataset2.csv
data/raw/train_supplement/dataset3.csv
data/raw/train_supplement/dataset4.csv
```

## 第二步：安装依赖

建议使用 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
```

RDKit 安装在不同平台上可能不一样。如果 `pip install rdkit` 失败，可以先跳过；代码会自动退化为纯 SMILES 文本特征。

## 第三步：本地交叉验证

```bash
python scripts/train_cv.py --config config/default.json
```

输出内容包括每个目标的 MAE、近似加权 MAE，以及保存到 `models/` 的模型文件。

## 第四步：生成提交文件

```bash
python scripts/make_submission.py --config config/default.json
```

生成：

```text
data/submissions/submission.csv
```

## Kaggle Notebook 提交方式

Kaggle Code Competition 要求：

- Notebook 运行时间不超过 9 小时
- Internet 关闭
- 输出文件名必须是 `submission.csv`

推荐方式：

1. 在本地用本项目训练并调参。
2. 将最终代码整理到 Kaggle Notebook。
3. 如果要直接用本项目结构，可以把 `notebooks/kaggle_inference.py` 的内容复制到 Kaggle Notebook 中运行。

## Baseline 策略

本 baseline 做了四件事：

1. 对 `SMILES` 做字符级 TF-IDF，捕捉局部化学结构片段。
2. 如果环境有 RDKit，额外提取分子量、环数、TPSA、LogP、HBD/HBA 等描述符。
3. 五个目标分别训练，因为每个目标的缺失模式和尺度不同。
4. 对每个目标使用 KFold 验证，最后用全量训练数据重训并保存模型。

## 后续提升方向

- 加入 Morgan fingerprint、MACCS keys、聚合物专用规则特征。
- 使用 LightGBM、CatBoost、XGBoost 做 stacking。
- 对 `train_supplement` 做更细致的字段映射和去重。
- 使用公开预训练分子模型生成 embedding。
- 针对每个 property 单独调参，因为 `Tg`、`FFV`、`Tc`、`Density`、`Rg` 的噪声和数据量差异很大。
