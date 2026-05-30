# 京东商品评论情感三分类

机器学习课程大作业 —— 基于京东真实电商评论数据的情感三分类（好评 / 中评 / 差评），三层 7 个模型对比。

## 选题背景

电商评论带有用户打出的 1-5 星评分，但星级和真实情感之间存在模糊地带——3 星评论"还行吧，凑合用"到底是中评还是差评？这种模糊性是文本情感分类的核心挑战。本课题将 1-5 星映射为三分类（1-2→差评，3→中评，4-5→好评），用三层递进模型对比各种方法在中文评论上的表现，特别关注少数类（差评、中评）的识别能力。

**主要评价指标为 macro-F1**（而非 accuracy），因为类别极度不平衡（好评占 80%）。

## 模型方案

| 层 | 模型 | 脚本 | 特点 |
|---|---|---|---|
| 传统 ML | MultinomialNB、LinearSVC、XGBoost | `scripts/02_train_traditional.py` | 共享 TF-IDF 特征（5000 维 unigram+bigram） |
| DL 从头训练 | TextCNN、BiGRU+Attention | `scripts/03_train_dl.py` | 共享预训练 Word2Vec 词向量（百度百科 300 维） |
| 预训练微调 | BERT-base-Chinese、RoBERTa-wwm-ext | `scripts/04_train_pretrained.py` | 110M 参数，加载预训练权重后在全量数据上微调 |

三层每一层都独立使用类别权重 / WeightedRandomSampler 处理类别不平衡。

## 项目结构

```
├── src/                         # 共享模块（config / preprocess / metrics / plot / dataset / models）
├── scripts/                     # 独立训练入口
│   ├── 01_sampling.py           # 数据抽样 → train/val/test.csv
│   ├── 02_train_traditional.py  # 传统 ML
│   ├── 03_train_dl.py           # DL 从头训练
│   ├── 04_train_pretrained.py   # 预训练微调
│   └── 05_evaluate_all.py       # 汇总所有模型结果 + 画对比图
├── app/                         # Gradio Web 演示
├── data/
│   ├── raw/                     # 原始数据（需自己下载解压）
│   ├── processed/               # 抽样后产出（脚本生成）
│   └── embeddings/              # 预训练词向量（需自己下载，仅 03 需要）
├── checkpoints/                 # 训练好的模型权重
├── outputs/
│   ├── figures/                 # 训练过程图 + 混淆矩阵 + 评估报告
│   └── results.csv              # 所有模型结果汇总（同名模型的结果会被覆盖）
├── requirements.txt
├── setup.sh
└── README.md
```

## 快速开始

### 1. 环境

```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
```

注意：`requirements.txt` 中的 torch 是 CPU 版，需要按你的 CUDA 版本重新安装以启用 GPU 加速：

```bash
# CUDA 12.6（如 RTX 4060 笔记本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 不确定 CUDA 版本时先查看：nvidia-smi
```

### 2. 下载数据集

从 OpenI 启智社区下载训练集压缩包：

[https://openi.pcl.ac.cn/thomas-yanxin/Commodity_Review_Sentiment_Forecast/datasets](https://openi.pcl.ac.cn/thomas-yanxin/Commodity_Review_Sentiment_Forecast/datasets)

登录后，在"数据集"页面下载数据集 zip 文件，解压后将以下文件直接放到 `data/raw/` 目录下（如果解压后有子文件夹，提取其中的 csv 文件）：

- `训练集.csv`
- `测试集.csv`
- `商品信息.csv`
- `商品类别列表.csv`

如果文件路径不同，运行 `01_sampling.py` 时通过 `--input` 指定，如：

```bash
python scripts/01_sampling.py --input data/raw/Commodity_Review_Sentiment_Forecast/训练集.csv
```

### 3. 数据预处理

```bash
python scripts/01_sampling.py
```

脚本自动检测列名（`评论内容` → text, `评分` → star），清洗无效行，打三分类标签，分层切分 train/val/test = 8:1:1，输出到 `data/processed/`。

**数据集概况**：7 万条有标签评论，三分类分布约 差评 7.8% / 中评 11.4% / 好评 80.8%（极度不平衡）。切分后三个 CSV 的分布一致。

### 4. 训练模型

```bash
# 传统 ML（最快，2-3 分钟）
python scripts/02_train_traditional.py

# DL 从头训练（需先下载预训练词向量至 data/embeddings/，约 1 小时）
python scripts/03_train_dl.py --wv-path data/embeddings/sgns.baidubaike.bigram-char

# 预训练微调（首次运行自动下载模型，约 50 分钟/个）
python scripts/04_train_pretrained.py
```

单独跑某个模型：

```bash
python scripts/03_train_dl.py --model textcnn
python scripts/04_train_pretrained.py --model bert
```

### 5. 汇总对比

```bash
python scripts/05_evaluate_all.py
```

打印最优模型，生成 `outputs/figures/model_comparison.png` 和 `per_class_f1.png`。

### 6. Web 演示

> 需要已经训练好至少一个模型（`checkpoints/` 下有模型权重），并且 `outputs/results.csv` 中有记录。

```bash
python app/demo.py
```

浏览器打开 `http://localhost:7860`。

## 预训练词向量（仅 03 脚本需要）

`03_train_dl.py` 中的 TextCNN 和 BiGRU+Attention 使用预训练 Word2Vec 词向量初始化 Embedding 层。如果不提供词向量，脚本会自动用**随机初始化**（能跑通但效果较差）。

推荐使用 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 项目中的**百度百科 300 维**词向量：

- 下载地址：[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- 文件名：`sgns.baidubaike.bigram-char`
- 放到 `data/embeddings/`

> 传统 ML（02）不需要词向量，它用的是 TF-IDF。预训练微调（04）也不需要，BERT/RoBERTa 自带词表。

## 输出说明

每个模型跑完后在 `outputs/figures/` 下自动生成：

| 文件 | 说明 |
|---|---|
| `*_confusion_matrix.png` | 混淆矩阵热力图，反映各类别分类错误的方向 |
| `*_training_curves.png` | 训练过程的 loss 和 macro-F1 变化曲线（仅 DL / 预训练） |
| `*_report.txt` | 所有评估指标的格式化文本报告 |
