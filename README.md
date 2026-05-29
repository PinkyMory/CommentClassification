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
| 预训练微调 | BERT-base-Chinese、RoBERTa-wwm-ext | `scripts/04_train_pretrained.py` | 110M 参数，字/全词掩码，冻结 backbone 仅微调分类头最后一层 |

三层的每一层都独立使用类别权重 / WeightedRandomSampler 处理不平衡。

## 三人分工

| 同学 | 负责内容 | 脚本 | 核心 src 文件 |
|---|---|---|---|
| **A** | 数据工程 + 传统 ML | `01_sampling.py` `02_train_traditional.py` | `preprocess.py` `metrics.py` |
| **B** | DL 从头训练 | `03_train_dl.py` | `dataset.py` `models/textcnn.py` `models/bigru_attn.py` `train_utils.py` |
| **C** | 预训练微调 | `04_train_pretrained.py` | `plot.py`（画对比图） |

`src/config.py` 和 `05_evaluate_all.py` 三人共同维护。

## 项目结构

```
├── src/                         # 共享模块（config / preprocess / metrics / plot / dataset / models）
├── scripts/                     # 独立训练入口
│   ├── 01_sampling.py           # 数据抽样 → train/val/test.csv
│   ├── 02_train_traditional.py
│   ├── 03_train_dl.py
│   ├── 04_train_pretrained.py
│   └── 05_evaluate_all.py       # 汇总所有模型结果 + 画对比图
├── app/                         # Gradio Web 演示
├── data/
│   ├── raw/                     # 原始数据（自己下载）
│   ├── processed/               # 抽样后产出（脚本生成，不入库）
│   └── embeddings/              # 预训练词向量（自己下载）
├── checkpoints/                 # 模型权重（不入库）
├── outputs/
│   ├── figures/                 # 训练过程图 + 混淆矩阵（不入库）
│   └── results.csv              # 所有模型结果汇总（入库，用于对比）
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

CUDA 版 PyTorch（RTX 4060 8GB 显存够用）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 下载数据集

从 OpenI 启智社区下载训练集：

[https://openi.pcl.ac.cn/thomas-yanxin/Commodity_Review_Sentiment_Forecast](https://openi.pcl.ac.cn/thomas-yanxin/Commodity_Review_Sentiment_Forecast)

登录后，在页面顶部"数据集"标签页下载 `训练集.csv`（约 13MB，7 万条）。

放到 `data/raw/Commodity_Review_Sentiment_Forecast/训练集.csv`。

（测试集 `测试集.csv` 没有评分标签，比赛提交用，本项目不需要。）

### 3. 数据预处理

```bash
python scripts/01_sampling.py
```

脚本自动检测列名（`评论内容` → text, `评分` → star），清洗无效行，打三分类标签（1-2→差评 / 3→中评 / 4-5→好评），分层切分 train/val/test = 56k/7k/7k，输出到 `data/processed/`。

三个 CSV 的类别分布一致（约 7.8% 差评 / 11.4% 中评 / 80.8% 好评）。

### 4. 训练模型

```bash
# 传统 ML（最快，2-3 分钟）
python scripts/02_train_traditional.py

# DL 从头训练（需先下载预训练词向量，约 1 小时）
python scripts/03_train_dl.py --wv-path data/embeddings/sgns.baidubaike.bigram-char

# 预训练微调（首次需下载模型，约 50 分钟/个）
python scripts/04_train_pretrained.py --model bert
python scripts/04_train_pretrained.py --model roberta
```

单独跑某个模型：

```bash
python scripts/03_train_dl.py --model textcnn     # 只训 TextCNN
python scripts/03_train_dl.py --model bigru_attn  # 只训 BiGRU
```

### 5. 汇总对比

```bash
python scripts/05_evaluate_all.py
```

打印最优模型，生成 `outputs/figures/model_comparison.png` 和 `per_class_f1.png`。

### 6. Web 演示

```bash
python app/demo.py
```

浏览器打开 `http://localhost:7860`。

## 当前结果

| 模型 | Accuracy | Macro-F1 | 差评 F1 | 中评 F1 | 好评 F1 |
|---|---|---|---|---|---|
| MultinomialNB | 0.8234 | 0.4383 | 0.3329 | 0.0772 | 0.9050 |
| LinearSVC | 0.8296 | 0.4928 | 0.4289 | 0.1392 | 0.9103 |
| XGBoost | 0.6997 | 0.5260 | 0.4211 | 0.3336 | 0.8233 |
| BERT | 0.8271 | **0.6568** | **0.6187** | **0.4346** | 0.9172 |

（RoBERTa、TextCNN、BiGRU-Attn 待训练）

## 输出说明

每个模型跑完后在 `outputs/figures/` 下自动生成：

| 文件 | 说明 |
|---|---|
| `*_confusion_matrix.png` | 混淆矩阵热力图，反映各类别分类错误的方向 |
| `*_training_curves.png` | 训练过程的 loss 和 macro-F1 变化曲线（仅 DL / 预训练） |
| `*_report.txt` | 所有评估指标的格式化文本报告 |

## 预训练词向量

推荐 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 项目中的**百度百科 300 维**：

[下载 sgns.baidubaike.bigram-char](https://github.com/Embedding/Chinese-Word-Vectors)

放到 `data/embeddings/`，`03_train_dl.py` 自动加载。不下载也可以跑（会随机初始化 embedding，但效果差）。

## 注意事项

- 所有路径通过 `src/config.py` 中的 `PROJECT_ROOT` 自动推导，不要写死绝对路径
- 数据集极度不平衡，模型对比时看 macro-F1，不要看 accuracy
- 预训练模型在 HuggingFace 缓存到 `C:\Users\<用户名>\.cache\huggingface\hub\`，如果下载慢设镜像：`set HF_ENDPOINT=https://hf-mirror.com`
