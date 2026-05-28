# 京东商品评论情感三分类

机器学习课程大作业 —— 基于京东真实评论数据的情感三分类（好评/中评/差评），三层 7 个模型对比。

## 项目结构

```
├── data/
│   ├── raw/                         # 原始数据（OpenI 训练集.csv）
│   ├── processed/                   # 抽样后的 train/val/test.csv
│   └── embeddings/                  # 预训练 Word2Vec 词向量
│
├── src/                             # 共享模块
│   ├── config.py                    # 全局配置（路径、超参数、种子）
│   ├── preprocess.py                # 文本清洗 + jieba 分词
│   ├── dataset.py                   # PyTorch Dataset、词表构建、Embedding 矩阵
│   ├── models/
│   │   ├── textcnn.py               # TextCNN（多尺寸卷积核 3/4/5）
│   │   └── bigru_attn.py            # BiGRU + Attention
│   ├── train_utils.py               # 训练循环、早停、checkpoint
│   ├── metrics.py                   # 评估函数、结果写入
│   └── plot.py                      # 混淆矩阵、训练曲线、模型对比图
│
├── scripts/                         # 训练入口（各层独立）
│   ├── 01_sampling.py               # 数据抽样
│   ├── 02_train_traditional.py      # 传统 ML：NB / SVM / XGBoost
│   ├── 03_train_dl.py               # DL 从头训练：TextCNN / BiGRU-Attn
│   ├── 04_train_pretrained.py       # 预训练微调：BERT / RoBERTa
│   └── 05_evaluate_all.py           # 汇总对比
│
├── app/                             # Web 演示
│   ├── demo.py                      # Gradio 网页界面
│   └── model_loader.py              # 模型加载统一接口
│
├── checkpoints/                     # 训练好的模型权重
├── outputs/
│   ├── figures/                     # 混淆矩阵图、训练曲线、对比图
│   └── results.csv                  # 所有模型结果汇总
│
├── requirements.txt
├── setup.sh
└── README.md
```

## 模型方案

| 层 | 模型 | 代码位置 |
|---|---|---|
| 传统 ML | MultinomialNB、LinearSVC、XGBoost | `scripts/02_train_traditional.py` |
| DL 从头训练 | TextCNN、BiGRU+Attention | `scripts/03_train_dl.py` |
| 预训练微调 | BERT-base-Chinese、RoBERTa-wwm-ext | `scripts/04_train_pretrained.py` |

**共享特征**：传统 ML 共享 TF-IDF；DL 从头训练共享预训练 Word2Vec 词向量。

**类别不平衡处理**：全部使用 class_weight / WeightedRandomSampler 处理差评、中评样本过少的问题。

## 快速开始

### 环境

```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
```

### 1. 下载数据

从 [OpenI 启智社区](https://openi.pcl.ac.cn/thomas-yanxin/Commodity_Review_Sentiment_Forecast) 下载训练集，放到 `data/raw/Commodity_Review_Sentiment_Forecast/训练集.csv`。

### 2. 数据抽样

```bash
python scripts/01_sampling.py
```

从 7 万条原始数据中打三分类标签，分层切分 train/val/test=8:1:1，输出到 `data/processed/`。

### 3. 训练模型

```bash
# 传统机器学习（最快，约 3 分钟）
python scripts/02_train_traditional.py

# 深度学习从头训练（约 1 小时，需先下载预训练词向量）
python scripts/03_train_dl.py --wv-path data/embeddings/sgns.baidubaike.bigram-char

# 预训练模型微调（约 2-3 小时，首次会自动下载模型）
python scripts/04_train_pretrained.py
```

每个脚本支持单独跑某个模型：

```bash
python scripts/03_train_dl.py --model textcnn
python scripts/04_train_pretrained.py --model bert
```

### 4. 汇总对比

```bash
python scripts/05_evaluate_all.py
```

生成 `model_comparison.png` 和 `per_class_f1.png`，打印最优模型。

### 5. Web 演示

```bash
python app/demo.py
```

浏览器打开 `http://localhost:7860`。

## 数据集

- 来源：JD.com E-Commerce Data（WWW 会议）
- 原始大小：7 万条训练集 + 3 万条测试集（无标签）
- 三星映射：1-2星→差评，3星→中评，4-5星→好评
- 分布：差评 7.8% / 中评 11.4% / 好评 80.8%（极度不平衡）
- 切分：train 56k / val 7k / test 7k（分层保持分布一致）

## 预训练词向量

推荐 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 百度百科 300 维：

```
sgns.baidubaike.bigram-char
```

下载后放到 `data/embeddings/`。

## 输出文件

训练完成后 `outputs/figures/` 下每个模型会有：

- `*_confusion_matrix.png` — 混淆矩阵热力图
- `*_training_curves.png` — 训练过程 loss/F1 曲线（仅 DL 和预训练）
- `*_report.txt` — 完整评估指标
- `model_comparison.png` / `per_class_f1.png` — 汇总对比（05 脚本生成）
