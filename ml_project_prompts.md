# 机器学习大作业提示词全集

## 最终技术方案

```
任务：京东电商评论情感三分类（好评 / 中评 / 差评）

数据层：
  JD 720万评论数据集（OpenI 启智社区）
  → 分层抽样取 5 万条子集
  → 1-2 星→差评 / 3 星→中评 / 4-5 星→好评

传统机器学习（3 个，共享 TF-IDF 特征）：
  ├─ 朴素贝叶斯 (MultinomialNB)
  ├─ SVM (LinearSVC)
  └─ XGBoost

深度学习-从零训练（2 个，共享 Word2Vec 词向量）：
  ├─ TextCNN（Kim 2014）
  └─ BiGRU + Attention（王伟等 2019）

预训练微调（3 个）：
  ├─ BERT-base-Chinese（Devlin 2019）
  ├─ Chinese-BERT-wwm-ext（Cui 2021）
  └─ Chinese-RoBERTa-wwm-ext（Liu 2019 + Cui 2021）

环境：Google Colab T4 GPU
时间：两周（14 天），每天 3-4 小时
引用论文：共 11 篇
```

---

## 论文引用清单

| # | 论文 | 对应组件 | 出现在哪个章节 |
|---|------|---------|-------------|
| 1 | Kim (2014) TextCNN, EMNLP | TextCNN 模型 | 方法论 3.2 |
| 2 | 王伟等 (2019) BiGRU-Attention 文本情感分类, 计算机应用研究 | BiGRU+Attention 模型 | 方法论 3.2 |
| 3 | Yang et al. (2016) HAN, NAACL | Attention 权重机制 | 方法论 3.2 |
| 4 | Bahdanau et al. (2015) Attention 起源, ICLR | Attention 机制起源 | 方法论 3.2 |
| 5 | Devlin et al. (2019) BERT, NAACL | BERT 微调 | 方法论 3.3 |
| 6 | Liu et al. (2019) RoBERTa | RoBERTa 改进 | 方法论 3.3 |
| 7 | Cui et al. (2021) 中文 BERT-wwm | 全词掩码（BERT-wwm + RoBERTa-wwm） | 方法论 3.3 |
| 8 | Vaswani et al. (2017) Attention Is All You Need, NeurIPS | Transformer 基础 | 方法论 3.3 |
| 9 | Chen & Guestrin (2016) XGBoost, KDD | XGBoost 模型 | 方法论 3.1 |
| 10 | Sun et al. (2019) How to Fine-Tune BERT for Text Classification? | BERT 微调策略 | 方法论 3.3 |
| 11 | Zhao et al. (2023) 中文电商 ABSA | 研究背景支撑 | 引言 |

---

## 提示词 1：项目总体规划确认

```

我是一名计算机本科生，机器学习课程大作业，选题为【京东电商评论情感三分类（好评/中评/差评）】。

我的最终技术方案（三层对比，共 8 个模型）：

第一层-传统机器学习（3 个，共享 TF-IDF 特征）：
  - 朴素贝叶斯 (MultinomialNB)
  - SVM (LinearSVC)
  - XGBoost

第二层-深度学习从零训练（2 个，共享 Word2Vec 词向量）：
  - TextCNN（多尺寸卷积核 3/4/5-gram）
  - BiGRU + Attention

第三层-预训练微调（3 个）：
  - BERT-base-Chinese（字级掩码）
  - Chinese-BERT-wwm-ext（全词掩码）
  - Chinese-RoBERTa-wwm-ext（全词掩码 + 动态掩码 + 更大数据）

数据集：
  来源：OpenI 启智社区 JD 720 万条评论数据集（京东真实数据，含 1-5 星评分）
  我的处理：分层抽样取 5 万条子集，1-2 星 → 差评，3 星 → 中评，4-5 星 → 好评
  train/val/test = 8:1:1（训练 4 万 / 验证 5 千 / 测试 5 千）

工作量：两周，每天 3-4 小时，Google Colab T4 GPU

拟引用论文（11 篇）：
- 传统 ML：Chen & Guestrin (2016) XGBoost, KDD
- CNN：Kim (2014) TextCNN, EMNLP
- RNN+Attn：王伟等 (2019) BiGRU-Attention 文本情感分类, 计算机应用研究
            Yang et al. (2016) HAN, NAACL
            Bahdanau et al. (2015) Attention 起源, ICLR
- Transformer/预训练：Vaswani et al. (2017) Attention Is All You Need, NeurIPS
                    Devlin et al. (2019) BERT, NAACL
                    Liu et al. (2019) RoBERTa
                    Cui et al. (2021) 中文 BERT-wwm
                    Sun et al. (2019) BERT 微调策略
- 应用背景：Zhao et al. (2023) 中文电商 ABSA

请帮我确认：
1. 这个技术方案是否完整合理？有没有明显的坑？
2. 8 个模型对两周来说会不会太多？如果太多，建议怎么精简？
3. 1-5 星映射为三分类（1-2→差评，3→中评，4-5→好评）是否合理？有没有更好的映射策略？
4. 从 720 万条中抽 5 万条子集，分层抽样时需要注意什么问题？
5. 给一个项目代码结构建议
```

---

## 提示词 2：数据集获取与预处理

```
我需要从 OpenI 启智社区的京东评论数据集（720 万条）中构建实验数据，请提供完整的 Python 代码。

数据集信息：
- 来源：OpenI 启智社区 thomas-yanxin/Commodity_Review_Sentiment_Forecast
- 字段：评论内容、评论标题、评分（1-5 星整数）、用户ID、商品ID、时间戳
- 原始数据较大（约 67 MB CSV），需要在 Colab 上直接下载

我的目标：
1. 从 720 万条中分层抽样 5 万条，保证各星级均衡
2. 1-2 星 → 标签 0（差评），3 星 → 标签 1（中评），4-5 星 → 标签 2（好评）
3. 数据清洗：去除空评论、去除纯数字/纯符号评论、去除过短（<5 字）和过长（>200 字）评论
4. jieba 分词 + 去停用词，保留分词结果（用于传统 ML 和 DL 从零训练）
5. 同时保留原始文本（用于 BERT 系列，BERT 用原始文本不需要分词）
6. 划分 train/val/test = 8:1:1，分层划分保证各类别比例一致
7. 保存为 CSV 文件（train.csv, val.csv, test.csv），供后续所有实验复用

要求：
- 代码要能在 Google Colab 上直接运行，开头加上必要的 !pip install
- EDA 内容：类别分布柱状图、文本长度分布直方图（按类别着色）、每个类别的高频词词云
- 打印数据集统计信息：总条数、各类别数量、平均文本长度、词汇量
- 分析类别分布是否均衡，如果不均衡给出处理建议
- 保存最终数据时，CSV 包含三列：original_text（原始文本）, tokens（分词后以空格分隔）, label（0/1/2）
```

---

## 提示词 3：引言/动机章节写作

```
帮我写一份机器学习课程大作业报告的【引言/背景】章节草稿。

项目信息：
- 题目：基于传统机器学习与深度学习方法的中文电商评论情感三分类对比研究
- 任务：将京东电商评论分类为好评、中评、差评
- 技术方案：三层八模型对比
  - 传统ML：朴素贝叶斯 + SVM + XGBoost（TF-IDF 特征）
  - 从零DL：TextCNN + BiGRU+Attention（Word2Vec 词向量）
  - 预训练：BERT → BERT-wwm → RoBERTa-wwm（中文预训练模型微调）
- 数据集：京东真实评论（720 万条中分层抽样 5 万条，按星级映射三分类）

请在引言中引用这些论文作为背景支撑：
- Zhao et al. (2023) 中文电商方面级情感分析
- 可简要提及电商评论分析的商业价值（不需要深入引用）

章节要求：
1. 开场：用一个具体的京东场景说明情感分析的价值——
   例：商家上架一款新品后，每天收到数千条评论。仅靠平均分（如 4.2 分）无法知道用户究竟哪里不满意。
   不少用户打 3 星但文字表达的是强烈不满，也有用户打 5 星但文字是"还可以"——
   评分和文字情感之间存在不一致，这是问题的起点

2. 说明三分类比二分类更有实际意义：
   - 好评代表满意度，差评代表流失风险
   - 中评往往包含建设性批评和具体改进方向，对商家最有价值
   - 三分类能帮商家做更精细的用户反馈分析

3. 从规则方法的局限 → 传统ML的方法 → 深度学习的发展 → 预训练范式的崛起，
   形成"为什么要做三层对比"的自然递进逻辑

4. 清晰列出本文的研究问题（Research Questions）：
   - RQ1: 传统ML、从零DL、预训练微调三种范式在中文电商评论上的性能边界在哪里？
   - RQ2: 同一层内不同方法之间是否存在显著差异？（如 CNN vs RNN+Attn，BERT vs RoBERTa）
   - RQ3: 模型性能与推理效率之间如何权衡？在实际电商场景中应如何选型？

5. 语气像本科生能写出来的，不要太花哨，500-600 字
6. 用中文写作
```

---

## 提示词 4：方法论章节（含公式 + 论文引用）

```
帮我写机器学习大作业报告的【方法论】章节草稿，要求包含核心公式（LaTeX $$...$$ 格式）和对应的论文引用。

项目技术方案（三层八模型对比）：

===== 3.1 传统机器学习方法 =====

3.1.1 文本特征：TF-IDF
- 写出 TF(t,d) 和 IDF(t) 的计算公式
- 写出 TF-IDF(t,d) = TF(t,d) × IDF(t) 的最终形式
- 特征维度 5000，ngram_range=(1,2)，包含 unigram 和 bigram
- 参数选择依据和合理性说明

3.1.2 朴素贝叶斯分类器
- 写出多项式朴素贝叶斯的条件概率公式
- 说明其在文本分类中的独立性假设及该假设在实践中的合理性

3.1.3 线性支持向量机
- 写出线性 SVM 的优化目标公式（hinge loss + L2 正则化）
- 说明高维稀疏文本上线性 SVM 使用 LinearSVC（而非 kernel SVM）的效率和效果优势

3.1.4 XGBoost
- 写出 XGBoost 的近似目标函数——泰勒展开二阶形式 + 叶子节点个数与权重 L2 正则项
- 引用：Chen & Guestrin (2016) XGBoost: A Scalable Tree Boosting System, KDD
- 说明 XGBoost 在结构化特征上表现优异，但对高维稀疏文本不如线性模型，这个对比本身是实验的看点

===== 3.2 深度学习从零训练方法 =====

3.2.1 词向量（Word2Vec）
- 使用 jieba 分词
- 使用 gensim 在训练集上训练 Word2Vec（skip-gram, 维度=300, 窗口=5, min_count=3）
- 未登录词用均匀分布 U(-0.01, 0.01) 随机初始化，并在训练中更新
- TextCNN 和 BiGRU+Attention 共享同一份词向量矩阵

3.2.2 TextCNN
- 写出卷积操作公式：c_i = f(W · x_{i:i+h-1} + b)
- ReLU 激活 + 最大池化 + Concat + Dropout(0.5) + 全连接分类层
- 卷积核尺寸 h ∈ {3, 4, 5}，每种 100 个滤波器
- 引用：Kim (2014) Convolutional Neural Networks for Sentence Classification, EMNLP
- 说明 CNN 在文本上的核心直觉：不同尺寸的卷积核捕捉不同长度的 n-gram 局部特征

3.2.3 BiGRU + Attention
- 写出 GRU 门控公式（重置门 r_t = σ(W_r·[h_{t-1}, x_t])，更新门 z_t = σ(W_z·[h_{t-1}, x_t])）
- 画出 BiGRU 双向编码：→h_t 和 ←h_t 拼接为 h_t = [→h_t; ←h_t]
- 写出 Attention 权重计算：u_i = tanh(W_h·h_i + b_h)，α_i = softmax(u_i^T·u_ω)，c = Σ α_i·h_i
- 隐藏层维度 128，双向拼接后 256
- Dropout(0.5) + 全连接分类
- 引用：王伟等 (2019)《基于 BiGRU-attention 神经网络的文本情感分类模型》, 计算机应用研究（核心引用）
- 引用：Yang et al. (2016) HAN, NAACL（Attention 权重设计）
- 引用：Bahdanau et al. (2015) Neural Machine Translation by Jointly Learning to Align and Translate, ICLR（Attention 机制起源）
- 说明用 GRU 而非 LSTM 的理由：GRU 参数减少约 1/4，训练更快，在短文本分类上与 LSTM 性能相当

3.2.4 训练设置
- 优化器 Adam（lr=1e-3）
- Batch size=64
- Epoch=20，Early Stopping（patience=3, monitor val_loss）
- 交叉熵损失函数
- 梯度裁剪（max_norm=5）

===== 3.3 预训练语言模型微调 =====

3.3.1 BERT-base-Chinese
- 写出 Transformer 多头自注意力：MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O，
  其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)，Attention(Q,K,V) = softmax(QK^T/√d_k)V
- BERT 分类头：P(label|input) = softmax(W·h_{[CLS]} + b)
- 引用：Vaswani et al. (2017) Attention Is All You Need, NeurIPS（Transformer 原理）
- 引用：Devlin et al. (2019) BERT, NAACL（BERT 预训练 + 微调范式）

3.3.2 Chinese-BERT-wwm-ext
- 解释全词掩码（Whole Word Masking, WWM）对中文的核心价值：
  中文词由多个字组成，BERT 原版按字 mask（如"苹果"可能 mask 成"[MASK]果"），
  模型通过"果"字可轻松推断缺失字，掩码训练效果打折扣。
  WWM 将同一词的所有字一起 mask，迫使模型学习更深的语义。
- 引用：Cui et al. (2021) Pre-Training with Whole Word Masking for Chinese BERT

3.3.3 Chinese-RoBERTa-wwm-ext
- 解释 RoBERTa 相对 BERT 三项改进：
  (1) 动态掩码——每个 epoch 的 mask 位置随机变化 vs BERT 静态同一份
  (2) 去掉 NSP（Next Sentence Prediction）
  (3) 更大 batch size 和更多训练数据
- 引用：Liu et al. (2019) RoBERTa
- 引用：Cui et al. (2021)（RoBERTa-wwm 中文实现）

3.3.4 微调策略
- 超参：lr=2e-5, epoch=3, batch_size=16, max_length=128, warmup_ratio=0.1
- 引用：Sun et al. (2019) How to Fine-Tune BERT for Text Classification?（作为超参选择的理论依据）
- 三个 BERT 系列模型使用完全相同的微调超参，保证可比性

===== 3.4 评估指标 =====

- 准确率 (Accuracy)：所有正确预测 / 总样本
- 精确率 (Precision)：TP/(TP+FP)，对差评（0）/中评（1）/好评（2）分别计算
- 召回率 (Recall)：TP/(TP+FN)
- F1 值：2 × Precision × Recall / (Precision + Recall)
- 宏平均 F1 (Macro-F1)：三类 F1 的简单平均（不按样本数加权），
  因为类别可能不均衡，宏平均比微平均更能反映模型在所有类别上的综合能力（写公式）
- 额外记录指标：训练时间（秒）、推理速度（samples/sec，在测试集上计算）、模型参数量、模型文件大小（MB）

===== 3.5 模型架构对比总览表 =====

用表格对比 8 个模型的：参数量级、训练方式、是否需要分词、是否需要预训练、理论复杂度。
```

---

## 提示词 5：实验框架代码

```
帮我写一个完整的 PyTorch 实验框架代码，用于京东电商评论情感三分类的多模型对比实验。

===== 需要实现的模型（8 个）=====

传统 ML（sklearn，共享 TF-IDF 特征）：
  - MultinomialNB（多项式朴素贝叶斯）
  - LinearSVC（线性 SVM，参数 C=1.0）
  - XGBClassifier（XGBoost，参数 max_depth=6, n_estimators=100）

深度学习从零训练（PyTorch，共享 Word2Vec 词向量矩阵）：
  - TextCNN：卷积核尺寸 [3,4,5]，每种 100 个滤波器，Dropout=0.5
  - BiGRU + Attention：GRU 隐藏层 128，双向拼接后 256，Dropout=0.5

预训练微调（HuggingFace transformers）：
  - bert-base-chinese
  - hfl/chinese-bert-wwm-ext
  - hfl/chinese-roberta-wwm-ext

===== 项目结构 =====

config.py          — 所有超参集中管理
data_loader.py     — 数据加载、预处理、分词、Word2Vec 训练、DataLoader 构建
traditional_ml.py  — 朴素贝叶斯 + SVM + XGBoost（TF-IDF + 训练 + 评估）
dl_models.py       — TextCNN 和 BiGRU+Attention 的 PyTorch 模型类定义
train_eval.py      — 深度学习训练循环、评估函数、EarlyStopping、结果记录
main.py            — 主入口，依次运行所有 8 个模型的实验

===== 代码要求 =====

[通用]
- 统一输入格式：使用提示词 2 产出的 train.csv / val.csv / test.csv（三列：original_text, tokens, label）
- BERT 系列用 original_text 列，其余模型用 tokens 列（空格分隔的 jieba 分词结果）
- 设置全局随机种子 42（random / numpy / torch / transformers）
- 所有实验指标写入同一个汇总 CSV（results.csv）

[传统 ML 部分]
- TF-IDF 向量化：max_features=5000, ngram_range=(1,2)，仅在训练集上 fit，然后 transform 验证集和测试集
- 三个模型共享同一份 TF-IDF 特征矩阵
- 每个模型：fit → predict → 输出 classification_report → 记录所有指标和训练/推理时间
- 保存 vectorizer（pickle），方便后续使用

[深度学习从零训练部分]
- 在训练集的 tokens 上训练 Word2Vec（gensim, skip-gram, dim=300, window=5, min_count=3）
- 构建词表（word → index），未登录词映射到 <UNK>
- embedding 层用 Word2Vec 向量初始化，<UNK> 和 <PAD> 用 U(-0.01, 0.01) 随机初始化
- embedding 层在训练中不冻结（trainable）
- 统一训练设置：optimizer=Adam(lr=1e-3), batch_size=64, epoch=20
- EarlyStopping: monitor='val_loss', patience=3, restore_best_weights=True
- 梯度裁剪 max_norm=5
- 每个 epoch 记录 train_loss, val_loss, val_acc
- 训练结束后：画出 loss 和 accuracy 的 epoch 曲线、画出混淆矩阵（3×3）

[预训练微调部分]
- 三个模型统一微调超参：lr=2e-5, epoch=3, batch_size=16, max_length=128, warmup_ratio=0.1
- 使用 HuggingFace AutoTokenizer + AutoModelForSequenceClassification（num_labels=3）
- 训练结束后：画出 loss 曲线、画出混淆矩阵

[汇总输出]
- 生成汇总表格（DataFrame），行=8 个模型，列=Accuracy/Precision/Recall/F1(每类+宏平均)/训练时间/推理速度/参数量/模型大小
- 画"准确率 vs 推理速度"散点图（8 个点，用颜色区分模型层级）
- 画 8 个混淆矩阵集中展示在一张大图上（2 行 × 4 列）
- 保存所有结果为文件：results.csv / results.png / confusion_matrices.png

[Colab 兼容]
- 所有 !pip install 放在文件开头
- 使用 tqdm 显示进度条
- 及时 del 不需要的模型并 torch.cuda.empty_cache() 释放显存
- BERT 三个模型训练完一个释放一个再训下一个，避免 OOM
```

---

## 提示词 6：两周时间计划

```
我是一个计算机本科生，有两周（14 天）时间完成机器学习大作业。请帮我制定详细的时间计划表。

项目信息：
- 任务：京东电商评论情感三分类（好评/中评/差评）
- 数据：JD 720 万条中抽样 5 万条（1-5 星映射为三分类）
- 模型：8 个
  - 传统 ML：朴素贝叶斯 / SVM / XGBoost
  - 从零 DL：TextCNN / BiGRU+Attention
  - 预训练微调：BERT / BERT-wwm / RoBERTa-wwm
- 开发环境：Google Colab（T4 GPU）
- 每天可投入 3-4 小时

训练时间估算（T4 GPU）：
- TF-IDF + 3 个传统 ML：< 15 分钟（纯 CPU）
- Word2Vec 训练：~10 分钟（CPU）
- TextCNN（20 epochs）：~1 小时（GPU）
- BiGRU+Attention（20 epochs）：~1.5 小时（GPU）
- BERT 微调 × 3（各 3 epochs）：~2-3 小时/个，共 ~7 小时（GPU）
- 总训练时间：约 10 小时（GPU 部分可以穿插着跑）

约束条件：
- Colab 免费版有 GPU 使用时长限制，连续用太久会被限速
- 建议 DL 从零训练的两个模型同一天跑，BERT 系列分两天跑

输出要求：
- 共 14 天，每天明确任务标题 + 具体交付物
- 标注 3-4 个"关键节点"（绝对不能拖延的天）
- 预留 2 天缓冲时间
- 标注哪天开始写报告、哪天完成初稿
- 考虑到 Colab GPU 限制，给出分批训练的具体安排
- 最后一天用于最终检查和润色
- 每天标注预计耗时（小时）
```

---

## 提示词 7：结果分析与讨论

```
我的京东电商评论情感三分类对比实验跑完了，请帮我写【实验结果与分析】章节的讨论框架。

实验设计回顾：
- 数据：京东真实评论 5 万条（train/val/test = 4万/5千/5千），1-5 星映射为三分类
- 模型：8 个，分三层
  - 传统 ML：朴素贝叶斯 / SVM / XGBoost（TF-IDF 向量化）
  - 从零 DL：TextCNN / BiGRU+Attention（Word2Vec 词向量）
  - 预训练微调：BERT-base / BERT-wwm / RoBERTa-wwm
- 指标：Accuracy / Precision / Recall / 各类别 F1 / Macro-F1 / 训练时间 / 推理速度 / 模型大小

请帮我：

===== 4.1 实验结果总览 =====
1. 设计一个汇总结果对比表（8 行 × 10 列），按 Macro-F1 从高到低排序
2. 标注每一层的最优方法（用粗体或符号标记）
3. 设计一个"准确率 vs 推理速度"双轴散点图，8 个点按层着色，
   在旁边标注每个点对应的模型名称

===== 4.2 层间对比分析（回答 RQ1）=====
讨论三个范式之间的性能差距：
  a) 传统 ML 的 F1 天花板大概在哪？三个传统方法中表现最好的是哪个？为什么？
  b) 从零 DL 比传统 ML 提升多少？这个提升的来源是什么（词向量语义表达 + 非线性特征提取）？
  c) 预训练微调比从零 DL 又提升了多少？预训练知识迁移的核心价值在哪里？
  d) 这些提升在三个类别（好/中/差）上是均匀的还是不均匀的？如果不均匀，说明什么？

===== 4.3 层内对比分析（回答 RQ2）=====
  a) 传统 ML 层：SVM vs XGBoost，线性模型在文本分类上是否优于树模型？
     朴素贝叶斯的朴素假设在什么情况下成立/不成立？
  b) DL 层：BiGRU+Attention vs TextCNN，序列模型 vs 卷积模型在中文短文本上谁更优？
     中评（中性）这个类别在两种模型上的 F1 差异是否更大？如果更大，可能原因是什么？
  c) 预训练层：BERT-wwm 比 BERT 提升多少？（全词掩码的实际收益）
     RoBERTa-wwm 比 BERT-wwm 提升多少？（动态掩码+去掉NSP+Larger Batch的实际收益）
     三个预训练模型的"边际收益递减"是否明显？是否值得升级到 RoBERTa？

===== 4.4 效率对比（回答 RQ3）=====
  a) 画一张表：Macro-F1 vs 推理速度 vs 模型文件大小，三列数值对比
  b) 如果要在电商平台上真实部署，你会推荐哪个模型？给出你的"性能-效率"帕累托分析
  c) 分场景讨论：
     - 实时在线服务（毫秒级响应）→ 推荐哪个？
     - 离线批量分析（分钟级可接受）→ 推荐哪个？
     - 标注数据极少（<1000条）→ 哪个模型更鲁棒？

===== 4.5 "中评"分类难点分析 =====
  a) 所有模型中，中评（标签 1）的 F1 是否普遍低于好评和差评？
  b) 分析原因：中性评论的语义模糊性——"还行""一般""凑合吧"这些词
     在不同上下文中的真实情感倾向可能完全不同。从混淆矩阵看，中评最容易被误分成什么？
  c) 这对电商情感分析系统的实际部署意味着什么？

===== 4.6 错误分析 =====
  a) 设计一个表格模板，展示从最佳模型（预计是 RoBERTa-wwm）中挑出的 6-8 个分类错误案例
  b) 表格栏：原始文本 / 真实标签 / 预测标签 / 错误类型
  c) 错误类型分类：
     - 评分与文字不一致（文字强烈不满但打了 3 星 → 真实为中评，模型判断为差评）
     - 反讽/阴阳怪气（"快递真快，三天就到了"实际是抱怨）
     - 语境依赖（"这个价格也就这样了"可能是中性也可能是差评）
     - 标签映射边界模糊（2 星和 3 星的语义距离 < 3 星和 4 星的语义距离？）

===== 4.7 消融实验（选做，如果来不及做就在讨论里说明其必要性）=====
  a) TF-IDF 特征维度对传统 ML 的影响（对比 2000 vs 5000 vs 10000 维）
  b) BiGRU 去掉 Attention 层后的性能变化（纯 BiGRU vs BiGRU+Attention）
  c) 冻结 vs 不冻结 Word2Vec 词向量对从零 DL 的影响
  d) 如果时间不够做全部，选 1 个最有信息量的做

===== 4.8 局限性与未来工作 =====
  a) 本研究局限：
     - 未做系统性的超参搜索（受限于时间）
     - 数据仅来自京东一个平台，结论的外推性有限
     - 三分类的评分映射策略（1-2→差/3→中/4-5→好）是否最优未做验证
     - 未探索更大模型（如 RoBERTa-large）
  b) 未来方向：
     - 方面级情感分析（ABSA）：不只是整体情感，而是针对具体方面（物流、质量、价格）的分析
     - 跨平台迁移：在京东上训练的模型在淘宝数据上泛化如何？
     - 数据增强：用大模型生成更多中评样本缓解类别模糊问题

语气要求：客观学术、实事求是（不要吹嘘）、本科生能写出来的水平。用中文写作。
```

---

## 使用顺序

| 第几天 | 提示词 | 用途 |
|--------|--------|------|
| Day 1 | 提示词 1 + 提示词 6 | 规划确认 + 时间表 |
| Day 1-2 | 提示词 2 | 数据准备 |
| Day 2-3 | 提示词 5 | 搭建实验框架 |
| Day 8-10 | 提示词 3 | 引言写作（模型训练期间穿插） |
| Day 10-12 | 提示词 4 | 方法论写作（实验跑完后） |
| Day 12-14 | 提示词 7 | 结果分析与讨论（实验全部跑完后） |

---

## 引用文献 BibTeX（备用）

```bibtex
@inproceedings{kim2014convolutional,
  title={Convolutional Neural Networks for Sentence Classification},
  author={Kim, Yoon},
  booktitle={EMNLP},
  year={2014}
}

@article{wang2019bigru,
  title={基于BiGRU-attention神经网络的文本情感分类模型},
  author={王伟 and 孙玉霞 and 齐庆杰 and 孟祥福},
  journal={计算机应用研究},
  volume={36},
  number={12},
  year={2019}
}

@inproceedings{yang2016hierarchical,
  title={Hierarchical Attention Networks for Document Classification},
  author={Yang, Zichao and Yang, Diyi and Dyer, Chris and He, Xiaodong and Smola, Alex and Hovy, Eduard},
  booktitle={NAACL},
  year={2016}
}

@inproceedings{bahdanau2015neural,
  title={Neural Machine Translation by Jointly Learning to Align and Translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  booktitle={ICLR},
  year={2015}
}

@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={NAACL},
  year={2019}
}

@article{liu2019roberta,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{cui2021pre,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing},
  journal={arXiv preprint arXiv:1906.08101},
  year={2021}
}

@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={KDD},
  year={2016}
}

@article{sun2019finetune,
  title={How to Fine-Tune BERT for Text Classification?},
  author={Sun, Chi and Qiu, Xipeng and Xu, Yige and Huang, Xuanjing},
  journal={arXiv preprint arXiv:1905.05583},
  year={2019}
}

@inproceedings{zhao2023pos,
  title={POS-ATAEPE-BiLSTM: An Aspect-Based Sentiment Analysis Algorithm Considering Part-of-Speech Embedding},
  author={Zhao, Qing and Mo, Zhenfang and Fan, Meng},
  booktitle={Applied Intelligence},
  volume={53},
  pages={27440--27458},
  year={2023}
}
```
