"""传统机器学习模型训练: TF-IDF → NB / SVM / XGBoost
用法: python scripts/02_train_traditional.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.config import (
    SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    CHECKPOINT_DIR, RESULTS_PATH, FIGURE_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
)
from src.preprocess import clean_text
from src.metrics import compute_metrics, print_metrics, append_to_results_csv, save_metrics_to_file
from src.plot import plot_confusion_matrix
import jieba

np.random.seed(SEED)


def tokenize_and_join(texts: list[str]) -> list[str]:
    """jieba 分词后用空格连接"""
    return [" ".join(jieba.cut(clean_text(t))) for t in texts]


def main():
    print("加载数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 合并 train+val 训练传统模型
    train_all = pd.concat([train_df, val_df], ignore_index=True)
    X_train_raw = train_all["text"].tolist()
    y_train = train_all["label"].tolist()
    X_test_raw = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"训练集: {len(X_train_raw):,} 条, 测试集: {len(X_test_raw):,} 条")

    # TF-IDF
    print("\n=== TF-IDF 特征提取 ===")
    print("分词中...")
    X_train_tokens = tokenize_and_join(X_train_raw)
    X_test_tokens = tokenize_and_join(X_test_raw)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_tokens)
    X_test_tfidf = vectorizer.transform(X_test_tokens)
    print(f"特征维度: {X_train_tfidf.shape[1]}")

    # ---- 模型训练（全部加 class_weight 处理不平衡）----
    models = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(
            C=1.0, max_iter=2000, random_state=SEED, dual=False,
            class_weight="balanced",
        )),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="mlogloss",
        ),
    }
    # XGBoost 用 scale_pos_weight 不够直接，改算样本权重
    from sklearn.utils.class_weight import compute_sample_weight
    xgb_sample_weight = compute_sample_weight("balanced", y_train)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"训练 {name}...")
        if name == "XGBoost":
            model.fit(X_train_tfidf, y_train, sample_weight=xgb_sample_weight)
        else:
            model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        metrics = compute_metrics(y_test, y_pred)
        print_metrics(metrics)

        # 保存模型
        ext = "pkl" if name != "XGBoost" else "json"
        save_path = CHECKPOINT_DIR / f"{name.lower()}.{ext}"
        if ext == "pkl":
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model.save_model(str(save_path))
        print(f"模型已保存到 {save_path}")

        append_to_results_csv(RESULTS_PATH, name, metrics)

        # 保存混淆矩阵图
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        model_key = name.lower()
        cm_path = FIGURE_DIR / f"{model_key}_confusion_matrix.png"
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), ["差评", "中评", "好评"],
            save_path=str(cm_path), title=f"{name} 混淆矩阵",
        )
        plt.close("all")

        # 保存评估报告文本
        report_path = FIGURE_DIR / f"{model_key}_report.txt"
        save_metrics_to_file(metrics, str(report_path), name)

    # 保存 vectorizer 供 demo 使用
    vectorizer_path = CHECKPOINT_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nTF-IDF Vectorizer 已保存到 {vectorizer_path}")


if __name__ == "__main__":
    main()
