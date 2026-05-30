import matplotlib
matplotlib.use("Agg")  # 无头服务器兼容，必须在 import pyplot 之前
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 中文支持（Linux 无 SimHei 时回退到 DejaVu Sans）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: str = None, title: str = "混淆矩阵"):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(history: dict, save_path: str = None):
    """绘制训练/验证损失和 F1 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss 曲线")
    ax1.legend()

    ax2.plot(epochs, history["val_f1"], "g-", label="Val Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("验证集 Macro-F1")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_csv_path: str, save_path: str = None):
    """从 results.csv 读取所有模型结果，画对比柱状图"""
    df = pd.read_csv(results_csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - width, df["accuracy"], width, label="Accuracy")
    ax.bar(x, df["macro_f1"], width, label="Macro-F1")
    ax.bar(x + width, df["weighted_f1"], width, label="Weighted-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylabel("Score")
    ax.set_title("模型对比")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_f1(results_csv_path: str, save_path: str = None):
    """绘制各模型在三类上的 F1 对比"""
    df = pd.read_csv(results_csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["f1_差评"], width, label="差评 F1")
    ax.bar(x, df["f1_中评"], width, label="中评 F1")
    ax.bar(x + width, df["f1_好评"], width, label="好评 F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylabel("F1 Score")
    ax.set_title("各类别 F1 对比")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
