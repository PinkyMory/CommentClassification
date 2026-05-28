"""预训练模型微调: BERT-base-Chinese + RoBERTa-wwm-ext
用法: python scripts/04_train_pretrained.py [--model bert|roberta|all]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    CHECKPOINT_DIR, RESULTS_PATH, FIGURE_DIR,
    BERT_MODEL_NAME, ROBERTA_MODEL_NAME,
    BATCH_SIZE_PRETRAINED, EPOCHS_PRETRAINED, LR_PRETRAINED,
    MAX_SEQ_LEN_BERT, NUM_CLASSES, LABEL_MAP,
)
from src.preprocess import clean_text
from src.metrics import compute_metrics, print_metrics, append_to_results_csv, save_metrics_to_file
from src.plot import plot_confusion_matrix, plot_training_curves
import matplotlib.pyplot as plt

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightedTrainer(Trainer):
    """带类别权重的 Trainer，用于处理不平衡数据"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(pred):
    """HuggingFace Trainer 用的评估函数"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def load_csv_as_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].astype(str).apply(clean_text)
    return Dataset.from_pandas(df[["text", "label"]])


def train_model(model_name: str, model_path: str):
    print(f"\n{'='*50}")
    print(f"微调 {model_name}")

    # 加载数据
    ds_train = load_csv_as_dataset(TRAIN_PATH)
    ds_val = load_csv_as_dataset(VAL_PATH)
    ds_test = load_csv_as_dataset(TEST_PATH)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            padding="max_length", max_length=MAX_SEQ_LEN_BERT,
        )

    ds_train = ds_train.map(tokenize_fn, batched=True)
    ds_val = ds_val.map(tokenize_fn, batched=True)
    ds_test = ds_test.map(tokenize_fn, batched=True)

    # 模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_CLASSES,
    )

    # 计算类别权重
    train_labels = pd.read_csv(TRAIN_PATH)["label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"类别权重: 差评={weights[0]:.3f}, 中评={weights[1]:.3f}, 好评={weights[2]:.3f}")

    # 训练参数
    output_dir = CHECKPOINT_DIR / f"{model_name}_intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS_PRETRAINED,
        per_device_train_batch_size=BATCH_SIZE_PRETRAINED,
        per_device_eval_batch_size=BATCH_SIZE_PRETRAINED * 2,
        learning_rate=LR_PRETRAINED,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # 测试集评估
    test_preds = trainer.predict(ds_test)
    y_true = test_preds.label_ids
    y_pred = test_preds.predictions.argmax(-1)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # 保存模型
    save_path = CHECKPOINT_DIR / f"{model_name}_best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"模型已保存到 {save_path}")

    append_to_results_csv(RESULTS_PATH, model_name, metrics)

    # 提取训练历史
    log_history = trainer.state.log_history
    train_loss = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in log_history if "eval_loss" in e]
    val_loss = [e["eval_loss"] for e in eval_logs]
    val_f1 = [e.get("eval_macro_f1", 0) for e in eval_logs]

    history = {"train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}

    # 保存训练曲线 + 混淆矩阵 + 报告
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if len(train_loss) > 0 and len(val_loss) > 0:
        plot_training_curves(history, str(FIGURE_DIR / f"{model_name}_training_curves.png"))
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), ["差评", "中评", "好评"],
        save_path=str(FIGURE_DIR / f"{model_name}_confusion_matrix.png"),
        title=f"{model_name} 混淆矩阵",
    )
    save_metrics_to_file(metrics, str(FIGURE_DIR / f"{model_name}_report.txt"), model_name)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["bert", "roberta", "all"])
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("bert", "all"):
        train_model("bert", BERT_MODEL_NAME)

    if args.model in ("roberta", "all"):
        train_model("roberta", ROBERTA_MODEL_NAME)


if __name__ == "__main__":
    main()
