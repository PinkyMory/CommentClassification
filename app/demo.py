"""Gradio 网页演示"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd
from app.model_loader import SentimentPredictor

# ---- 自动选择最优模型 ----
def _find_best_model():
    results_path = Path(__file__).resolve().parent.parent / "outputs" / "results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        best = df.loc[df["macro_f1"].idxmax()]
        print(f"自动选择最优模型: {best['model']} (Macro-F1: {best['macro_f1']:.4f})")
        return best["model"]
    return None


BEST_MODEL_NAME = _find_best_model()

# ---- 模型路径映射（按需修改）----
MODEL_CONFIGS = {
    "textcnn": {
        "type": "textcnn",
        "path": "checkpoints/textcnn_best.pth",
    },
    "bigru_attn": {
        "type": "bigru_attn",
        "path": "checkpoints/bigru_attn_best.pth",
    },
    "bert": {
        "type": "bert",
        "path": "checkpoints/bert_best",
    },
    "roberta": {
        "type": "roberta",
        "path": "checkpoints/roberta_best",
    },
}

predictor = None


def load_model(model_name: str):
    global predictor
    cfg = MODEL_CONFIGS[model_name]
    # 这里需要配套的 vectorizer / word2idx，实际使用时从训练脚本 pickle 加载
    # 简化版：预训练模型不需要额外组件
    if cfg["type"] in ("bert", "roberta"):
        predictor = SentimentPredictor(cfg["type"], cfg["path"])
        return f"已加载 {model_name}"
    else:
        return f"{model_name} 需要配套 vectorizer/word2idx，请先运行对应训练脚本"


def classify(text: str):
    if predictor is None:
        return {"请先选择模型": 1.0}
    label, probs = predictor.predict(text)
    return {predictor.labels[i]: float(p) for i, p in enumerate(probs)}


if BEST_MODEL_NAME:
    load_model(BEST_MODEL_NAME)

model_choices = list(MODEL_CONFIGS.keys())
default_model = BEST_MODEL_NAME if BEST_MODEL_NAME in model_choices else model_choices[0]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=5, placeholder="在这里输入一条京东商品评论...", label="评论文本"
    ),
    outputs=gr.Label(num_top_classes=3, label="情感预测"),
    title="京东商品评论情感三分类演示",
    description=f"当前模型: {BEST_MODEL_NAME or default_model}",
    examples=[
        ["这个商品质量非常好，用了一段时间了，很满意！"],
        ["一般般吧，凑合能用，没有想象中好。"],
        ["太差了，用了两天就坏了，差评差评差评！"],
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
