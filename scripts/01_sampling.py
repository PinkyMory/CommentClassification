"""分层抽样: 从 JD 评论训练集中制作三分类数据集, train/val/test = 8:1:1
用法: python scripts/01_sampling.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import SEED, PROCESSED_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, star_to_label

np.random.seed(SEED)


def load_raw_data(input_path: str) -> pd.DataFrame:
    """加载原始数据"""
    print(f"加载原始数据: {input_path}")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".tsv"):
        df = pd.read_csv(input_path, sep="\t")
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")
    return df


def find_text_and_star_columns(df: pd.DataFrame) -> tuple[str, str]:
    """自动检测评论列和评分列"""
    # 优先级从高到低：越具体的匹配越靠前
    # 优先级从高到低：越具体的匹配越靠前
    text_candidates = ["评论内容", "评论标题", "content", "comment", "text", "review", "内容"]
    star_candidates = ["评分", "star", "score", "rating", "stars", "星级"]

    # 用 candidate 去匹配列（而非列去匹配 candidate），保证优先级高的先命中
    text_col = None
    star_col = None

    for candidate in text_candidates:
        for col in df.columns:
            if candidate in col.lower():
                text_col = col
                break
        if text_col:
            break

    for candidate in star_candidates:
        for col in df.columns:
            if candidate in col.lower():
                star_col = col
                break
        if star_col:
            break

    if text_col is None or star_col is None:
        print(f"可用列: {list(df.columns)}")
        raise RuntimeError("未自动检测到评论列或评分列，请手动指定 --text-col 和 --star-col")

    print(f"检测到 → 评论列: '{text_col}', 评分列: '{star_col}'")
    return text_col, star_col


def prepare_data(df: pd.DataFrame, text_col: str, star_col: str) -> pd.DataFrame:
    """清洗、打标签，保留全部有效数据"""
    df = df.copy()
    df["label"] = df[star_col].apply(star_to_label)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    col_map = {text_col: "text"}
    if star_col != "star":
        col_map[star_col] = "star"
    df = df.rename(columns=col_map)
    # 确保有 text, star, label 三列
    cols_to_keep = ["text", "label"]
    if "star" in df.columns:
        cols_to_keep = ["text", "star", "label"]
    return df[cols_to_keep].reset_index(drop=True)


def show_distributions(df: pd.DataFrame):
    """显示评分和标签分布"""
    label_names = {0: "差评", 1: "中评", 2: "好评"}

    print(f"\n总有效数据: {len(df):,} 条\n")

    # 原始星级分布
    if "star" in df.columns:
        print("原始星级分布:")
        star_dist = df["star"].value_counts().sort_index()
        for star, count in star_dist.items():
            print(f"  {int(star)} 星: {count:>6,} 条 ({100 * count / len(df):5.1f}%)")

    # 三分类分布
    print("\n三分类标签分布:")
    label_dist = df["label"].value_counts().sort_index()
    for label, count in label_dist.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {label_names[label]}: {count:>6,} 条 ({pct:5.1f}%) {bar}")

    # 类别不平衡比例
    min_count = label_dist.min()
    max_count = label_dist.max()
    print(f"\n不平衡比 (最多/最少): {max_count / min_count:.1f}:1")


def split_and_save(df: pd.DataFrame):
    """分层切分 8:1:1 并保存"""
    # 先分出 train + temp (9:1)
    train, temp = train_test_split(
        df, test_size=0.2,
        stratify=df["label"], random_state=SEED,
    )
    # 再平分 temp → val + test
    val, test = train_test_split(
        temp, test_size=0.5,
        stratify=temp["label"], random_state=SEED,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    label_names = {0: "差评", 1: "中评", 2: "好评"}

    print(f"\n数据集已保存（分层切分 8:1:1，分布一致）:")
    for name, path in [("训练集", TRAIN_PATH), ("验证集", VAL_PATH), ("测试集", TEST_PATH)]:
        sub = pd.read_csv(path)
        dist = sub["label"].value_counts().sort_index()
        parts = ", ".join(f"{label_names[k]}: {v:,}" for k, v in dist.items())
        print(f"  {name}: {len(sub):,} 条 → {path}")
        print(f"         {parts}")


def main():
    parser = argparse.ArgumentParser(description="JD 评论数据分层抽样（三分类）")
    parser.add_argument("--input", type=str,
                        default="data/raw/训练集.csv",
                        help="原始数据路径")
    parser.add_argument("--text-col", type=str, default=None,
                        help="评论列名（自动检测）")
    parser.add_argument("--star-col", type=str, default=None,
                        help="评分列名（自动检测）")
    args = parser.parse_args()

    df = load_raw_data(args.input)

    if args.text_col and args.star_col:
        text_col, star_col = args.text_col, args.star_col
    else:
        text_col, star_col = find_text_and_star_columns(df)

    df = prepare_data(df, text_col, star_col)
    show_distributions(df)
    split_and_save(df)

    print("\n抽样完成！")
    print("注意: 类别不平衡（好评多、差评少），训练时已配置 class_weight 处理。")


if __name__ == "__main__":
    main()
