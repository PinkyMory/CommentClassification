import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.preprocess import tokenize


class TokenizedDataset(Dataset):
    """将分词后的文本转为索引序列，供 TextCNN/BiGRU 使用"""

    def __init__(self, csv_path: str, word2idx: dict, max_len: int = 128):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.max_len = max_len
        self.word2idx = word2idx
        self.pad_idx = word2idx.get("<PAD>", 0)
        self.unk_idx = word2idx.get("<UNK>", 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        # 截断或填充到 max_len
        token_ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        token_ids = token_ids[: self.max_len]
        attention_mask = [1] * len(token_ids)
        # padding
        pad_len = self.max_len - len(token_ids)
        token_ids += [self.pad_idx] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_data_loader(
    csv_path: str, word2idx: dict, batch_size: int = 64, max_len: int = 128, shuffle: bool = True
) -> DataLoader:
    dataset = TokenizedDataset(csv_path, word2idx, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_vocab_from_csv(csv_path: str, min_freq: int = 2, max_vocab: int = 30000) -> dict:
    """从训练数据构建词表"""
    from collections import Counter

    df = pd.read_csv(csv_path)
    counter = Counter()
    for text in df["text"]:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def build_embedding_matrix(word2idx: dict, wv_path: str, embed_dim: int = 300) -> np.ndarray:
    """从预训练词向量构建 embedding matrix"""
    from gensim.models import KeyedVectors

    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)
    matrix = np.random.normal(scale=0.01, size=(len(word2idx), embed_dim)).astype(np.float32)
    matrix[0] = 0.0  # <PAD>

    hit = 0
    for word, idx in word2idx.items():
        if word in wv:
            matrix[idx] = wv[word]
            hit += 1

    print(f"词表覆盖: {hit}/{len(word2idx)} ({100 * hit / len(word2idx):.1f}%)")
    return matrix
