import re
import jieba


def clean_text(text: str) -> str:
    """去除HTML标签、URL、特殊字符，保留中英文和数字"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """jieba 分词，返回词语列表"""
    cleaned = clean_text(text)
    return [w for w in jieba.cut(cleaned) if w.strip()]


def tokenize_for_pretrained(text: str) -> str:
    """用于 BERT/RoBERTa：只清洗，不分词（由 tokenizer 处理）"""
    return clean_text(text)
