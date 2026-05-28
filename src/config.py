from pathlib import Path

# ---- 项目根目录（自动推断，不写死）----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- 数据路径 ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# ---- 输出路径 ----
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.csv"

# ---- 随机种子 ----
SEED = 42

# ---- 采样参数 ----
USE_ALL_DATA = True           # True: 使用全部可用数据; False: 按数量抽样
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
USE_CLASS_WEIGHT = True       # 用类别权重处理不平衡

# ---- 文本预处理 ----
MAX_SEQ_LEN = 128
MAX_SEQ_LEN_BERT = 256

# ---- TF-IDF ----
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ---- Word2Vec ----
EMBEDDING_DIM = 300

# ---- 深度学习从头训练 ----
BATCH_SIZE_DL = 64
EPOCHS_DL = 30
LR_DL = 1e-3
DROPOUT = 0.5

# ---- 预训练模型 ----
BERT_MODEL_NAME = "bert-base-chinese"
ROBERTA_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
BATCH_SIZE_PRETRAINED = 16
EPOCHS_PRETRAINED = 5
LR_PRETRAINED = 2e-5

# ---- 标签映射 ----
LABEL_MAP = {0: "差评", 1: "中评", 2: "好评"}
NUM_CLASSES = 3

# ---- 星评 → 三分类映射 ----
def star_to_label(star: int) -> int:
    """1-2星 → 差评(0), 3星 → 中评(1), 4-5星 → 好评(2)"""
    if star <= 2:
        return 0
    elif star == 3:
        return 1
    else:
        return 2
