from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 1.数据
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

#2. 文件
RAW_IN_DATA_FILE = 'in.txt'
RAW_OUT_DATA_FILE = 'out.txt'
BERT_MODEL = 'fnlp/bart-base-chinese'

#3. 超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 10
SAVE_STEPS = 50