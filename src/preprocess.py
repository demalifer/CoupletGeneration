from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from config import *

def preprocess():
    print('The preprocessing of data is starting...')

    with open(RAW_DATA_DIR/RAW_IN_DATA_FILE, 'r', encoding='utf-8') as f:
        inputs = [line.strip().replace(' ', '') for line in f.readlines()]

    with open(RAW_DATA_DIR/RAW_OUT_DATA_FILE, 'r', encoding='utf-8') as f:
        targets = [line.strip().replace(' ', '') for line in f.readlines()]

    dataset = Dataset.from_dict({
        'inputs': inputs,
        'targets': targets,
    })

    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)

    def batch_encode(example):
        inputs = tokenizer(
            text=example['inputs'],
            text_target=example['targets'],
            return_token_type_ids=False
        )
        return inputs

    dataset = dataset.map(batch_encode, batched=True, remove_columns=['inputs', 'targets'])


    dataset.save_to_disk(PROCESSED_DATA_DIR)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()
