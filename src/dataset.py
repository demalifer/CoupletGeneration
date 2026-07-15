from torch.utils.data import DataLoader
from config import *
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration

def get_dataloader(tokenizer, model):
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset.set_format(type='torch')
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        padding=True,
        return_tensors='pt'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL)

    dataloader = get_dataloader(tokenizer, model)

    for batch in dataloader:
        for key, value in batch.items():
            print(key, '->', value)
        break