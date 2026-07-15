import torch
from torch import optim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from transformers import BartForConditionalGeneration, AutoTokenizer

from torch.utils.tensorboard import SummaryWriter
import time

def train_one_step(model, inputs, optimizer, device):
    model.train()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    loss_value = outputs.loss
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_value.item()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)

    dataloader = get_dataloader(tokenizer, model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    min_loss = float('inf')
    step = 1
    for epoch in range(EPOCHS):
        print('='*15, f'EPOCH {epoch+1}', '='*15)

        for inputs in tqdm(dataloader, desc='training: '):
            this_loss = train_one_step(model, inputs, optimizer, device)

            if step % SAVE_STEPS == 0:
                tqdm.write(f'[Epoch {epoch+1} | Step {step}] Loss {this_loss:.4f}')
                writer.add_scalar('loss', this_loss, step)

                if this_loss < min_loss:
                    min_loss = this_loss
                    model.save_pretrained(MODEL_DIR)
                    tqdm.write('Model Saved successfully')
            step += 1
    writer.close()

if __name__ == '__main__':
    train()