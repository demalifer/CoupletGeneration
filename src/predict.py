import torch

from config import *
from transformers import AutoTokenizer, BartForConditionalGeneration

def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return outputs

def predict(text, model, tokenizer, device):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False,
    )
    inputs = {k:v.to(device) for k,v in inputs.items()}

    result = predict_batch(model, inputs)

    return tokenizer.batch_decode(result, skip_special_tokens=True)[0].replace(' ', '')

def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    print('vocabulary load success!')

    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    print('model load success!')

    print('Welcome to INTELEGER couplet generation model! print q or quit to exit...')
    while True:
        user_input = input('>the upper couplet: ')
        if user_input.strip().lower() in ['q', 'quit']:
            print('bye!')
            break
        elif user_input.strip() == '':
            print('please input valid content!')
            continue

        result = predict(user_input, model, tokenizer, device)
        print(f'the next couplet: {result}')

if __name__ == '__main__':
    run_predict()