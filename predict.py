import torch
from transformers import AutoTokenizer
from src.model import BertClassifier

def get_prediction(text, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)

    return "Toxic" if prediction.item() == 1 else "Neutral"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name)
    
    
    model.to(device)
    model.eval()

    print("Проверка классификатора Hate Speech")
    #тестовый текст
    user_input = "Курс доллара стабильно растет,что же нам делать?"
    result = get_prediction(user_input, model, tokenizer, device)
    print(f"Текст: {user_input}")
    print(f"Результат: {result}")
