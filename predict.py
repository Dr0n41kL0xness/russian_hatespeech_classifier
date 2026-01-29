import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "Dr0n41k/rubert-toxic-classifier"

def predict(text):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    
    return "Toxic" if prediction == 1 else "Neutral"

if __name__ == "__main__":
    test_text = "Курс доллара стабильно растет, что же нам делать?"
    result = predict(test_text)
    print(f"Текст: {test_text}")
    print(f"Результат: {result}")
