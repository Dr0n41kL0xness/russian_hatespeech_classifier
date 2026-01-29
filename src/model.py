import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_classes=2):
        """
        num_classes: количество классов (0 - нейтральный, 1 - токсичный).
        """
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        output = self.dropout(pooled_output)
        return self.out(output)
