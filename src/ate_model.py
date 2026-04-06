import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class PhoBERT_CRF(nn.Module):
def **init**(self, num_labels):
super().**init**()
self.bert = AutoModel.from_pretrained("vinai/phobert-base")
self.dropout = nn.Dropout(0.3)

```
    self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    self.crf = CRF(num_labels, batch_first=True)

def forward(self, input_ids, attention_mask, labels=None):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.classifier(outputs.last_hidden_state)

    if labels is not None:
        loss = -self.crf(logits, labels, mask=attention_mask.bool())
        return loss
    else:
        return self.crf.decode(logits, mask=attention_mask.bool())
```
