from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

tag2id = {'O': 0, 'B-ASP': 1, 'I-ASP': 2}

def tokenize_ate(examples):
tokenized = tokenizer(
examples["sentence"],
truncation=True,
padding="max_length",
max_length=128
)

```
labels = []

for i, aspect in enumerate(examples["aspect_term"]):
    tokens = tokenizer.tokenize(examples["sentence"][i])
    aspect_tokens = tokenizer.tokenize(aspect)

    label_ids = [0] * len(tokens)

    for j in range(len(tokens)):
        if tokens[j:j+len(aspect_tokens)] == aspect_tokens:
            label_ids[j] = 1
            for k in range(1, len(aspect_tokens)):
                label_ids[j+k] = 2

    label_ids = label_ids[:128] + [0]*(128-len(label_ids))
    labels.append(label_ids)

tokenized["labels"] = labels
return tokenized
```
