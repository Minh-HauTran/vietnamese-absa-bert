import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from data_loader import load_data
from preprocess import tokenize_ate
from ate_model import PhoBERT_CRF
from atsc_model import build_atsc_model

# Load data

data = load_data("data/raw/dataset.csv")
dataset = Dataset.from_pandas(data)

# ATE preprocess

dataset_ate = dataset.map(tokenize_ate, batched=True)
dataset_ate.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

loader = DataLoader(dataset_ate, batch_size=8, shuffle=True)

# Model

model = PhoBERT_CRF(num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train ATE

for epoch in range(3):
model.train()
for batch in loader:
optimizer.zero_grad()

```
    loss = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device)
    )

    loss.backward()
    optimizer.step()
```

# Save ATE

torch.save(model.state_dict(), "outputs/models/ate.pt")

# Train ATSC

model_atsc = build_atsc_model()
torch.save(model_atsc.state_dict(), "outputs/models/atsc.pt")
