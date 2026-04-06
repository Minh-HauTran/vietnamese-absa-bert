import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments
)

from data_loader import load_flipkart_db
from preprocess import build_absa_dataset
from ate_model import build_ate_model
from atsc_model import build_atsc_model

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from seqeval.metrics import f1_score

# ------------------
# Load data
# ------------------
df_raw = load_flipkart_db("data/raw/flipkart_products.db")
df = build_absa_dataset(df_raw)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ------------------
# ATE
# ------------------

tag2id = {"O": 0, "B-ASP": 1, "I-ASP": 2}

def tokenize_ate(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

dataset_ate = dataset.map(tokenize_ate)

model_ate = build_ate_model()

args_ate = TrainingArguments(
    output_dir="outputs/ate",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer_ate = Trainer(
    model=model_ate,
    args=args_ate,
    train_dataset=dataset_ate,
)

trainer_ate.train()

# ------------------
# ATSC
# ------------------

sentiment2id = {"negative": 0, "neutral": 1, "positive": 2}

def tokenize_atsc(example):
    tokens = tokenizer(
        example["sentence"],
        example["aspect_term"],
        truncation=True,
        padding="max_length"
    )
    tokens["labels"] = sentiment2id[example["sentiment"]]
    return tokens

dataset_atsc = dataset.map(tokenize_atsc)

model_atsc = build_atsc_model()

args_atsc = TrainingArguments(
    output_dir="outputs/atsc",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer_atsc = Trainer(
    model=model_atsc,
    args=args_atsc,
    train_dataset=dataset_atsc,
)

trainer_atsc.train()
