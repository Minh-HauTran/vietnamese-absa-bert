from transformers import BertForSequenceClassification

def build_atsc_model(num_labels=3):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
