from transformers import BertForTokenClassification

def build_ate_model(num_labels=3):
    return BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
