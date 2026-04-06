from transformers import AutoModelForSequenceClassification

def build_atsc_model():
return AutoModelForSequenceClassification.from_pretrained(
"vinai/phobert-base",
num_labels=3
)
