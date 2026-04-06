import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def build_absa_dataset(df):
    """
    Convert raw reviews → synthetic ABSA format
    """
    data = []

    for _, row in df.iterrows():
        sentence = clean_text(row.get("review", ""))

        if sentence == "":
            continue

        # simple heuristic
        aspect = "product"
        sentiment = "positive" if row.get("rating", 3) >= 4 else "negative"

        data.append({
            "sentence": sentence,
            "aspect_term": aspect,
            "sentiment": sentiment
        })

    return pd.DataFrame(data)
