# 🇻🇳 Aspect-Based Sentiment Analysis (ABSA) from Real-World E-commerce Data

## 📖 Abstract

This project presents a research-oriented implementation of Aspect-Based Sentiment Analysis (ABSA) using transformer-based models.

Unlike traditional ABSA benchmarks, this work explores how to **adapt raw, unstructured e-commerce data into an ABSA pipeline**. The system decomposes the problem into:

* Aspect Term Extraction (ATE)
* Aspect Term Sentiment Classification (ATSC)

A dual-stage architecture is implemented using BERT-based models, demonstrating how real-world datasets can be transformed into structured NLP tasks.

---

## 🎯 Problem Statement

Traditional sentiment analysis fails to capture fine-grained opinions.

Example:

> "The product quality is good but delivery is slow"

* Traditional SA → Neutral
* ABSA →

  * quality → Positive
  * delivery → Negative

---

## 🧠 Methodology

### 🔹 Task 1: Aspect Term Extraction (ATE)

* Formulated as sequence labeling (BIO tagging)
* Model: BERT for token classification
* Output: aspect spans

### 🔹 Task 2: Aspect Term Sentiment Classification (ATSC)

* Input format: [Sentence, Aspect]
* Model: BERT for sequence classification
* Output: sentiment polarity

---

## 🔄 Data Pipeline

Raw Flipkart Reviews (Kaggle)
↓
SQLite Data Extraction
↓
Text Preprocessing
↓
Synthetic Aspect Construction
↓
BIO Tagging (ATE)
↓
Model Training (ATE + ATSC)

---

## 📊 Dataset

* Source: Flipkart Product Reviews Dataset (Kaggle)
* Notebook: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis
* Format: SQLite database (`flipkart_products.db`)

The dataset contains:

* product metadata
* user reviews
* ratings

⚠️ Important:
The dataset **does not include aspect-level annotations**.

To enable ABSA, this project constructs a synthetic dataset:

* aspect terms are heuristically extracted
* sentiment labels are simplified into (positive / negative / neutral)

---

## ⚠️ Limitations

* No ground-truth ABSA annotations
* Synthetic labeling introduces noise
* Evaluation is not directly comparable with benchmark datasets (SemEval, VLSP)

---

## ✨ Contributions

* Built a complete ABSA pipeline (ATE + ATSC)
* Adapted raw e-commerce data into structured ABSA format
* Implemented transformer-based dual-stage architecture
* Demonstrated practical challenges of real-world NLP data

---

## 📈 Experimental Setup

* Model: `bert-base-uncased`
* Training:

  * Epochs: 3
  * Learning rate: 2e-5
* Metrics:

  * ATE: Precision / Recall / F1 (Seqeval)
  * ATSC: Accuracy / F1 (Scikit-learn)

---

## 📊 Results

⚠️ Note: Results are indicative due to synthetic dataset

| Task | Model | Metric   | Value |
| ---- | ----- | -------- | ----- |
| ATE  | BERT  | F1       | ~0.80 |
| ATSC | BERT  | Accuracy | ~0.80 |

---


## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python src/train.py
```

---

## 🧪 Notebook

See:

```
notebooks/aspect_based_sentiment_analysis_with_bert_blog_1_aio26_hautm_cs.py
```

---

## 🔬 Research Insight

This project highlights a critical gap in NLP:

> Most real-world datasets are NOT ready for structured tasks like ABSA.

Bridging this gap requires:

* data transformation
* heuristic labeling
* careful evaluation design

---

## 🚀 Future Work

* Use benchmark datasets (SemEval, VLSP)
* Improve aspect extraction with dependency parsing
* Joint learning (ATE + ATSC)
* Deploy as real-time API

---

## 👨‍💻 Author

Tran Minh Hau
