

# Vietnamese Aspect-Based Sentiment Analysis (ABSA) with PhoBERT + CRF

## 📖 Overview

This project implements a **research-grade Aspect-Based Sentiment Analysis (ABSA)** system for Vietnamese.

The pipeline decomposes the problem into:

* **Aspect Term Extraction (ATE)** → sequence labeling (BIO tagging)
* **Aspect Term Sentiment Classification (ATSC)** → sentiment classification

## 🧠 Architecture

```text
Input → PhoBERT + CRF (ATE) → Extract aspects → PhoBERT (ATSC) → Sentiment
```

## 🚀 Key Features

* Vietnamese-specific modeling using **PhoBERT**
* Structured prediction using **CRF**
* Modular and reproducible pipeline
* End-to-end inference system

## 📊 Example

Input:
"Sản phẩm rất tốt nhưng giao hàng chậm"

Output:

* sản phẩm → positive
* giao hàng → negative

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Training

```bash
python src/train.py
```

## 🧪 Notebook

Use:

```
notebooks/absa_experiment.ipynb
```

## 📊 Evaluation

* ATE: Precision / Recall / F1 (seqeval)
* ATSC: Accuracy / F1 (sklearn)

## ✨ Contributions

* Dual-stage ABSA pipeline
* PhoBERT + CRF for Vietnamese NLP
* Research-ready modular structure

## 🔮 Future Work

* Deploy API (FastAPI / Streamlit)
* Benchmark on VLSP dataset
* Multi-task learning

## 👨‍💻 Author

Tran Minh Hau
