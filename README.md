# 🛒 AmazonSentiment-BERTLSTM  
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)  
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)](https://en.wikipedia.org/wiki/Sentiment_analysis)  
[![Models](https://img.shields.io/badge/Models-LSTM%20%7C%20BERT%20%7C%20BitNet-orange)](https://huggingface.co/)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Benchmarking LSTM, BERT, and BitLinear BERT for large-scale sentiment analysis on Amazon product reviews.

---

## 📘 Overview

This project compares **three deep learning models** — **LSTM**, **BERT**, and **BitLinear BERT** — for sentiment analysis on the **Massive Amazon Reviews dataset (14M reviews)**.  
It highlights the trade-offs between accuracy, efficiency, and resource usage, exploring how quantized transformer layers can maintain performance while improving computational efficiency.

> **Goal:** Evaluate performance, robustness, and scalability of modern NLP architectures for large-scale sentiment classification.

---

## ⚙️ Installation & Run

### 🧩 1. Clone the Repository
```bash
git clone https://github.com/HEsiyun/AmazonSentiment-BERTLSTM.git
cd AmazonSentiment-BERTLSTM
```

### 🧠 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### 📦 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🚀 4. Run the Models
#### Run LSTM model
```bash
python lstm_model.py
```

#### Run BERT fine-tuning
```bash
python bert_model.py
```

#### Run BitLinear BERT
```bash
python bitlinear_bert.py
```

### 📊 5. Visualize Results
```bash
python visualize.py
```
Generates confusion matrices, ROC curves, and performance summaries under `results/`.

> 💡 Tip: If running on limited hardware, reduce batch size (e.g. 8–16) and set `max_length` to 100 in the tokenizer.

---

## 🧠 Models Compared

### 1️⃣ LSTM
- Implemented with **TensorFlow/Keras**.  
- Handles sequential dependencies efficiently using embedding + LSTM layers.  
- Three variants were tested:
  - **Single-layer LSTM**  
  - **Multi-layer LSTM**  
  - **Bidirectional LSTM**  
- Best performance from **Bidirectional LSTM (input dim 10,000, dropout 0.2, lr=5e-5)** achieved:
  - **Accuracy:** 0.84
  - **Precision:** 0.83
  - **Recall:** 0.85
  - **AUC:** 0.92

> LSTM remains strong for sequential data but is sensitive to hyperparameters and batch sizes.

### 2️⃣ BERT
- Based on **Hugging Face’s bert-base-cased**.
- Trained from scratch with a **custom classification head** for binary sentiment prediction.  
- Uses **AdamW optimizer** and **early stopping** to prevent overfitting.  
- Token length capped at 150 tokens for efficiency.
- Best configuration (batch=32, lr=3e-5, 3 epochs):
  - **Accuracy:** 0.9068
  - **Precision:** 0.91
  - **Recall:** 0.91
  - **AUC:** 0.97

> BERT achieved the **highest performance** across all metrics, proving robust to hyperparameter variation.

### 3️⃣ BitLinear BERT (BitNet Quantized)
- Applies **1-bit quantization** via **BitLinear layers** to BERT’s dense and attention components.  
- Reduces computational and memory overhead without major accuracy loss.  
- Best configuration (batch=64, lr=2e-5, 3 epochs):
  - **Accuracy:** 0.836
  - **Precision:** 0.84
  - **Recall:** 0.84
  - **AUC:** 0.92

> A **resource-efficient** BERT variant suitable for constrained environments — trades a small accuracy drop for major efficiency gains.

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | AUC | Notes |
|:------|:----------|:-----------|:--------|:-----|:------|
| **BERT** | **0.9068** | **0.91** | **0.91** | **0.97** | Best overall performer |
| **LSTM (Bi)** | 0.84 | 0.83 | 0.85 | 0.92 | Strong sequential model |
| **BitLinear BERT** | 0.836 | 0.84 | 0.84 | 0.92 | Efficient quantized variant |

> BERT consistently achieved the highest accuracy and AUC, while BitLinear BERT offered a promising trade-off between efficiency and performance.

---

## 🧩 Dataset

- **Source:** [Massive Amazon Reviews Collection (14M lines)](https://www.kaggle.com/datasets/zakariaolamine/massive-amazon-reviews-collection-14m-lines)  
- **Preprocessing Steps:**
  - Random sampling (0.5% of data ≈ 70k reviews).  
  - Cleaning: removed HTML tags, URLs, and non-alphabetic tokens.  
  - Undersampling for class balance (≈21k positive, 21k negative).  
  - Sequence padding/truncation to 100–150 tokens.  
- **Visualizations:**
  - Class distributions before/after balancing.  
  - Word clouds pre/post cleaning.  
  - Sentence length histogram and vocabulary coverage plots.

---

## 🧰 Tech Stack

- **Python 3.10**  
- **TensorFlow / Keras** – LSTM Implementation  
- **PyTorch + Hugging Face Transformers** – BERT & BitLinear BERT  
- **BitNet Library (Kye Gomez)** – 1-bit quantization layer replacement  
- **Pandas / Matplotlib / WordCloud** – Data visualization and preprocessing  

---

## 🧾 Project Structure

```
AmazonSentiment-BERTLSTM/
├── data/                          # Preprocessed samples from Kaggle dataset
├── preprocessing.py               # Cleaning and balancing scripts
├── lstm_model.py                  # LSTM implementation
├── bert_model.py                  # BERT fine-tuning pipeline
├── bitlinear_bert.py              # BitNet quantized BERT model
├── train_utils.py                 # Training and evaluation functions
├── visualize.py                   # Word clouds, confusion matrices, ROC plots
└── report/                        # Final report and appendix
```

---

## 🎓 Author

**Siyun He**  
Khoury College of Computer Sciences, Northeastern University  
📧 he.siyun@northeastern.edu  
🌐 [GitHub: HEsiyun](https://github.com/HEsiyun)

---

## 💬 Acknowledgments

- **Professor Uzair Ahmad** – Course Instructor, CS6120 Natural Language Processing  
- **Venelin Valkov** – BERT fine-tuning tutorial  
- **Jason Brownlee** – LSTM sequence classification guide  
- **Kye Gomez** – Creator of BitNet quantization library  

---

⭐ **If this project helps you, please give it a star!**

