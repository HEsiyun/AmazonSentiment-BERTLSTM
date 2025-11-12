# 🧠 Sentiment Analysis using LSTM, BERT, and BitLinear BERT

> Exploring deep learning architectures for large-scale sentiment classification on Amazon reviews.

---

## 📘 Overview

This project investigates three NLP architectures — **LSTM**, **BERT**, and **BitLinear BERT** — for **sentiment analysis** on a massive Amazon reviews dataset (14M+ entries).  
It benchmarks model performance, efficiency, and scalability, providing insights into how modern transformer models compare with classical recurrent networks under different resource constraints.

> **Goal:** Identify which architecture best captures sentiment patterns in large-scale review data, balancing accuracy and computational cost.

---

## 🧩 Models Implemented

### 1. LSTM (Long Short-Term Memory)
- Built with TensorFlow / Keras.
- Explores single-layer, multi-layer, and bidirectional variants.
- Hyperparameters tuned for input dimension, dropout, and batch size.
- Achieved **84% test accuracy**, with **AUC = 0.92** on balanced review data.

### 2. BERT (Transformer-based Model)
- Implemented using `Hugging Face Transformers` (`bert-base-cased`).
- Fine-tuned from scratch for binary sentiment classification.
- Trained with **AdamW optimizer**, 3 epochs, learning rates 2e-5 ~ 5e-5.
- Achieved **90.7% test accuracy**, **precision = recall = 0.91**, **AUC = 0.97** — the best performer. 
### 3. BitLinear BERT (1-bit Quantized BERT)
- Leverages **BitNet’s 1-bit quantization** via open-source `BitNet` library by Kye Gomez.
- Replaces linear layers with **BitLinear** for reduced computation and memory.
- Reaches **~84% accuracy**, on par with LSTM but at a fraction of the resource cost.  
  > Demonstrates efficiency potential for LLMs in low-resource environments. 

---

## 🧹 Data Preparation

- Dataset: [Amazon Reviews (14M lines)](https://www.kaggle.com/datasets/zakariaolamine/massive-amazon-reviews-collection-14m-lines)
- Sampled **0.5% subset (~43k reviews)** for local experiments.
- Balanced classes via **undersampling** (21,685 positive / 21,685 negative).  
- Text cleaning:
  - Removed HTML, URLs, non-alphabetic characters.
  - Tokenized and padded to consistent length (100 for LSTM, 150 for BERT).
---

## 📊 Results Summary

| Model | Test Accuracy | Precision | Recall | AUC |
|:------|:--------------:|:----------:|:--------:|:----:|
| **LSTM (Bidirectional)** | 0.84 | 0.83 | 0.85 | 0.92 |
| **BERT (Base)** | **0.91** | **0.91** | **0.91** | **0.97** |
| **BitLinear BERT** | 0.84 | 0.84 | 0.84 | 0.92 |

> 🔍 *BERT clearly leads in all metrics, but BitLinear BERT shows promise for efficient deployment on limited hardware.*

---

## 💬 Discussion

- **BERT** excels in contextual understanding and robustness, requiring minimal hyperparameter tuning.
- **LSTM** remains viable for lightweight setups, though sensitive to learning rate and dropout adjustments.
- **BitLinear BERT** proves that **quantization** can substantially reduce memory and compute needs without drastic performance loss.  
  A promising direction for edge deployment and on-device inference.

---

## 🧠 Key Takeaways

- Transformer models dominate large-scale sentiment analysis tasks.
- Quantized models offer practical trade-offs for constrained systems.
- Data preprocessing and balanced sampling are critical for fair evaluation.

---

## ⚙️ Tech Stack

- Python 3.10  
- TensorFlow / Keras  
- PyTorch + Hugging Face Transformers  
- BitNet (1-bit quantization)  
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn  

---

## 📂 Project Structure
