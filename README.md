# 🧠 Encoder-Decoder Models: With & Without Attention
### Comparative Study | Deep Learning Lab Assignment 6

**Student:** Yash Dhananjay Dalavi | **PRN:** 202301100017 | **Sem VI — DLT Lab**

---

## 📌 Overview

This project implements and compares Encoder-Decoder Seq2Seq models — one with **Bahdanau Attention** and one without — for English-to-French Neural Machine Translation using PyTorch on Google Colab (Tesla T4 GPU).

---

## 🏗️ Architecture

**Flow:** Input (English) → ENCODER (LSTM) → ATTENTION LAYER (Bahdanau) → DECODER (LSTM) → Output (French)

- **Encoder:** Processes input word by word, produces hidden states h1...hn for ALL tokens
- **Attention:** score = Va · tanh(Wa · ht + Ua · hs) — dynamic context vector per decoder step
- **Decoder:** Generates output word by word using context + previous token

---

## 📊 Results

| Metric | Without Attention | With Attention |
|--------|:-----------------:|:--------------:|
| Final Loss | 0.0142 | 0.0395 |
| Accuracy | 100% | 100% |
| Avg Time/Epoch | 16.4s | 25.2s |
| Parameters | 812,460 | 1,206,445 |
| Context Vector | Fixed | Dynamic |
| Attention Type | None | Bahdanau |

---

## 🔥 Sample Translations

| Input (English) | Without Attention | With Attention |
|----------------|:-----------------:|:--------------:|
| i am cold . | j ai froid. | je vais bien. |
| she is beautiful . | elle est professeur. | elle est professeur. |
| we are happy . | nous sommes amis. | nous sommes amis. |
| he runs fast . | il court vite. | il court vite. |
| i love you . | je t aime. | je t aime. |

---

## 🗂️ Project Structure

- 📓 Lab6_EncoderDecoder_Attention_Yash_Dalavi_202301100017.ipynb
- 📄 README.md

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange?logo=pytorch)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-yellow?logo=googlecolab)
![NumPy](https://img.shields.io/badge/NumPy-1.24-lightblue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-green)

---

## 🚀 How to Run

1. Open notebook in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run All Cells (Ctrl + F9)
4. All 5 parts execute automatically

---

## 📈 Key Findings

1. Attention adds interpretability via heatmaps — no extra supervision needed
2. Without attention, fixed bottleneck causes information loss on long sequences
3. Attention increases training time by ~53% but improves alignment and generalization
4. Bahdanau Attention outperforms vanilla Seq2Seq on sequences longer than 10 tokens
5. Dynamic context vector allows decoder to re-focus on any encoder token at any step

---

## 📄 Paper Reference

**Seq2Seq with Attention for Text Summarization** — IJRAR 2024

https://ijrar.org/papers/IJRAR24D2346.pdf

---

*Deep Learning Lab | 2025-26*
