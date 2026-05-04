# 🎵 Music Genre Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c.svg)](https://pytorch.org/)

## 📌 Project Overview
The objective of this project is to design a machine learning system that automatically classifies an input audio track into its correct music genre. Given a short music clip, the model extracts meaningful audio features and predicts the most probable genre (e.g., classical, jazz, pop, rock, metal, etc.) with high accuracy.

## 📊 Dataset
This project uses the famous **[GTZAN Dataset – Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)**. 
The dataset consists of 1,000 audio tracks, each 30 seconds long. It contains 10 genres, each represented by 100 tracks: *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.*

## 🧠 Technical Architecture
To achieve high accuracy and robustness, this project implements a **Dual-Model Pipeline** to compare feature extraction techniques:

1. **YAMNet Pipeline (TensorFlow):**
   * Uses Google's pre-trained YAMNet audio event classifier to extract deep acoustic embeddings from 16kHz audio.
   * Feeds embeddings into a custom classification head (saved as `.h5` for cross-platform stability).

2. **MERT Pipeline (PyTorch / Hugging Face):**
   * Utilizes **MERT** (Music Extraction Representation Toolkit), a state-of-the-art acoustic foundation model specifically fine-tuned for music understanding.
   * Processes 24kHz audio through the transformer layers to generate rich musical embeddings.
   * Classifies genres using an optimized Multi-Layer Perceptron (MLP) trained on the extracted features.

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/Music-Genre-Classification.git](https://github.com/your-username/Music-Genre-Classification.git)
cd Music-Genre-Classification