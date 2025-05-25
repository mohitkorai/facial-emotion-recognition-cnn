# facial-emotion-recognition-cnn
Convolutional Neural Network to classify facial emotions using the FER-2013 dataset. Built with TensorFlow/Keras. 

# facial-emotion-recognition-cnn
Facial Emotion Classification using CNN and FER-2013 dataset — Achieved 82% accuracy using deep learning with class balancing, normalization, and Keras model tuning.

# Facial Emotion Recognition - CNN Project

This project applies a Convolutional Neural Network (CNN) to classify facial expressions using the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset. The dataset contains 48×48 grayscale images representing 7 different emotions.

🎯 **Final Accuracy:** 82%  
🧠 **Model Type:** Deep CNN with BatchNorm, Dropout, and Softmax output

---

## 🔍 Problem Statement

The goal is to automatically detect and classify human facial emotions from grayscale images. Given a dataset of cropped facial regions labeled with emotions, we train a deep CNN model to identify:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📂 Project Overview

This notebook demonstrates an end-to-end deep learning pipeline:

- **Data Preprocessing**
  - Loaded and reshaped `pixels` column into image arrays
  - Normalized pixel values to [0, 1]
  - Balanced the dataset using `RandomOverSampler`

- **Model Architecture**
  - 5 convolutional layers with ReLU activations
  - Batch Normalization and MaxPooling
  - Fully connected dense layer with dropout
  - Softmax output layer for 7-class prediction

- **Training**
  - Optimized with `Adam` (learning rate = 0.0001)
  - Early stopping and learning rate scheduling via callbacks
  - Trained for 30 epochs with validation

- **Evaluation**
  - Accuracy and loss curves
  - Classification report (precision, recall, F1-score)
  - Confusion matrix heatmap

---

## 📊 Results Summary

| Metric       | Value |
|--------------|-------|
| Accuracy     | 82%   |
| Classes      | 7     |
| Input Shape  | 48×48 grayscale |

---

## 🧰 Tools & Libraries

- Python (NumPy, Pandas)
- TensorFlow / Keras (CNN model)
- scikit-learn (metrics, train-test split)
- imbalanced-learn (oversampling)
- Matplotlib / seaborn (visualizations)

---

## 📁 Repository Contents

- `facial_emotion_cnn.ipynb`: Complete notebook including data loading, model training, and evaluation
- `README.md`: Project overview and results

---

## 📌 Author

**Mohit Venkata Rama Swamy Korai**  
Machine Learning & Data Science Enthusiast  
[GitHub](https://github.com/mohitkorai) • [Kaggle](https://www.kaggle.com/mohitkorai) • [LinkedIn](https://www.linkedin.com/in/venkatasw/)
