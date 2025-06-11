# 📊 Sentiment Analysis Using Naive Bayes

This project is a simple Sentiment Analysis system that classifies movie reviews as **positive** or **negative** using Python, NLTK, and Scikit-learn. It uses a Naive Bayes classifier trained on the NLTK movie reviews dataset.

---

## 🧠 What is Sentiment Analysis?

Sentiment analysis is a Natural Language Processing (NLP) technique used to detect emotions or opinions in text data. It is commonly used in applications like product reviews, social media monitoring, and customer feedback systems.

---

## 🚀 Features

- Binary sentiment classification: Positive or Negative
- Preprocessing and cleaning of text data
- TF-IDF vectorization
- Naive Bayes classification model
- Custom text input for testing
- Accuracy and classification report

---

## 📁 Project Structure

# intallations:
pip install nltk scikit-learn pandas numpy

# Output:
📝 Review: I absolutely loved the movie, it was amazing and well-acted.
Predicted Sentiment: Positive



🔍 Model Evaluation:
Accuracy: 0.84
Classification Report:
              precision    recall  f1-score   support

    negative       0.85      0.82      0.83       100
    positive       0.83      0.86      0.85       100

    accuracy                           0.84       200
   macro avg       0.84      0.84      0.84       200
weighted avg       0.84      0.84      0.84       200
