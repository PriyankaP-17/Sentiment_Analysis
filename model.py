
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  # For better plots


nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv('sentiment_analysis_dataset.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    try:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    except Exception as e:
        print(f"Error processing text: {text}")
        print(e)
        return []

df['Tokens'] = df['Review'].apply(clean_text)

# Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Visualization - Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualization - Metrics Bar Plot
metrics_df = pd.DataFrame(report).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8,5), color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Precision, Recall, F1-score per Class')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Comparison Table
print("\nActual vs Predicted Sentiments:")
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(comparison_df.head())

