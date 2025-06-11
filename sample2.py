
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load custom dataset (Replace with the actual path to your dataset)
df = pd.read_csv('sentiment_analysis_dataset.csv')

# Step 2: Data Preprocessing
stop_words = set(stopwords.words('english'))  # Set of English stopwords

# Tokenization and cleaning the text
def clean_text(text):
    try:
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        return tokens
    except Exception as e:
        print(f"Error processing text: {text}")
        print(e)
        return []

df['Tokens'] = df['Review'].apply(clean_text)  # Apply the cleaning function 

# Step 3: Feature Extraction (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Review'])  # Convert text into numerical features

# Sentiment column is already in binary form (0 or 1)
y = df['Sentiment']  # Labels

# Step 4: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model using Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)  # Train the model

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))  # Accuracy score
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Detailed report

# Step 8: Print binary output for predictions
print("\nBinary Predictions (1 = Positive, 0 = Negative):")
print(y_pred)

# You can also print the actual vs predicted values for a comparison
print("\nActual vs Predicted Sentiments:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)