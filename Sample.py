import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample data (You can replace this with your dataset)
data = [
    ("I love this movie, it was fantastic!", "positive"),
    ("What a terrible movie. Waste of time.", "negative"),
    ("Absolutely brilliant! Best I've seen.", "positive"),
    ("Horrible. I hated every moment.", "negative"),
    ("It was okay, not great but not bad.", "neutral"),
    ("The plot was boring and predictable.", "negative"),
    ("Such an inspiring and beautiful story.", "positive"),
]

# Split data into texts and labels
texts, labels = zip(*data)

# Preprocess and vectorize text
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(texts)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Try it with new input
new_text = ["I really enjoyed the performance!"]
new_text_vector = vectorizer.transform(new_text)
prediction = model.predict(new_text_vector)
print("Sentiment:", prediction[0])
