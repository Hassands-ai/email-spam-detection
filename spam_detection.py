import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 2: Convert labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Convert text to numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Step 4: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predict on test set
y_pred = model.predict(X_test)

# Step 7: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Test with your own message
sample_message = ["Congratulations! You have won a $1000 Walmart gift card. Call now!"]
sample_vector = vectorizer.transform(sample_message)
prediction = model.predict(sample_vector)
print("\nSample message prediction:", "Spam" if prediction[0] == 1 else "Ham")
import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import nltk

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------- Step 1: Load dataset ----------
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# ---------- Step 2: Text cleaning function ----------
def clean_text(text):
    text = text.lower()  # lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

df['message'] = df['message'].apply(clean_text)

# ---------- Step 3: Convert labels to binary ----------
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------- Step 4: Vectorization using TF-IDF ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# ---------- Step 5: Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Step 6: Train model ----------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- Step 7: Evaluation ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- Step 8: Save model and vectorizer ----------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved!")

# ---------- Step 9: Test with a new message ----------
def predict_message(message):
    message = clean_text(message)
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example test
print("\nSample message prediction:", predict_message("Win a free iPhone now!"))
import nltk
nltk.download('stopwords')
exit()

