# File: train_model.py
import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create folder structure
os.makedirs("app/model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset (copy this file manually)
data_path = "data/SMSSpamCollection.txt"
df = pd.read_csv(data_path, sep="\t", names=["label", "text"])

# Map labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text preprocessing
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean)

df["clean_text"] = df["text"].apply(preprocess)

# TF-IDF Vectorization
TF_IDF = TfidfVectorizer(max_features=200)
X = TF_IDF.fit_transform(df["clean_text"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "app/model/spam_classifier_model.pkl")
joblib.dump(TF_IDF, "app/model/tfidf_vectorizer.pkl")

print("Training complete!")
print("Model and vectorizer saved to app/model/")
