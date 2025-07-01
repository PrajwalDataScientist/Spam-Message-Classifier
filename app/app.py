# ✅ File: app/app.py

import streamlit as st
import joblib
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# 📦 Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load model and vectorizer
model = joblib.load("app/model/spam_classifier_model.pkl")
vectorizer = joblib.load("app/model/tfidf_vectorizer.pkl")

# 🧹 Preprocessing function
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean)

# 🎨 Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .stApp {
        background-color: #e8f0fe;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        color: #0b5394;
        text-align: center;
        font-size: 2.5rem;
    }
    .footer {
        font-size: 0.9rem;
        text-align: center;
        color: #444;
        margin-top: 30px;
    }
    a {
        color: #0b5394;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.markdown("<h1 class='title'>📩 Spam Message Classifier</h1>", unsafe_allow_html=True)
st.markdown("Use Machine Learning to classify whether a message is **Spam** or **Ham** (Not Spam).")

# 🔤 Input
user_input = st.text_area("✉️ Enter a message to classify:", height=150, value=st.session_state.get("user_input", ""))

# 🔍 Predict Button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        clean_text = preprocess(user_input)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown(f"<div style='background-color:#ffccd5;padding:15px;border-radius:8px'><h4>🚨 Spam Detected!</h4></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#d1f0d1;padding:15px;border-radius:8px'><h4>✅ Ham Message</h4></div>", unsafe_allow_html=True)

# 💬 Sample Messages
st.markdown("---")
st.markdown("### 💬 Try Sample Messages")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ Ham Examples")
    if st.button("📨 Hey, are we still meeting for lunch today?"):
        st.session_state["user_input"] = "Hey, are we still meeting for lunch today?"
    if st.button("📨 Can you send me the project files by 5 PM?"):
        st.session_state["user_input"] = "Can you send me the project files by 5 PM?"

with col2:
    st.markdown("#### 🚨 Spam Examples")
    if st.button("💸 Congratulations! You've won a $1000 gift card. Click here to claim now!"):
        st.session_state["user_input"] = "Congratulations! You've won a $1000 gift card. Click here to claim now!"
    if st.button("📞 Free entry in a contest! Text WIN to 80086 now!"):
        st.session_state["user_input"] = "Free entry in a contest! Text WIN to 80086 now!"

# 👣 Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    👨‍💻 Built with ❤️ by <a href='https://www.linkedin.com/in/prajwal10DS/' target='_blank'>Prajwal — Data Scientist</a><br><br>
    🔗 <a href='https://github.com/PrajwalDataAnalyst/Spam-Message-Classifier' target='_blank'>GitHub</a> |
    🐳 <a href='https://hub.docker.com/repositories/prajwal1027' target='_blank'>DockerHub</a> |
    💼 <a href='https://www.linkedin.com/in/prajwal10DS/' target='_blank'>LinkedIn</a><br><br>
    🤖 Logistic Regression + TF-IDF + Streamlit + NLTK
</div>
""", unsafe_allow_html=True)
