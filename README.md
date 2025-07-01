# 📩 Spam Message Classifier 🚀

A machine learning-powered **NLP spam classifier web app** built using **TF-IDF**, **Logistic Regression**, and **Streamlit**. The app detects whether a given message is **Spam** or **Ham** in real time.

---

## 🔍 Project Summary

This project uses Natural Language Processing techniques and a supervised learning model to classify SMS/text messages as spam or ham (not spam).

### 💡 What We Implemented:
- Text Preprocessing using **NLTK**:
  - Lowercasing
  - Removing special characters
  - Tokenization
  - Stopword removal
  - Lemmatization
- Feature Extraction:
  - **TF-IDF Vectorization**
- Model:
  - **Logistic Regression**
- Deployment:
  - Web app with **Streamlit**
  - Containerized with **Docker**

---

## ✅ Model Performance

| Model              | Accuracy  |
|-------------------|-----------|
| Logistic Regression (TF-IDF) | ✅ **97.1% (Test)** |
| Naive Bayes (TF-IDF)         | 96.8%              |

---

## 🗂️ Folder Structure

spam_classifier/
│
├── app/
│ ├── app.py # Streamlit UI
│ └── model/
│ ├── spam_classifier_model.pkl # Trained model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── data/
│ └── SMSSpamCollection.txt # Original dataset
│
├── notebooks/
│ └── spam_model_dev.ipynb # Model development & training notebook
│
├── preprocessing.py # Optional preprocessing module
├── requirements.txt # Required Python packages
└── README.md # You are here


---

## 🛠️ How to Run Locally (VS Code)

### 1. 🔁 Clone this repo
```bash
git clone https://github.com/PrajwalDataAnalyst/Spam-Message-Classifier.git
cd Spam-Message-Classifier

2. 📦 Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate


3. 📥 Install dependencies
pip install -r requirements.txt

4. ▶️ Run Streamlit App
streamlit run app/app.py

🐳 Run Using Docker
1. 📥 Pull the image from Docker Hub
docker pull prajwal1027/spam-classifier:latest

2. ▶️ Run the container
docker run -p 8501:8501 prajwal1027/spam-classifier

Then open your browser and go to:
http://localhost:****

👨‍💻 Author
Prajwal — Data Scientist
