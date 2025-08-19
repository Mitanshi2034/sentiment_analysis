import streamlit as st
import numpy as np
import pickle
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Models ---

# 1. XGBoost + TF-IDF
xgb_model = joblib.load("models/xgboost_sentiment_model.pkl")
tokenizer = joblib.load("models/tfidf_vectorizer.pkl")

# 2. RoBERTa using Transformers (Fixes meta tensor issue)
roberta_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
roberta_model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")
roberta = pipeline(
    "sentiment-analysis",
    model=roberta_model,
    tokenizer=roberta_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# 3. VADER
vader_analyzer = SentimentIntensityAnalyzer()

# --- Ensemble Weights ---
weights = {
    "xgb": 0.4,
    "roberta": 0.4,
    "vader": 0.2
}

# --- Preprocess input ---
def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# --- Individual model predictions ---
def get_xgb_pred(text):
    vec = tokenizer.transform([text])
    prob = xgb_model.predict_proba(vec)[0][1]
    return prob

def get_roberta_pred(text):
    result = roberta(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']

def get_vader_pred(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    return (score + 1) / 2  # from -1:1 to 0:1

# --- Weighted ensemble prediction ---
def get_final_prediction(text):
    text_clean = clean_text(text)
    xgb_p = get_xgb_pred(text_clean)
    roberta_p = get_roberta_pred(text_clean)
    vader_p = get_vader_pred(text_clean)

    final_score = (
        weights["xgb"] * xgb_p +
        weights["roberta"] * roberta_p +
        weights["vader"] * vader_p
    )
    label = "Positive 😊" if final_score >= 0.5 else "Negative 😠"
    return label, final_score, xgb_p, roberta_p, vader_p

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Sentiment Ensemble App", page_icon="🧠")
st.title("ReviewBot 🤖📝 – Your futuristic sentiment analysis assistant")
st.markdown("**Using:** XGBoost + RoBERTa + VADER (Weighted Voting)")

st.markdown("---")
review = st.text_area("✍️ Enter a product review", height=150)

if st.button("🔍 Analyze Review"):
    if not review.strip():
        st.warning("Please write a review first.")
    else:
        label, final_score, xgb_p, roberta_p, vader_p = get_final_prediction(review)

        # Display label
        st.markdown(f"### ✅ Predicted Sentiment: **{label}**")
        st.markdown(f"### 📊 Confidence Score: `{final_score*100:.2f}%`")

        # Confidence Gauge using bar
        st.markdown("#### 🎯 Confidence Meter")
        fig, ax = plt.subplots(figsize=(6, 1.2))
        sns.barplot(x=[final_score*100], y=[""], palette=["green" if final_score > 0.5 else "red"], ax=ax)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.set_yticks([])
        st.pyplot(fig)

        # Model-wise table
        st.markdown("#### 📋 Model-wise Confidence Scores")
        st.dataframe({
            "Model": ["XGBoost", "RoBERTa", "VADER"],
            "Confidence (Positive)": [f"{xgb_p:.2f}", f"{roberta_p:.2f}", f"{vader_p:.2f}"],
            "Weight": [weights["xgb"], weights["roberta"], weights["vader"]]
        })

        # Summary info
        st.info("✅ Weighted ensemble automatically applies model confidences with their weights to give a final decision.")

        # Footer
        st.markdown("---")
        st.markdown("<center><sub>Powered by Machine Learning ✨</sub></center>", unsafe_allow_html=True)
