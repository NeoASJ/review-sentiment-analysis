import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# --- Data Loading & Model Setup ---
@st.cache_resource
def load_data_and_model():
    if not os.path.exists('content/IMDB Dataset.csv'):
        with zipfile.ZipFile('IMDB Dataset.csv.zip') as f:
            f.extractall('content')
    
    df = pd.read_csv('content/IMDB Dataset.csv')
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    return model, vectorizer

model, vectorizer = load_data_and_model()

# --- Streamlit UI ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment:")

user_review = st.text_area("Your Review", height=150)

if st.button("Predict Sentiment", type="primary"):
    if user_review.strip():
        review_vec = vectorizer.transform([user_review])
        prediction = model.predict(review_vec)[0]
        proba = model.predict_proba(review_vec)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        # Main result display with emoji
        if prediction == 1:
            st.markdown(f"""
            <div style="
                background-color: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #c3e6cb;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 20px;
            ">
                😊 Positive Sentiment ({confidence*100:.1f}% confidence)
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div style="
                background-color: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 20px;
            ">
                😞 Negative Sentiment ({confidence*100:.1f}% confidence)
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence bar with dynamic color
        progress_color = "#28a745" if prediction == 1 else "#dc3545"
        st.markdown(f"""
        <style>
            .stProgress > div > div > div {{
                background-color: {progress_color} !important;
            }}
        </style>
        """, unsafe_allow_html=True)
        st.progress(int(confidence * 100))
        
        # Top influential words
        words = vectorizer.get_feature_names_out()
        coef = model.coef_[0]
        top_words = sorted(zip(words, coef), 
                         key=lambda x: x[1], 
                         reverse=prediction==1)[:5]
        
        st.subheader("Key Influencing Words:")
        cols = st.columns(5)
        for i, (word, score) in enumerate(top_words):
            with cols[i]:
                st.metric(
                    label=word.capitalize(),
                    value=f"{'↑' if score > 0 else '↓'} {abs(score):.2f}"
                )
                
    else:
        st.warning("Please enter a review to analyze.")

# --- Style Improvements ---
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)