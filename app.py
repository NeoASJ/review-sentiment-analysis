import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- Data Loading & Model Setup ---
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_data_and_model():
    try:
        # Load dataset directly from the repository
        df = pd.read_csv('IMDB Dataset.csv')
        
        # Preprocess data
        X = df['review']
        y = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Initialize and fit model
        with st.spinner("Training model (this may take a minute)..."):
            vectorizer = CountVectorizer(stop_words='english', max_features=5000)
            X_vec = vectorizer.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_vec, y)
            
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analysis",
        page_icon="🎬",
        layout="wide"
    )
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app predicts sentiment of IMDB movie reviews using:
        - **Logistic Regression** classifier
        - **CountVectorizer** for text processing
        - Trained on 50,000 IMDB reviews
        """)
    
    # Main content
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.caption("Enter a movie review to analyze its sentiment (positive/negative)")
    
    # Load model
    model, vectorizer = load_data_and_model()
    
    # User input
    user_review = st.text_area(
        "Your Review:", 
        height=150,
        placeholder="Type or paste your movie review here...",
        help="The model will analyze whether this review is positive or negative"
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review first")
            return
            
        try:
            # Make prediction
            review_vec = vectorizer.transform([user_review])
            prediction = model.predict(review_vec)[0]
            proba = model.predict_proba(review_vec)[0]
            confidence = proba[prediction]
            
            # Display result
            col1, col2 = st.columns([1, 4])
            with col1:
                if prediction == 1:
                    st.success("😊 Positive")
                else:
                    st.error("😞 Negative")
            
            with col2:
                st.progress(int(confidence * 100))
                st.caption(f"Confidence: {confidence*100:.1f}%")
            
            # Show top influential words
            st.subheader("🔍 Key Words Influencing This Prediction:")
            words = vectorizer.get_feature_names_out()
            coef = model.coef_[0]
            top_words = sorted(zip(words, coef), 
                            key=lambda x: x[1], 
                            reverse=prediction==1)[:5]
            
            cols = st.columns(5)
            for i, (word, score) in enumerate(top_words):
                with cols[i]:
                    st.metric(
                        label=word.capitalize(),
                        value=f"{'↑' if score > 0 else '↓'} {abs(score):.2f}"
                    )
            
            # Show balloons for positive reviews
            if prediction == 1:
                st.balloons()
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()

# --- Custom CSS ---
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold !important;
        text-align: center !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 16px !important;
        text-align: center !important;
    }
    .stProgress > div > div > div {
        background-image: linear-gradient(to right, #4facfe, #00f2fe);
    }
    .stButton>button {
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)