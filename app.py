import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data (extract if needed)
if not os.path.exists('c00ntent/IMDB Dataset.csv'):
    with zipfile.ZipFile('IMDB Dataset.csv.zip') as f:
        f.extractall('c00ntent')

df = pd.read_csv('c00ntent/IMDB Dataset.csv')

# Preprocessing (simplified)
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Model training (or load a pre-trained model if available)
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer for re-use (optional)
# joblib.dump(model, 'imdb_model.pkl')
# joblib.dump(vectorizer, 'imdb_vectorizer.pkl')

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment:")

user_review = st.text_area("Your Review")

if st.button("Predict Sentiment"):
    if user_review.strip():
        review_vec = vectorizer.transform([user_review])
        prediction = model.predict(review_vec)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")
