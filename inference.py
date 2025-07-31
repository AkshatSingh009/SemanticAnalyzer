import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from src.utils import ids_to_names
from collections import Counter

# ✅ Load tokenizer and model
tokenizer = joblib.load("artifacts/tokenizer.pkl")
model = load_model("artifacts/sentiment_model.h5")

# ✅ Setup NLTK only when needed
@st.cache_resource
def setup_nltk():
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    return word_tokenize, WordNetLemmatizer(), Counter(stopwords.words())

# ✅ Preprocess a single tweet
def preprocess_single_tweet(tweet):
    word_tokenize, lemmatiser, stop_english = setup_nltk()
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    cleaned = [lemmatiser.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_english]
    return " ".join(cleaned)

# ✅ Predict sentiment
def predict_sentiment(tweet):
    cleaned = preprocess_single_tweet(tweet)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    prediction = model.predict(padded)[0]
    predicted_index = np.argmax(prediction)
    return ids_to_names([predicted_index])[0]

# ✅ Streamlit UI
st.set_page_config(page_title="Social Media Sentiment Predictor", layout="centered")
st.title("🧠 Social Media Sentiment Predictor")
st.write("Enter a tweet below to predict its sentiment:")

tweet = st.text_area("Tweet", height=150)

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        sentiment = predict_sentiment(tweet)
        st.success(f"Predicted Sentiment: **{sentiment}**")
