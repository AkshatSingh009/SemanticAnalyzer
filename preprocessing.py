import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatiser = WordNetLemmatizer()
stop_english = Counter(stopwords.words('english'))

def remove_stopwords(tokens):
    return " ".join([lemmatiser.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_english])

def preprocess_tweets(df):
    df["Tweet_Content_Split"] = df["Tweet_Content"].apply(word_tokenize)
    df["Tweet_Content_Split"] = df["Tweet_Content_Split"].apply(remove_stopwords)
    return df
