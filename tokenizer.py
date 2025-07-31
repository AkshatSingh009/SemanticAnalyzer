from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(texts, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def tokenize_and_pad(texts, tokenizer, max_len=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, truncating='post', padding='post', maxlen=max_len)
