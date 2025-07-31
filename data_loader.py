import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"]).dropna()
    test_df = pd.read_csv(test_path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"]).dropna()
    return train_df, test_df
