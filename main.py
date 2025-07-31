from src.data_loader import load_data
from src.preprocessing import preprocess_tweets
from src.tokenizer import create_tokenizer, tokenize_and_pad
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import names_to_ids, ids_to_names

import os
import joblib
import mlflow
import mlflow.tensorflow
import numpy as np


def main():
    print("Loading data...")
    train_df, test_df = load_data("data/twitter_training.csv", "data/twitter_validation.csv")

    print("Converting sentiment labels...")
    train_df["Sentiment"] = names_to_ids(train_df["Sentiment"])
    test_df["Sentiment"] = names_to_ids(test_df["Sentiment"])

    print("Preprocessing tweets...")
    train_df = preprocess_tweets(train_df)
    test_df = preprocess_tweets(test_df)

    print("Creating tokenizer...")
    tokenizer = create_tokenizer(train_df["Tweet_Content_Split"])

    print("Tokenizing and padding sequences...")
    X_train = tokenize_and_pad(train_df["Tweet_Content_Split"], tokenizer)
    X_test = tokenize_and_pad(test_df["Tweet_Content_Split"], tokenizer)

    y_train = train_df["Sentiment"]
    y_test = test_df["Sentiment"]

    print("Building model...")
    model = build_model()

    print("Training model...")
    train_model(model, X_train, y_train, X_test, y_test)

    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test, ids_to_names)

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    print("Saving model and tokenizer...")
    model_path = "artifacts/sentiment_model.h5"
    tokenizer_path = "artifacts/tokenizer.pkl"
    model.save(model_path)
    joblib.dump(tokenizer, tokenizer_path)

    # MLflow tracking setup
    mlflow.set_tracking_uri("file:///C:/Users/goyal/OneDrive/Desktop/Social-Media-Sentiment-Analysis-main/mlruns")
    mlflow.set_experiment("Sentiment-Analysis-Experiment")

    print("Logging to MLflow...")
    with mlflow.start_run():
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_metric("accuracy", accuracy)

        # Prepare an input example to avoid warnings
        input_example = X_test[:1]

        # Log the TensorFlow model with input_example
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        # Log artifacts if available
        heatmap_path = "artifacts/heatmap.png"
        if os.path.exists(heatmap_path):
            mlflow.log_artifact(heatmap_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(tokenizer_path)


if __name__ == "__main__":
    main()
