import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.url_data import URLData  # Please ensure URLData.extract_features() returns a dict including 'url'

# Define paths for cached files
NUMERIC_FEATURES_ORDER_PATH = os.path.join(os.path.dirname(__file__), "../cache/numeric_features_order.pkl")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "../cache/cached_features.pkl")

def find_newest_artifacts_dir(base_dir: str) -> str:
    """
    Find the newest created/modified subdirectory under base_dir (based on mtime) and return its path.
    """
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"{base_dir} does not exist or is not a directory.")
    subdirs = [entry for entry in os.scandir(base_dir) if entry.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_dir}")
    subdirs.sort(key=lambda d: d.stat().st_mtime_ns, reverse=True)
    return subdirs[0].path

def load_artifacts_and_model():
    """
    Find the latest training output directory and load model.joblib, vectorizer.pkl, and encoder.pkl.
    Returns: (model, vectorizer, encoder)
    """
    base_dir = os.path.join(os.path.dirname(__file__), "../result_model")
    newest_dir = find_newest_artifacts_dir(base_dir)
    print(f"Latest training output directory found: {newest_dir}")

    model_path = os.path.join(newest_dir, "model.joblib")
    vectorizer_path = os.path.join(newest_dir, "vectorizer.pkl")
    encoder_path = os.path.join(newest_dir, "encoder.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model.joblib not found in {newest_dir}")
    if not os.path.isfile(vectorizer_path):
        raise FileNotFoundError(f"vectorizer.pkl not found in {newest_dir}")
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"encoder.pkl not found in {newest_dir}")

    print(f"Loading model file: {model_path}")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")

    with open(vectorizer_path, "rb") as fv:
        vectorizer = pickle.load(fv)
    print("✅ Vectorizer loaded successfully!")

    with open(encoder_path, "rb") as fe:
        encoder = pickle.load(fe)
    print("✅ Encoder loaded successfully!")

    return model, vectorizer, encoder

def transform_features(features, vectorizer, encoder, numeric_feature_names):
    """
    Transform the feature dictionary (from URLData.extract_features()) into model input:
      - For 'ngram_hostname': join the list into a string and transform with the TF-IDF vectorizer.
      - For 'tld': transform with the One-Hot encoder.
      - For numeric features: extract them in the order saved during training.
      - Finally, horizontally stack all features into a numpy array.
    """
    # Text feature transformation
    ngram_text = " ".join(features.get("ngram_hostname", []))
    X_ngram = vectorizer.transform([ngram_text])
    if hasattr(X_ngram, "toarray"):
        X_ngram = X_ngram.toarray().astype(np.float64)

    # tld feature transformation
    tld_value = str(features.get("tld", ""))
    X_tld = encoder.transform([[tld_value]])
    if hasattr(X_tld, "toarray"):
        X_tld = X_tld.toarray().astype(np.float64)

    # Numeric features: extract using the saved order
    X_numeric = np.array([[features[name] for name in numeric_feature_names]], dtype=np.float64)

    X_final = np.hstack((X_numeric, X_ngram, X_tld))
    return X_final

def backtest_model(df, model, vectorizer, encoder, numeric_feature_names):
    """
    Iterate over the dataset, predict with the model, and collect predictions and true labels.
    Returns (predictions, true_labels, urls)
    """
    predictions = []
    true_labels = []
    urls = []

    for idx, row in df.iterrows():
        features = row.to_dict()
        try:
            X_input = transform_features(features, vectorizer, encoder, numeric_feature_names)
            pred = model.predict(X_input)[0]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue
        predictions.append(pred)
        true_labels.append(features.get("label"))
        urls.append(features.get("url"))
    return predictions, true_labels, urls

def backtest():
    # Load model and transformers
    model, vectorizer, encoder = load_artifacts_and_model()

    with open(NUMERIC_FEATURES_ORDER_PATH, "rb") as fn:
        numeric_feature_names_loaded = pickle.load(fn)

    # Load cached feature data for backtesting
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError("Cached feature data does not exist. Please generate cached_features.pkl first.")
    df = pd.read_pickle(CACHE_PATH)
    df = df.dropna()  # Ensure no missing values

    # Run backtest: pass in numeric_feature_names_loaded
    predictions, true_labels, urls = backtest_model(df, model, vectorizer, encoder, numeric_feature_names_loaded)

    # Compute backtest metrics
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)

    print(f"\nBacktest Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    phishing_urls = [url for pred, url in zip(predictions, urls) if pred == 1]
    print("\nThe following URLs are predicted as phishing sites:")
    for url in phishing_urls:
        print(url)


def get_result(url: str):
    """
    Load the trained model and transformers, then predict the label for the given URL.
    Returns:
      1 if predicted as phishing,
      0 if predicted as legitimate,
     -1 if the URL is invalid.
    """
    model, vectorizer, encoder = load_artifacts_and_model()
    with open(NUMERIC_FEATURES_ORDER_PATH, "rb") as fn:
        numeric_feature_names = pickle.load(fn)

    url_data = URLData(url)
    if url_data.is_valid:
        features = url_data.extract_features()
        X_new = transform_features(features, vectorizer, encoder, numeric_feature_names)
        prediction = model.predict(X_new)
        return prediction[0] # 1- phishing, 0- legitimate
    else:
        print("The URL is invalid or missing some features, unable to predict.")
        return -1 # -1 for invalid URL

if __name__ == "__main__":
    backtest()
