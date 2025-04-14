import os
import pickle
from datetime import datetime
import cudf
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from zmq.utils.garbage import gc
from src.url_data import URLData
import joblib


# Limit the number of threads to reduce resource contention
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ========================================
# Paths and Filenames
# ========================================
RESULT_PATH = os.path.join(os.path.dirname(__file__), f'../result_model/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
MODEL_PATH = f'{RESULT_PATH}/model.joblib'
CACHE_PATH = os.path.join(os.path.dirname(__file__), "../cache/cached_features.pkl")
VECTORIZER_PATH = f'{RESULT_PATH}/vectorizer.pkl'
ENCODER_PATH = f'{RESULT_PATH}/encoder.pkl'
NUMERIC_FEATURES_ORDER_PATH = os.path.join(os.path.dirname(__file__), "../cache/numeric_features_order.pkl")

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# ========================================
# Load and Cache Features
# ========================================
if not os.path.exists(CACHE_PATH):
    # Read data from CSV/JSON
    first = cudf.read_csv(os.path.join(os.path.dirname(__file__), '../../data/first.csv'))
    second = cudf.read_csv(os.path.join(os.path.dirname(__file__), '../../data/second.csv'))
    third = cudf.read_json(os.path.join(os.path.dirname(__file__), '../../data/third.json'))
    fourth = cudf.read_json(os.path.join(os.path.dirname(__file__), '../../data/fourth.json'))

    # Convert each record to a URLData object and extract features
    # (make sure extract_features() returns the 'url' field)
    first_feature_list = [
        URLData(url, label)
        for url, label in zip(first['URL'].to_pandas(), first['label'].to_pandas())
        if URLData(url, label).is_valid and URLData(url, label).has_all_features
    ]
    second_feature_list = [
        URLData(url, label)
        for url, label in zip(second['url'].to_pandas(),
                              second['status'].map({'legitimate': 0, 'phishing': 1}).to_pandas())
        if URLData(url, label).is_valid and URLData(url, label).has_all_features
    ]
    third_feature_list = [
        URLData(url, label)
        for url, label in zip(third['text'].to_pandas(),
                              third['label'].astype('int64').to_pandas())
        if URLData(url, label).is_valid and URLData(url, label).has_all_features
    ]
    fourth_feature_list = [
        URLData(url, label)
        for url, label in zip(fourth['text'].to_pandas(),
                              fourth['label'].astype('int64').to_pandas())
        if URLData(url, label).is_valid and URLData(url, label).has_all_features
    ]

    # Convert to DataFrame (each dict contains the 'url' field)
    df1 = pd.DataFrame([obj.extract_features() for obj in first_feature_list])
    df2 = pd.DataFrame([obj.extract_features() for obj in second_feature_list])
    df3 = pd.DataFrame([obj.extract_features() for obj in third_feature_list])
    df4 = pd.DataFrame([obj.extract_features() for obj in fourth_feature_list])

    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = df.dropna()

    # Cache locally to avoid reprocessing
    df.to_pickle(CACHE_PATH)
else:
    df = pd.read_pickle(CACHE_PATH)

# ========================================
# Construct Feature Matrix
# ========================================
# Process the text feature "ngram_hostname": join list into a string
ngram_texts = df["ngram_hostname"].apply(lambda x: " ".join(x) if x else "")
# Process the category feature "tld": convert to string
tld_values = df["tld"].astype(str)

# Create TF-IDF vectorizer and perform fit_transform
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_ngram = vectorizer.fit_transform(ngram_texts)

# Create OneHotEncoder and perform fit_transform
encoder = OneHotEncoder(handle_unknown="ignore")
X_tld = encoder.fit_transform(tld_values.values.reshape(-1, 1))
y = df["label"].values

# Get the order of numeric features by dropping non-numeric columns ("label", "url", "ngram_hostname", "tld")
numeric_feature_names = list(df.drop(columns=["label", "url", "ngram_hostname", "tld"]).columns)

# Save the numeric feature order to file
with open(NUMERIC_FEATURES_ORDER_PATH, "wb") as fn:
    pickle.dump(numeric_feature_names, fn)

X_numeric = df[numeric_feature_names].to_numpy()


# Process sparse matrices and convert data types
if isinstance(X_tld, csr_matrix):
    X_tld = X_tld.toarray()
X_tld = X_tld.astype(np.float64)
X_numeric = X_numeric.astype(np.float64)
X_ngram = X_ngram.toarray().astype(np.float64)

# Combine all features
X_final = np.hstack((X_numeric, X_ngram, X_tld))

# Create an index array to preserve original indices
indices = np.arange(len(X_final))

# ========================================
# Split Dataset (and return indices)
# ========================================
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X_final, y, indices, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.5, random_state=42
)

# Uncomment below to use custom grid search function to find the best hyperparameters
# best_params, best_val_accuracy = custom_grid_search_rf(X_train, y_train, X_val, y_val)
# print("🔹 Best Parameters:", best_params)
# print(f"🔹 Best Validation Accuracy: {best_val_accuracy:.4f}")

# ========================================
# Train Model
# ========================================
param_grid = {
    'n_estimators': 100,
    'max_depth': 10,
    'max_features': 1.0
}

model = cuRF(**param_grid, random_state=42)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred)

print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print("📊 Test Classification Report:\n", test_classification_report)

# Save model
joblib.dump(model, MODEL_PATH)

# Release resources
try:
    if model.treelite_model is not None:
        model.treelite_model.close()
        model.treelite_model = None
        model = None
        gc.collect()
except Exception as e:
    pass

# ========================================
# Save Vectorizer / Encoder
# ========================================
with open(VECTORIZER_PATH, "wb") as fv:
    pickle.dump(vectorizer, fv)

with open(ENCODER_PATH, "wb") as fe:
    pickle.dump(encoder, fe)

print(f"✅ vectorizer/encoder saved -> {VECTORIZER_PATH}, {ENCODER_PATH}")
