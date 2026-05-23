import pytest
import pickle
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import load_data

DATA_PATH = "data/Tweets.csv"
MODEL_PATH = "model/sentiment_model.pkl"

# ---- Data Tests ----

def test_data_loads():
    df = load_data(DATA_PATH)
    assert len(df) > 0, "Dataset should not be empty"

def test_data_has_required_columns():
    df = load_data(DATA_PATH)
    assert "text" in df.columns
    assert "airline_sentiment" in df.columns

def test_sentiment_labels():
    df = load_data(DATA_PATH)
    valid_labels = {"positive", "negative", "neutral"}
    assert set(df["airline_sentiment"].unique()).issubset(valid_labels)

def test_no_null_text():
    df = load_data(DATA_PATH)
    assert df["text"].isnull().sum() == 0, "No null values in text after cleaning"

# ---- Model Tests ----

def test_model_exists():
    assert os.path.exists(MODEL_PATH), "Model file must exist (run train.py first)"

def test_model_predicts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    result = model.predict(["The flight was delayed and staff were rude"])
    assert result[0] in ["positive", "negative", "neutral"]

def test_model_predict_proba():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    proba = model.predict_proba(["Great service, loved the flight!"])[0]
    assert abs(sum(proba) - 1.0) < 1e-6, "Probabilities must sum to 1"

def test_model_batch_predict():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    tweets = ["Amazing!", "Worst experience ever", "Flight was on time"]
    results = model.predict(tweets)
    assert len(results) == 3
