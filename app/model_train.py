import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


TRAIN_FEATURES = "data/kaggle/train_features.csv"
MODEL_PATH = "data/model.pkl"


def train_regression_model():

    if not os.path.exists(TRAIN_FEATURES):
        raise FileNotFoundError("Run /train/evaluate first to generate train_features.csv")

    df = pd.read_csv(TRAIN_FEATURES)

    # Drop rows with errors and missing data
    df = df.dropna(subset=["true_label", "len_words", "avg_word_len"])

    # Features used for training
    FEATURE_COLS = ["len_words", "avg_word_len", "fillers", "repetitions", "punctuation"]
    TARGET_COL = "true_label"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Simple split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choose RandomForest (stable, robust without huge data)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Validation metrics
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Model trained successfully",
        "model_path": MODEL_PATH,
        "val_mae": mae,
        "val_r2": r2
    }
