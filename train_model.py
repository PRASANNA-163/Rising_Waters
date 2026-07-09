"""
train_model.py
----------------
Loads the historical Kerala rainfall dataset (data/flood_prediction.csv),
preprocesses it, trains four classification models (Decision Tree, Random
Forest, KNN, XGBoost), compares their performance, and saves the
best-performing model - along with the fitted scaler and the training
column order - to the models/ folder so app.py can use them for
real-time prediction.

Expected raw dataset columns (the classic Kerala rainfall dataset):
    SUBDIVISION, YEAR, JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT,
    NOV, DEC, ANNUAL RAINFALL, FLOODS

From these, four seasonal rainfall totals are engineered:
    WinterRainfall      = JAN + FEB
    PreMonsoonRainfall  = MAR + APR + MAY
    MonsoonRainfall     = JUN + JUL + AUG + SEP
    PostMonsoonRainfall = OCT + NOV + DEC

Run:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/flood_prediction.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "FloodOccurred"
RANDOM_STATE = 42

WINTER_MONTHS = ["JAN", "FEB"]
PRE_MONSOON_MONTHS = ["MAR", "APR", "MAY"]
MONSOON_MONTHS = ["JUN", "JUL", "AUG", "SEP"]
POST_MONSOON_MONTHS = ["OCT", "NOV", "DEC"]


def generate_sample_dataset(n_rows: int = 118, seed: int = 42) -> pd.DataFrame:
    """Fallback synthetic dataset (same raw schema as the real Kerala
    dataset) used only if no CSV is present, so the pipeline can still
    run end-to-end during development."""
    rng = np.random.default_rng(seed)
    years = np.arange(1901, 1901 + n_rows)

    months = {}
    month_means = {
        "JAN": 15, "FEB": 15, "MAR": 35, "APR": 100, "MAY": 220,
        "JUN": 650, "JUL": 700, "AUG": 420, "SEP": 250, "OCT": 280,
        "NOV": 150, "DEC": 45,
    }
    for m, mean in month_means.items():
        months[m] = np.clip(rng.normal(mean, mean * 0.35, size=n_rows), 0, None).round(1)

    annual = sum(months.values())
    threshold = np.percentile(annual, 55)
    floods = np.where(annual > threshold, "YES", "NO")

    df = pd.DataFrame({"SUBDIVISION": "KERALA", "YEAR": years, **months})
    df["ANNUAL RAINFALL"] = annual.round(1)
    df["FLOODS"] = floods
    return df


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
    else:
        print(f"{DATA_PATH} not found. Generating a synthetic sample dataset instead...")
        df = generate_sample_dataset()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Sample dataset saved to {DATA_PATH}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the seasonal rainfall features used by the model from the
    raw monthly columns, and standardize the target column name/values."""
    df = df.copy()

    df["WinterRainfall"] = df[WINTER_MONTHS].sum(axis=1)
    df["PreMonsoonRainfall"] = df[PRE_MONSOON_MONTHS].sum(axis=1)
    df["MonsoonRainfall"] = df[MONSOON_MONTHS].sum(axis=1)
    df["PostMonsoonRainfall"] = df[POST_MONSOON_MONTHS].sum(axis=1)
    df["AnnualRainfall"] = df["ANNUAL RAINFALL"]
    df[TARGET_COL] = df["FLOODS"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})

    feature_cols = [
        "AnnualRainfall",
        "WinterRainfall",
        "PreMonsoonRainfall",
        "MonsoonRainfall",
        "PostMonsoonRainfall",
    ]
    return df[feature_cols + [TARGET_COL]]


def preprocess(df: pd.DataFrame):
    df = df.copy()

    numeric_cols = [c for c in df.columns if c != TARGET_COL]

    # --- Handle missing values (numeric-only dataset here) ---
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[numeric_cols]
    y = df[TARGET_COL]

    # --- Scale numeric features ---
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X_scaled, y, scaler, numeric_cols


def train_and_select_best(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = round(acc * 100, 2)
        fitted_models[name] = model
        print(f"\n{name} — Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, target_names=["No Flood", "Flood"], zero_division=0))

    best_name = max(results, key=results.get)
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]}% accuracy)")

    return best_name, best_model, results


def main():
    raw_df = load_dataset()
    print(f"Raw dataset shape: {raw_df.shape}")

    df = engineer_features(raw_df)
    print(f"Engineered feature set shape: {df.shape}")

    X, y, scaler, numeric_cols = preprocess(df)
    best_name, best_model, results = train_and_select_best(X, y)

    joblib.dump(best_model, MODELS_DIR / "flood_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / "model_columns.pkl")
    joblib.dump(numeric_cols, MODELS_DIR / "numeric_columns.pkl")

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "accuracy_by_model": results,
                "features": list(X.columns),
            },
            f,
            indent=2,
        )

    print(f"\nSaved best model ({best_name}) and preprocessing artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
