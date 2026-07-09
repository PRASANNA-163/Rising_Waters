"""
app.py
-------
Flask web application for Rising Waters — an early flood-warning system.
Loads the trained model (produced by train_model.py) and serves a simple
UI where meteorologists / disaster-response coordinators can enter
seasonal rainfall readings for a region/year and get an instant
flood-risk prediction.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "flood_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
COLUMNS_PATH = MODELS_DIR / "model_columns.pkl"
NUMERIC_COLUMNS_PATH = MODELS_DIR / "numeric_columns.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

model = None
scaler = None
model_columns = None
numeric_columns = None
metadata = {}


def load_artifacts():
    global model, scaler, model_columns, numeric_columns, metadata
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        numeric_columns = joblib.load(NUMERIC_COLUMNS_PATH)
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
    else:
        model = None


load_artifacts()


def build_feature_row(form) -> pd.DataFrame:
    """Turn the submitted seasonal rainfall readings into a single-row
    DataFrame matching the exact column layout the model was trained on."""
    winter = float(form.get("winter_rainfall", 0))
    pre_monsoon = float(form.get("pre_monsoon_rainfall", 0))
    monsoon = float(form.get("monsoon_rainfall", 0))
    post_monsoon = float(form.get("post_monsoon_rainfall", 0))
    annual = winter + pre_monsoon + monsoon + post_monsoon

    raw = {
        "AnnualRainfall": annual,
        "WinterRainfall": winter,
        "PreMonsoonRainfall": pre_monsoon,
        "MonsoonRainfall": monsoon,
        "PostMonsoonRainfall": post_monsoon,
    }

    df = pd.DataFrame([raw])
    df = df[model_columns]
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df, annual


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=model is not None,
        best_model_name=metadata.get("best_model"),
        accuracy_by_model=metadata.get("accuracy_by_model", {}),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "result.html",
            error="No trained model found. Run 'python train_model.py' first.",
        )

    try:
        features, annual_rainfall = build_feature_row(request.form)
        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(features)[0][prediction]) * 100
        else:
            confidence = None

        result = "High Flood Risk" if prediction == 1 else "Low Flood Risk"

        return render_template(
            "result.html",
            result=result,
            is_flood=(prediction == 1),
            confidence=round(confidence, 2) if confidence is not None else None,
            model_name=metadata.get("best_model", "Model"),
            annual_rainfall=round(annual_rainfall, 1),
        )
    except Exception as exc:  # noqa: BLE001
        return render_template("result.html", error=f"Prediction failed: {exc}")


if __name__ == "__main__":
    app.run(debug=True)
