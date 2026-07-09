# Project Workflow

The project is structured into 5 Epics, each focusing on a critical
component of the flood-prediction system. The workflow ensures proper
model selection, modular development, and continuous integration with
testing.

## Epic 1 — Data Collection
Collect and load the historical weather dataset (`data/flood_prediction.csv`)
into the working environment. If no dataset is supplied, `train_model.py`
automatically generates a realistic synthetic dataset so the pipeline can
run end-to-end immediately.

## Epic 2 — Visualizing and Analysing the Data
Explore the dataset using `data/univariate_analysis.py`, which produces:
- `univariate_continuous.png` — distribution of rainfall, visibility,
  humidity, temperature, and river discharge
- `univariate_categorical.png` — counts across region, season, and flood
  occurrence
- `multivariate_swarmplot.png` — seasonal rainfall vs. flood occurrence by
  season

## Epic 3 — Data Pre-processing
Handle missing values (mean for numeric columns, mode for categorical
columns), one-hot encode categorical features (Region, Season), and scale
numeric features with `StandardScaler`. The target column (`FloodOccurred`)
is label-encoded to 0/1 and excluded from scaling.

## Epic 4 — Model Building
Train and evaluate four classification algorithms:
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

Each model is evaluated on a held-out 20% test split (`stratify=y`,
`random_state=42`). Accuracy and classification reports are compared, and
the best-performing model (XGBoost, ~96.55% accuracy on test data) is
saved to `models/flood_model.pkl` along with the fitted scaler and column
schema.

## Epic 5 — Application Building
Build the Flask web application (`app.py`) that:
1. Loads the saved model, scaler, and column schema
2. Presents a form (`templates/index.html`) for entering regional weather
   readings
3. Runs inference and displays the flood-risk classification, confidence,
   and recommended action on `templates/result.html`

## Deployment
The application is designed for deployment on IBM Cloud so authorities and
disaster-response teams can access flood-risk predictions from anywhere.
