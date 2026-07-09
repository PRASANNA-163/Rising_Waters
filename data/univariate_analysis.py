"""
univariate_analysis.py
------------------------
Exploratory data analysis for the Rising Waters flood dataset.
Generates distribution plots for continuous and categorical features,
plus a multivariate swarmplot, saved into the data/ folder.

Run from the project root:
    python data/univariate_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("fivethirtyeight")

DATA_PATH = Path(__file__).parent / "flood_prediction.csv"
OUT_DIR = Path(__file__).parent

CONTINUOUS_COLS = [
    "AnnualRainfall",
    "SeasonalRainfall",
    "CloudVisibility",
    "Humidity",
    "Temperature",
    "RiverDischarge",
]
CATEGORICAL_COLS = ["Region", "Season", "FloodOccurred"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run train_model.py once first to "
            "generate the sample dataset."
        )
    return pd.read_csv(DATA_PATH)


def plot_continuous(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, col in zip(axes, CONTINUOUS_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="#1d7ea6")
        ax.set_title(col)
    fig.suptitle("Univariate Analysis — Continuous Features", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "univariate_continuous.png", dpi=150)
    plt.close(fig)


def plot_categorical(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        sns.countplot(x=col, data=df, ax=ax, palette="Blues_d")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Univariate Analysis — Categorical Features", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "univariate_categorical.png", dpi=150)
    plt.close(fig)


def plot_multivariate(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.swarmplot(
        x="Season",
        y="SeasonalRainfall",
        hue="FloodOccurred",
        data=df.sample(min(400, len(df)), random_state=42),
        ax=ax,
        palette={"Yes": "#d64545", "No": "#2e8b57"},
    )
    ax.set_title("Seasonal Rainfall vs Flood Occurrence by Season")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multivariate_swarmplot.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    print(df.shape)
    print(df.head())
    print(df.columns.tolist())

    plot_continuous(df)
    plot_categorical(df)
    plot_multivariate(df)
    print(f"Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
