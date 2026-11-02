"""Fit the deployment model (histogram Gradient Boosting, best random-split model) on ALL data.

Run from the project root:  python src/train_final.py   -> models/machine_model.joblib
Also imported by the Streamlit app, which rebuilds the model automatically if the saved
file is missing or was made with a different scikit-learn version.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "purlin_clean.csv"
MODEL = ROOT / "models" / "machine_model.joblib"
NUM = ["THICKNESS_MM", "WIDTH_MM", "GSM", "LENGTH_MM", "NO_OF_PURLIN", "UNIT_WEIGHT_KG"]


def train(save=True):
    df = pd.read_csv(DATA)
    y = df["MACHINE"].str.replace("MACHINE ", "", regex=False).str.title()
    pipe = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM),
                                                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["PARTY_GROUP"])])),
                     ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                        class_weight="balanced", random_state=42))])
    pipe.fit(df[NUM + ["PARTY_GROUP"]], y)
    if save:
        MODEL.parent.mkdir(exist_ok=True)
        joblib.dump(pipe, MODEL)
    return pipe


if __name__ == "__main__":
    p = train()
    print("saved", MODEL, list(p.named_steps["clf"].classes_))
