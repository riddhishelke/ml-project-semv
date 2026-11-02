"""[Legacy: April-June 2026 snapshot, as used for the main results in the report]
Task 1: classify which MACHINE ran a job from its specs.

Run from the project root:  python src/train_machine.py
Writes: figures/*.png, models/machine_pipeline.joblib, results_machine.csv
"""
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
NUM = ["THICKNESS_MM", "WIDTH_MM", "GSM", "LENGTH_MM", "NO_OF_PURLIN", "UNIT_WEIGHT_KG"]
CAT = ["PARTY_GROUP"]
TARGET = "MACHINE"

df = pd.read_csv("data/processed/purlin_clean_apr_jun.csv", parse_dates=["DATE"])
y = df[TARGET].str.replace("MACHINE ", "", regex=False).str.title()  # No.1 / No.2 / New Machine -> readable

# ---------- two splits -------------------------------------------------------
# (a) random stratified 80/20      (b) time-based: train Apr-May, test Jun
rand_idx = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y)
time_idx = (df.index[df.DATE < "2026-06-01"], df.index[df.DATE >= "2026-06-01"])
SPLITS = {"random": rand_idx, "time": time_idx}


def prep(use_party):
    cat = CAT if use_party else []
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              *([("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else [])])


MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                            {"clf__C": [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 7, 15]}),
    "SVM (RBF)": (SVC(class_weight="balanced"), {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 12]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]}),
}

rows, fitted = [], {}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for split, (tr, te) in SPLITS.items():
    for use_party in (True, False):
        feats = NUM + (CAT if use_party else [])
        for name, (est, grid) in MODELS.items():
            pipe = Pipeline([("prep", prep(use_party)), ("clf", est)])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
            gs.fit(df.loc[tr, feats], y[tr])
            p = gs.predict(df.loc[te, feats])
            rows.append(dict(split=split, party_feature=use_party, model=name,
                             cv_f1=gs.best_score_, accuracy=accuracy_score(y[te], p),
                             precision=precision_score(y[te], p, average="macro"),
                             recall=recall_score(y[te], p, average="macro"),
                             f1_macro=f1_score(y[te], p, average="macro"),
                             best_params=str(gs.best_params_)))
            fitted[(split, use_party, name)] = (gs.best_estimator_, te, p)
            print(f"{split:6s} party={use_party!s:5s} {name:20s} acc={rows[-1]['accuracy']:.3f} F1={rows[-1]['f1_macro']:.3f}")

res = pd.DataFrame(rows)
res.to_csv("results_machine_apr_jun.csv", index=False)

# baseline for context
for split, (tr, te) in SPLITS.items():
    maj = y[tr].mode()[0]
    print(f"baseline {split}: acc={(y[te] == maj).mean():.3f}, macroF1="
          f"{f1_score(y[te], [maj] * len(te), average='macro'):.3f}")

# ---------- figures ----------------------------------------------------------
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
order = (res[(res.split == "random") & res.party_feature]
         .sort_values("f1_macro").model.tolist())  # same row order in both panels
for ax, split in zip(axes, SPLITS):
    sub = (res[(res.split == split) & res.party_feature]
           .set_index("model").loc[order].reset_index())
    best = sub.f1_macro.idxmax()
    ax.barh(sub.model, sub.f1_macro, color=[ORANGE if i == best else BLUE for i in sub.index], height=.6)
    for y_, v in enumerate(sub.f1_macro):
        ax.text(v + .008, y_, f"{v:.3f}", va="center", color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_title(f"{split.title()} split", loc="left", color=INK)
    ax.set_xlabel("Macro F1 on test set")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("figures/apr_jun/model_comparison.png", dpi=200); plt.close()

# confusion matrix + importances for the best random-split model
best_row = res[(res.split == "random") & res.party_feature].sort_values("f1_macro").iloc[-1]
est, te, p = fitted[("random", True, best_row.model)]
fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay.from_predictions(y[te], p, ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"{best_row.model} (random split)", loc="left")
plt.tight_layout(); plt.savefig("figures/apr_jun/confusion_matrix.png", dpi=200); plt.close()

rf, _, _ = fitted[("random", True, "Random Forest")]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.barh([n.split("__")[1] for n in imp.index], imp.values, color=BLUE, height=.6)
ax.set_xlabel("Random Forest importance"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("figures/apr_jun/feature_importance.png", dpi=200); plt.close()

joblib.dump(fitted[("random", True, best_row.model)][0], "models/legacy_apr_jun_pipeline.joblib")
print("\nBest (random split, with party):", best_row.model, round(best_row.f1_macro, 3))
