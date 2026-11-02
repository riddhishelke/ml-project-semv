"""Task 2: how well can YIELD_PCT be predicted, and which inputs make it (trivially) predictable?

Run from the project root:  python src/train_yield.py
Writes: results_yield_apr_jun.csv, figures/apr_jun/yield_feature_sets.png [legacy April-June snapshot]

Key facts verified in the data (all rows with a recorded coil weight):
    PRODUCTION_KG = UNIT_WEIGHT_KG * NO_OF_PURLIN          (exact)
    YIELD_PCT     = PRODUCTION_KG / QTY_WT * 100           (exact)
so YIELD_PCT is "share of the coil used in this log entry", an arithmetic identity once the
purlin count and coil weight are known - not a learnable process-efficiency measure.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
df = pd.read_csv("data/processed/purlin_clean_apr_jun.csv", parse_dates=["DATE"])
d = df[df.YIELD_PCT.notna()].reset_index(drop=True)
y = d.YIELD_PCT

# ---- verify the identities -------------------------------------------------
print(f"rows with yield: {len(d)}")
print("max |PRODUCTION_KG - UNIT_WEIGHT*N|  :", float((d.UNIT_WEIGHT_KG * d.NO_OF_PURLIN - d.PRODUCTION_KG).abs().max()))
print("max |YIELD - PRODUCTION/QTY_WT*100|  :", float((d.PRODUCTION_KG / d.QTY_WT * 100 - y).abs().max()))

SPEC = ["THICKNESS_MM", "WIDTH_MM", "GSM", "LENGTH_MM", "UNIT_WEIGHT_KG"]
SETS = {
    "A: job specs only":            SPEC,
    "B: + purlin count":            SPEC + ["NO_OF_PURLIN"],
    "C: + coil weight (QTY_WT)":    SPEC + ["NO_OF_PURLIN", "QTY_WT"],
}
CAT = ["PARTY_GROUP", "MACHINE"]


def make(model, num):
    return Pipeline([("prep", ColumnTransformer([("n", StandardScaler(), num),
                                                 ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])),
                     ("m", model)])


MODELS = {"Ridge": Ridge(alpha=1.0),
          "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
          "Gradient Boosting": GradientBoostingRegressor(random_state=SEED)}

tr, te = d.index[d.DATE < "2026-06-01"], d.index[d.DATE >= "2026-06-01"]
print(f"time split: train {len(tr)} (Apr-May), test {len(te)} (Jun)")
rows = []
for sname, num in SETS.items():
    X = d[num + CAT]
    for mname, m in MODELS.items():
        cvp = cross_val_predict(make(m, num), X, y, cv=KFold(5, shuffle=True, random_state=SEED))
        p = make(m, num).fit(X.loc[tr], y[tr]).predict(X.loc[te])
        rows.append(dict(feature_set=sname, model=mname,
                         cv_r2=r2_score(y, cvp), cv_mae=mean_absolute_error(y, cvp),
                         time_r2=r2_score(y[te], p), time_mae=mean_absolute_error(y[te], p)))
        print(f"{sname:28s} {mname:18s} CV R2={rows[-1]['cv_r2']:.3f} MAE={rows[-1]['cv_mae']:.1f} | "
              f"time R2={rows[-1]['time_r2']:.3f} MAE={rows[-1]['time_mae']:.1f}")
res = pd.DataFrame(rows)
res.to_csv("results_yield_apr_jun.csv", index=False)

# baseline: predict the training mean
base = r2_score(y[te], np.full(len(te), y[tr].mean()))
print(f"mean-predictor baseline on time split: R2={base:.3f}, MAE={mean_absolute_error(y[te], np.full(len(te), y[tr].mean())):.1f}")

# the identity itself, as a 'model' with zero learning
ident = d.UNIT_WEIGHT_KG * d.NO_OF_PURLIN / d.QTY_WT * 100
print(f"closed-form formula R2={r2_score(y, ident):.6f}")

# ---- figure -----------------------------------------------------------------
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
best = res.groupby("feature_set")[["cv_r2", "time_r2"]].max()
fig, ax = plt.subplots(figsize=(7, 3.6))
xs = np.arange(len(best)); w = .36
ax.bar(xs - w / 2, best.cv_r2, w, color=BLUE, label="5-fold CV")
ax.bar(xs + w / 2, best.time_r2.clip(lower=0), w, color=ORANGE, label="Time split (June)")
for x, a, b in zip(xs, best.cv_r2, best.time_r2):
    ax.text(x - w / 2, a + .015, f"{a:.2f}", ha="center"); ax.text(x + w / 2, max(b, 0) + .015, f"{b:.2f}", ha="center")
ax.set_xticks(xs); ax.set_xticklabels(best.index, fontsize=8); ax.set_ylim(0, 1.12)
ax.set_ylabel("Best R² over models"); ax.legend(frameon=False, loc="upper left")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("figures/apr_jun/yield_feature_sets.png", dpi=200); plt.close()
