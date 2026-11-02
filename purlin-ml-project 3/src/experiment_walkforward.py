"""Walk-forward evaluation with 'previous routing' features (strictly earlier dates only).
Run from the project root: python src/experiment_walkforward.py
For each week from 1 Sep 2026 the model is retrained on all earlier jobs and predicts that week.
Features add: machine that last ran the same exact job (party, thickness, width, length), the same
party+thickness+width, and the same party; plus the party's 14-day routing shares.
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
df = pd.read_csv("data/processed/purlin_clean.csv", parse_dates=["DATE"])
df["M"] = df.MACHINE.str.replace("MACHINE ", "", regex=False).str.title()
CL = ["New Machine", "No.1", "No.2"]; code = df.M.map({c: i for i, c in enumerate(CL)}).values
NUM = ["THICKNESS_MM", "WIDTH_MM", "GSM", "LENGTH_MM", "NO_OF_PURLIN", "UNIT_WEIGHT_KG"]


def last_machine(keys):
    out = np.full(len(df), -1.0); gap = np.full(len(df), -1.0)
    for _, g in df.groupby(keys):
        idx = g.index.values; d = g.DATE.values.astype("datetime64[D]")
        o = np.argsort(d, kind="stable"); idx, d = idx[o], d[o]
        hi = np.searchsorted(d, d, side="left") - 1            # last job on a strictly earlier date
        for j in range(len(idx)):
            if hi[j] >= 0:
                # most recent earlier job (ties on the same earlier day -> latest in file order)
                out[idx[j]] = code[idx[hi[j]]]; gap[idx[j]] = (d[j] - d[hi[j]]).astype(int)
    return out, gap


def share(days=14):
    out = np.full((len(df), 4), -1.0)
    for _, g in df.groupby("PARTY_GROUP"):
        idx = g.index.values; d = g.DATE.values.astype("datetime64[D]"); o = np.argsort(d, kind="stable"); idx, d = idx[o], d[o]
        oh = np.stack([(code[idx] == k).astype(float) for k in range(3)], 1); cum = np.vstack([np.zeros(3), oh.cumsum(0)])
        hi = np.searchsorted(d, d, side="left"); lo = np.searchsorted(d, d - np.timedelta64(days, "D"), side="left")
        c = cum[hi] - cum[lo]; n = c.sum(1, keepdims=True)
        out[idx, :3] = np.where(n > 0, c / np.maximum(n, 1), -1); out[idx, 3] = n[:, 0]
    return out


X = df[NUM].copy(); X["PG"] = df.PARTY_GROUP.astype("category").cat.codes
for name, keys in [("exact", ["PARTY_GROUP", "THICKNESS_MM", "WIDTH_MM", "LENGTH_MM"]),
                   ("spec", ["PARTY_GROUP", "THICKNESS_MM", "WIDTH_MM"]), ("party", ["PARTY_GROUP"])]:
    X[f"last_{name}"], X[f"gap_{name}"] = last_machine(keys)
S = share(14)
for k in range(3): X[f"share_{k}"] = S[:, k]
X["n14"] = S[:, 3]

weeks = pd.date_range("2026-09-01", "2026-10-27", freq="7D")
pred = pd.Series(-1, index=df.index); base = pd.Series(-1, index=df.index)
for a, b in zip(weeks[:-1], weeks[1:]):
    tr, te = df.DATE < a, (df.DATE >= a) & (df.DATE < b)
    if not te.any(): continue
    m = RandomForestClassifier(300, random_state=42, class_weight="balanced", min_samples_leaf=2).fit(X[tr], code[tr])
    pred[te] = m.predict(X[te])
te = pred >= 0
# rule baseline without ML: last machine of the exact job, else spec, else party, else New Machine
rule = np.where(X.last_exact >= 0, X.last_exact, np.where(X.last_spec >= 0, X.last_spec, np.where(X.last_party >= 0, X.last_party, 0)))
print(f"jobs evaluated (Sep 1 - Oct 26): {te.sum()}")
print(f"majority baseline accuracy      : {(code[te] == 0).mean():.3f}")
print(f"'repeat last machine' rule      : acc={accuracy_score(code[te], rule[te]):.3f}  macroF1={f1_score(code[te], rule[te], average='macro'):.3f}")
print(f"walk-forward Random Forest      : acc={accuracy_score(code[te], pred[te]):.3f}  macroF1={f1_score(code[te], pred[te], average='macro'):.3f}")
print("exact-job history available for", f"{(X.last_exact[te] >= 0).mean():.1%}", "of evaluated jobs")
for lab, msk in [("RF, with exact-job history", X.last_exact >= 0), ("RF, no exact-job history", X.last_exact < 0)]:
    mm = te & msk
    print(f"  {lab:24s} n={mm.sum():5d}  acc={accuracy_score(code[mm], pred[mm]):.3f}")

for lab, msk in [("rule, with exact-job history", X.last_exact >= 0), ("rule, no exact-job history", X.last_exact < 0)]:
    mm = te & msk
    print(f"  {lab:28s} n={mm.sum():5d}  acc={accuracy_score(code[mm], rule[mm]):.3f}")
