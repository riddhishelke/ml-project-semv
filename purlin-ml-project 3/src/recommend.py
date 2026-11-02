"""Machine recommender = empirical feasibility envelope + historical routing probability.

The envelope is the min/max of each dimension observed per machine. It is NOT an engineering
specification - confirm limits with the plant. Importable by the Streamlit app.
"""
import pandas as pd

LIMIT_COLS = {"THICKNESS_MM": "Thickness (mm)", "WIDTH_MM": "Width (mm)",
              "GSM": "GSM", "LENGTH_MM": "Length (mm)"}


def machine_labels(df):
    return df["MACHINE"].str.replace("MACHINE ", "", regex=False).str.title()


def compute_envelope(df):
    """{machine: {col: (min, max)}} from observed jobs."""
    lab = machine_labels(df)
    return {m: {c: (float(g[c].min()), float(g[c].max())) for c in LIMIT_COLS}
            for m, g in df.groupby(lab)}


def check_job(job, envelope, tol=0.10):
    """Return {machine: [reasons it is outside the observed range]} (empty list = within range).

    Each limit is widened by `tol` x its observed span (default 10%), so values a hair outside the range
    (e.g. thickness 1.40 vs observed minimum 1.42) are not flagged."""
    out = {}
    for m, lims in envelope.items():
        why = []
        for c, (lo, hi) in lims.items():
            v, pad = job[c], tol * (hi - lo)
            if v < lo - pad or v > hi + pad:
                why.append(f"{LIMIT_COLS[c]} {v:g} outside observed {lo:g}-{hi:g}")
        out[m] = why
    return out


def recommend(job, envelope, proba, tol=0.10):
    """proba: {machine: probability}. Returns (ranked feasible [(machine, p_renormalised)], infeasible {machine: reasons})."""
    status = check_job(job, envelope, tol)
    feas = {m: proba[m] for m, w in status.items() if not w and m in proba}
    infeas = {m: w for m, w in status.items() if w}
    tot = sum(feas.values())
    if tot > 0:
        ranked = sorted(((m, p / tot) for m, p in feas.items()), key=lambda x: -x[1])
    else:  # nothing feasible by the envelope: fall back to the raw model
        ranked = sorted(proba.items(), key=lambda x: -x[1])
    return ranked, infeas
