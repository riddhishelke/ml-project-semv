"""Purlin machine recommender - interactive inference app.

Run from the project root:  streamlit run app/streamlit_app.py
Uses models/machine_model.joblib; rebuilds it automatically from the data if missing or incompatible.
"""
from pathlib import Path

import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE_PATH = ROOT / "models" / "machine_model.joblib"          # deployment model (train_final.py)
RF_PATH = ROOT / "models" / "machine_pipeline.joblib"      # fallback (train_machine.py)
MODEL_PATH = MODEL_FILE_PATH if MODEL_FILE_PATH.exists() else RF_PATH
sys.path.insert(0, str(ROOT / "src"))
from recommend import compute_envelope, recommend  # noqa: E402
DATA_PATH = ROOT / "data" / "processed" / "purlin_clean.csv"
STEEL_DENSITY = 7850  # kg/m^3; UNIT_WEIGHT_KG = t * w * L * density (exact in the data)

st.set_page_config(page_title="Purlin Machine Recommender", page_icon="🏭", layout="centered")


@st.cache_resource
def load_model():
    """Load the saved model; rebuild it from the data if missing or incompatible."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from train_final import train  # noqa: E402  (src/ is on sys.path)
        return train(save=False)


@st.cache_data
def load_defaults():
    """Typical values + valid ranges, taken from the cleaned data (None if unavailable)."""
    try:
        return pd.read_csv(DATA_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


if not MODEL_PATH.exists() and not DATA_PATH.exists():
    st.error("Neither the saved model nor data/processed/purlin_clean.csv was found.")
    st.stop()

model = load_model()
data = load_defaults()
parties = list(model.named_steps["prep"].named_transformers_["cat"].categories_[0])
classes = list(model.named_steps["clf"].classes_)
model_name = type(model.named_steps["clf"]).__name__
envelope = compute_envelope(data) if data is not None else None

st.title("🏭 Purlin Machine Recommender")
st.caption(f"Suggests where a purlin job can run, from observed machine ranges and historical routing. Model: **{model_name}**.")

# ---------- inputs -----------------------------------------------------------
med = (lambda c, d: float(data[c].median()) if data is not None else d)

st.subheader("Job specification")
c1, c2 = st.columns(2)
thickness = c1.number_input("Thickness (mm)", 1.0, 4.0, med("THICKNESS_MM", 2.0), 0.1, format="%.1f")
width = c2.number_input("Width (mm)", 150.0, 600.0, med("WIDTH_MM", 396.0), 1.0, format="%.0f")
c3, c4 = st.columns(2)
length = c3.number_input("Length (mm)", 300.0, 15000.0, med("LENGTH_MM", 8330.0), 10.0, format="%.0f")
gsm = c4.number_input("GSM (coating)", 50.0, 600.0, med("GSM", 275.0), 5.0, format="%.0f")
c5, c6 = st.columns(2)
purlins = c5.number_input("No. of purlins", 1, 200, int(med("NO_OF_PURLIN", 2)), 1)
party = c6.selectbox("Customer group", parties, index=parties.index("SISCOL") if "SISCOL" in parties else 0)

auto_weight = thickness * width * length * STEEL_DENSITY / 1e9
override = st.toggle("Enter unit weight manually", value=False)
unit_weight = (st.number_input("Unit weight (kg)", 0.5, 200.0, float(auto_weight), 0.1)
               if override else auto_weight)
st.caption(f"Unit weight (kg per purlin): **{unit_weight:.2f}**"
           + ("" if override else "  - computed from thickness x width x length"))

# ---------- sanity warnings (outside the range the model was trained on) -----
if data is not None:
    warn = []
    for label, val, col in [("Thickness", thickness, "THICKNESS_MM"), ("Width", width, "WIDTH_MM"),
                            ("Length", length, "LENGTH_MM"), ("Purlin count", purlins, "NO_OF_PURLIN")]:
        lo, hi = data[col].min(), data[col].max()
        if not lo <= val <= hi:
            warn.append(f"{label} {val:g} is outside the training range ({lo:g}-{hi:g}).")
    if party not in set(data["PARTY_GROUP"]):
        warn.append("Customer group was not seen in training.")
    if warn:
        st.warning("Prediction may be unreliable:\n\n" + "\n".join(f"- {w}" for w in warn))

# ---------- recommendation ---------------------------------------------------
if st.button("Recommend machine", type="primary", use_container_width=True):
    job = {"THICKNESS_MM": thickness, "WIDTH_MM": width, "GSM": gsm,
           "LENGTH_MM": length, "NO_OF_PURLIN": purlins,
           "UNIT_WEIGHT_KG": unit_weight, "PARTY_GROUP": party}
    proba = dict(zip(classes, model.predict_proba(pd.DataFrame([job]))[0]))
    if envelope:
        ranked, outside = recommend(job, envelope, proba)
    else:
        ranked, outside = sorted(proba.items(), key=lambda x: -x[1]), {}

    top, p_top = ranked[0]
    if outside and len(outside) == len(classes):
        st.warning("This job is outside the observed range of every machine. "
                   "Showing the raw model prediction - please check with the plant.")
    st.success(f"Recommended machine: **{top}**  ({p_top:.0%} of historical routing among machines that have run jobs like this)")
    if p_top < 0.6:
        st.info("Low confidence - the specification sits between machines. Treat as a suggestion.")

    st.markdown("**All machines**")
    for m in classes:
        if m in outside:
            st.markdown(f"- ⚠️ **{m}** - not previously run in this range: " + "; ".join(outside[m]))
        else:
            share = dict(ranked).get(m, 0.0)
            st.markdown(f"- ✅ **{m}** - within observed range, historical routing share {share:.0%}")
    st.caption("'Outside observed range' means no past job like this ran on that machine - it is a "
               "warning to verify, not proof the machine cannot run it.")

with st.expander("About this model"):
    st.markdown(
        "- Trained on April-October 2026 production logs (cleaned `purlin_clean.csv`).\n"
        "- Uses thickness, width, GSM, length, purlin count, unit weight and customer group.\n"
        "- Machine ranges are the min/max observed per machine (with a 10% margin) - they are NOT engineering limits.\n"
        "- It reflects past routing habits, not proven best performance; no cost, speed or downtime data was available.\n"
        "- Customer group is the strongest signal; accuracy drops when it is removed.\n"
        "- Machine routing changes over time (e.g. a customer moving between machines), so accuracy on "
        "future months is much lower than on a random split - see the report. Treat results as suggestions.")
