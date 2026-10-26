"""Purlin machine predictor - interactive inference for Task 1.

Run from the project root:  streamlit run app/streamlit_app.py
Needs models/machine_pipeline.joblib (created by src/train_machine.py).
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "machine_pipeline.joblib"
DATA_PATH = ROOT / "data" / "processed" / "purlin_clean.csv"
STEEL_DENSITY = 7850  # kg/m^3; UNIT_WEIGHT_KG = t * w * L * density (exact in the data)

st.set_page_config(page_title="Purlin Machine Predictor", page_icon="🏭", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_defaults():
    """Typical values + valid ranges, taken from the cleaned data."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train_machine.py` first.")
    st.stop()

model = load_model()
data = load_defaults()
parties = list(model.named_steps["prep"].named_transformers_["cat"].categories_[0])
classes = list(model.named_steps["clf"].classes_)
model_name = type(model.named_steps["clf"]).__name__

st.title("🏭 Purlin Machine Predictor")
st.caption(f"Predicts which machine ran a job from its specification. Model: **{model_name}**.")

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

# ---------- prediction -------------------------------------------------------
if st.button("Predict machine", type="primary", use_container_width=True):
    X = pd.DataFrame([{"THICKNESS_MM": thickness, "WIDTH_MM": width, "GSM": gsm,
                       "LENGTH_MM": length, "NO_OF_PURLIN": purlins,
                       "UNIT_WEIGHT_KG": unit_weight, "PARTY_GROUP": party}])
    proba = pd.Series(model.predict_proba(X)[0], index=classes).sort_values(ascending=False)
    top = proba.index[0]
    st.success(f"Predicted machine: **{top}**  ({proba.iloc[0]:.0%} confidence)")
    st.bar_chart(proba.rename("Probability"), horizontal=True)
    if proba.iloc[0] < 0.6:
        st.info("Low confidence - the specification sits between machines. Treat as a suggestion.")

with st.expander("About this model"):
    st.markdown(
        "- Trained on April-June 2026 production logs (cleaned `purlin_clean.csv`).\n"
        "- Uses thickness, width, GSM, length, purlin count, unit weight and customer group.\n"
        "- Customer group is the strongest signal; accuracy drops when it is removed.\n"
        "- On a **time-based** test (train Apr-May, test Jun) performance is noticeably lower than on a "
        "random split - expect real-world accuracy closer to the time-split figures in the report.")
