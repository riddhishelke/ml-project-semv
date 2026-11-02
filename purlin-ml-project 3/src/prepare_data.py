"""Raw daily production workbook -> data/processed/purlin_clean.csv

Run from the project root:  python src/prepare_data.py [path/to/DAILY_PRODUCTION_REPORT.xlsx]
Default input: data/raw/DAILY_PRODUCTION_REPORT-2026-2027.xlsx   (Apr-Oct 2026, 3 machine sheets)

Cleaning steps (row counts are printed at every step):
  0. Read the three machine sheets. Dates: only the first row of a day carries one, so it is
     forward-filled. Dates arrive in mixed formats (Excel dates, "12-07-2026", "17/7/2026",
     "2026-07-26"), parsed day-first; two corrupted first-day cells (years 4752 / 7490) are replaced by
     "previous date + 1 day"; stray non-dates are ignored. Numeric columns are coerced, text is
     stripped and upper-cased, COIL NO stays text. A few free-text annotations in column 13 are dropped.
  1. Remove downtime ("MACHINE OFF") and rows where every specification column is blank.
  2. Drop rows missing thickness, width, length or purlin count.
  3. Drop exact duplicate rows across all columns.
  4. Drop physically implausible rows (limits in LIMITS, e.g. width 40 mm, 2,308 mm; 0 or 643 purlins).
Then: impute missing GSM with the median GSM of the same thickness (nearest known thickness if none)
(flagged in GSM_WAS_MISSING),
derive yield, unit weight, calendar fields and a consolidated customer group.
"""
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "raw" / "DAILY_PRODUCTION_REPORT-2026-2027.xlsx"
OUT = ROOT / "data" / "processed" / "purlin_clean.csv"

SHEETS = ["NEW MACHINE", "MACHINE NO.1", "MACHINE NO.2"]
RAW_COLS = ["DATE", "COIL_NO", "QTY_WT", "THICKNESS_MM", "WIDTH_MM", "GSM", "PARTY_NAME",
            "DRAWING_PART", "NO_OF_PURLIN", "LENGTH_MM", "PRODUCTION_KG", "BALANCE_KG"]
NUMERIC = ["QTY_WT", "THICKNESS_MM", "WIDTH_MM", "GSM", "NO_OF_PURLIN", "LENGTH_MM",
           "PRODUCTION_KG", "BALANCE_KG"]
SPECS = ["THICKNESS_MM", "WIDTH_MM", "GSM", "NO_OF_PURLIN", "LENGTH_MM"]
ESSENTIAL = ["THICKNESS_MM", "WIDTH_MM", "LENGTH_MM", "NO_OF_PURLIN"]
# plausibility limits (physical range of purlin roll-forming); GSM is a coating weight
LIMITS = {"THICKNESS_MM": (1.0, 4.0), "WIDTH_MM": (150, 600), "LENGTH_MM": (200, 20000),
          "NO_OF_PURLIN": (1, 500), "GSM": (0, 1000)}
STEEL_DENSITY = 7850  # kg/m^3
DATE_YEAR = 2026

# ordered (regex on normalised name, group); first match wins, else OTHER
GROUP_RULES = [
    (r"WALONI.*DEEPAK SIR", "WALONI DEEPAK SIR"), (r"^WALONI", "WALONI"),
    (r"^VIKRAM SOLAR", "VIKRAM SOLAR"), (r"^SISCOL", "SISCOL"),
    (r"^TEJAS STEEL", "TEJAS STEEL"), (r"^PEB", "PEB"), (r"^MR\b", "MR"), (r"^AD\b", "AD"),
    (r"^SHREE (SHIV )?SHAKTI", "SHREE SHAKTI"), (r"^PM\b", "PM"), (r"^APOLLO", "APOLLO"),
    (r"^URC", "URC"), (r"^SPB", "SPB"), (r"^MAYUR ROOF", "MAYUR ROOF"), (r"^ASP\b", "ASP"),
    (r"^METALIFY", "METALIFY"), (r"^MECHA STEEL", "MECHA STEEL"),
    (r"^RADHIKA PROFILE", "RADHIKA PROFILE"), (r"^SHRAVANI SALES", "SHRAVANI SALES"),
    (r"^SANJAY BORGAON", "SANJAY BORGAON"), (r"^SENAPATHY", "SENAPATHY"), (r"^NMK", "NMK"),
    (r"^PRENCO", "PRENCO"), (r"^PRAGYA", "PRAGYA TRADE HUB"), (r"^CAR SHOWROOM", "CAR SHOWROOM"),
    (r"^SUPERSHEET", "SUPERSHEET"), (r"^DC\b", "DC"), (r"^PENNAR STEEL", "PENNAR STEEL"),
    (r"^STEELCO", "STEELCO"), (r"^KRISH(N|L)A ENTERPRISES", "KRISHNA ENTERPRISES"),
    (r"^METAL ART", "METAL ART"), (r"^WEBSOL", "SISCOL"),
]


def party_group(name):
    """Consolidate spelling variants of a customer into one group; rare customers -> OTHER."""
    if pd.isna(name):
        return "OTHER"
    n = re.sub(r"\s+", " ", str(name).upper().replace(".", "").replace("'", "")).strip()
    for pattern, group in GROUP_RULES:
        if re.search(pattern, n):
            return group
    return "OTHER"


def parse_date(x):
    """Return a Timestamp, 'BAD' for a corrupted date cell, or NaT for anything that is not a date."""
    if isinstance(x, (dt.datetime, pd.Timestamp)):
        return pd.Timestamp(x) if x.year == DATE_YEAR else "BAD"
    if isinstance(x, str):
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                return pd.Timestamp(dt.datetime.strptime(x.strip(), fmt))
            except ValueError:
                pass
    return pd.NaT


def sheet_dates(col):
    prev, out = pd.Timestamp(f"{DATE_YEAR}-03-31"), []
    for x in col:
        v = parse_date(x) if pd.notna(x) else pd.NaT
        if isinstance(v, str):                       # corrupted cell -> next day in the sequence
            prev = prev + pd.Timedelta(days=1)
            out.append(prev)
        elif pd.notna(v):
            prev = v
            out.append(v)
        else:
            out.append(pd.NaT)
    return pd.Series(out, index=col.index).ffill()


def load_raw(path):
    frames = []
    for sheet in SHEETS:
        d = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=2).iloc[:, :12]
        d.columns = RAW_COLS
        d["MACHINE"] = sheet
        d["DATE"] = sheet_dates(d["DATE"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["COIL_NO", "PARTY_NAME", "MACHINE"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    return df.drop(columns="DRAWING_PART")


def clean(df):
    log = [("raw rows", len(df))]
    df = df[~df["PARTY_NAME"].fillna("").str.contains("OFF")]
    df = df[~df[SPECS + ["QTY_WT", "PRODUCTION_KG"]].isna().all(axis=1)]
    log.append(("step 1: downtime / blank rows removed", len(df)))
    df = df.dropna(subset=ESSENTIAL)
    log.append(("step 2: missing dimensions removed", len(df)))
    df = df.drop_duplicates()
    log.append(("step 3: exact duplicates removed", len(df)))
    ok = pd.Series(True, index=df.index)
    for col, (lo, hi) in LIMITS.items():
        ok &= df[col].isna() | df[col].between(lo, hi)
    df = df[ok]
    log.append(("step 4: implausible values removed", len(df)))

    df = df.copy()
    df["IS_DOWNTIME"] = False
    df["GSM_WAS_MISSING"] = df["GSM"].isna().astype(int)
    med = df.groupby("THICKNESS_MM")["GSM"].median()                 # median GSM per thickness
    known = med.dropna()
    nearest = lambda t: known.iloc[(known.index.to_series() - t).abs().argmin()]
    df["GSM"] = df["GSM"].fillna(df["THICKNESS_MM"].map(med))
    df["GSM"] = df["GSM"].fillna(df["THICKNESS_MM"].map(nearest))     # thickness never seen with a GSM -> nearest thickness
    df["YIELD_PCT"] = df["PRODUCTION_KG"] / df["QTY_WT"] * 100          # only where coil weight is logged
    df["YEAR"], df["MONTH"] = df["DATE"].dt.year, df["DATE"].dt.month
    df["DAY_OF_WEEK"] = df["DATE"].dt.day_name()
    df["PARTY_GROUP"] = df["PARTY_NAME"].map(party_group)
    df["UNIT_WEIGHT_KG"] = df["THICKNESS_MM"] * df["WIDTH_MM"] * df["LENGTH_MM"] * STEEL_DENSITY / 1e9
    df = df.sort_values("DATE", kind="mergesort").reset_index(drop=True)
    cols = ["DATE", "COIL_NO", "QTY_WT", "THICKNESS_MM", "WIDTH_MM", "GSM", "PARTY_NAME",
            "NO_OF_PURLIN", "LENGTH_MM", "PRODUCTION_KG", "BALANCE_KG", "MACHINE", "IS_DOWNTIME",
            "YIELD_PCT", "YEAR", "MONTH", "DAY_OF_WEEK", "GSM_WAS_MISSING", "PARTY_GROUP", "UNIT_WEIGHT_KG"]
    return df[cols], log


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out, log = clean(load_raw(src))
    for label, n in log:
        print(f"{label:45s} {n:6d}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out)} rows, {out['GSM_WAS_MISSING'].sum()} with imputed GSM)")
    print("\nCustomer groups:", out["PARTY_GROUP"].value_counts().to_dict())
    print("Machines:", out["MACHINE"].value_counts().to_dict())
    print("Date range:", out["DATE"].min().date(), "->", out["DATE"].max().date())
