# Purlin Production ML Project
Riddhi Shelke (A9-40), Sanchi Sharma (A9-57), Anupama Sanjeevan (A9-44) - Ramdeobaba University

Machine-learning analysis of purlin manufacturing logs.
* **Task 1 - machine classification:** predict which machine (New Machine, No.1, No.2) ran a job.
* **Machine recommender:** model probabilities + each machine's *observed* capability range
  (historical routing, not proven best performance - no cost/speed/downtime data exists).
* **Task 2 - yield regression.**

The report's main results use the **April-June snapshot** (2,953 jobs); Section 9 of the report extends
the analysis to **April-October** (9,005 jobs) and shows why results change on later months.

## Quick start (macOS / Linux)
```bash
cd purlin-ml-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py        # works immediately: data + model are included
```
Optional - regenerate results:
```bash
# extended data (Apr-Oct, 9,005 jobs) - train_machine.py takes ~10 min on one core
python src/prepare_data.py              # data/raw/*.xlsx -> data/processed/purlin_clean.csv
python src/train_machine.py             # -> results_machine.csv, figures/
python src/train_final.py               # deployment model -> models/machine_model.joblib
python src/train_yield.py               # -> results_yield.csv
python src/experiment_recent_routing.py # recent-routing features
python src/experiment_walkforward.py    # weekly walk-forward test
# April-June snapshot (report Sections 2-8)
python src/legacy/train_machine_apr_jun.py   # -> results_machine_apr_jun.csv, figures/apr_jun/
python src/legacy/train_yield_apr_jun.py     # -> results_yield_apr_jun.csv
```
If the app reports a model/version problem run `python src/train_final.py`; the app also rebuilds the
model itself. `data/raw/` is git-ignored (confidential) - do not push it to GitHub.

## Structure
```
app/streamlit_app.py      interactive recommender       src/prepare_data.py    raw workbook -> clean CSV
src/recommend.py          observed ranges + ranking     src/train_machine.py   model comparison (Apr-Oct)
src/train_final.py        deployment model              src/train_yield.py     Task 2 (Apr-Oct)
src/legacy/               Apr-Jun scripts               data/processed/        purlin_clean.csv (Apr-Oct),
figures/, figures/apr_jun/  charts                                              purlin_clean_apr_jun.csv
results_*.csv             results                       report/                project report (.docx)
```

## Task 1 results (with customer group; macro-F1 in brackets)
| Model | Apr-Jun random | Apr-Jun time (June) | Apr-Oct random | Apr-Oct time (Sep-Oct) |
|---|---|---|---|---|
| Gradient Boosting | 97.8% (0.964) | 86.9% (0.706) | 93.9% (0.921) | 53.4% (0.415) |
| Random Forest | 97.5% (0.957) | 84.6% (0.728) | 93.9% (0.918) | 53.5% (0.400) |
| KNN | 94.6% (0.915) | 92.3% (0.759) | 88.9% (0.852) | 49.2% (0.370) |
| SVM (RBF) | 93.1% (0.889) | 95.8% (0.861) | 84.6% (0.811) | 53.1% (0.450) |
| Logistic Regression | 88.0% (0.824) | 74.9% (0.617) | 77.2% (0.721) | 47.8% (0.391) |
| Majority baseline | 71.7% (0.278) | 84.8% (0.306) | 62.1% (0.255) | 50.7% (0.224) |

Time split: Apr-Jun = train Apr-May / test June; Apr-Oct = train Apr-Aug / test Sep-Oct.

**Why the time-split result drops on the extended data:** June was an unusually easy month (85% New
Machine), and from September routing changed - e.g. SISCOL ran 100% on New Machine in Apr-Aug but 148 of
278 jobs went to No.1 in Sep-Oct, and new customers appeared. Recent-routing features lift Sep-Oct accuracy
to ~72% (Random Forest, 14-day lookback). In a weekly walk-forward test the rule "use the machine that ran the
same job last time" reaches 76.0% overall: 84.9% on jobs with an identical earlier job (31% of jobs), 72.0% on
the rest. Random-split results are inflated by repeated jobs (43.8% / 39.4% exact overlap with training).

## Task 2 - yield (best model per feature set; CV R2 / time-split R2)
| Feature set | Apr-Jun (n=534) | Apr-Oct (n=1,406) |
|---|---|---|
| A: job specs only | 0.33 / 0.02 | 0.31 / 0.06 |
| B: + purlin count | 0.68 / 0.61 | 0.75 / 0.62 |
| C: + purlin count + coil weight | 0.93 / 0.90 | 0.96 / 0.94 |

Set C is an exact identity (yield = unit weight x purlins / coil weight, R2 = 1.000), not forecasting skill.

## Notes / limitations
- `UNIT_WEIGHT_KG` = thickness x width x length x 7850 / 1e9 (adds no information beyond dimensions).
- Machine ranges are empirical, not engineering limits - confirm with the plant.
- Plausibility limits and customer groupings are documented choices in `src/prepare_data.py`.
- The Apr-Jun clean file was produced by an earlier version of the preparation script; `prepare_data.py`
  now processes the extended workbook. Results can differ slightly between scikit-learn versions.
