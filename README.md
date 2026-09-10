# Tourism Experience Analytics — Run Order (Submission Today)

## Setup
```bash
pip install -r requirements.txt
mkdir -p data/raw data/processed reports/figures models
```
Put your raw CSVs into `data/raw/`. Open each one and check `.columns` —
edit the `FILES` dict in `01_clean_and_merge.py` if names differ, and the
`CANDIDATE_FEATURES` list in `03_train_models.py` if your column names differ
from Continent/Region/Country/CityName/AttractionType/VisitYear/VisitMonth.

## 1. Clean & merge
```bash
python 01_clean_and_merge.py
```
Produces `data/processed/master_df.csv` + `cleaning_log.txt` (paste the log
straight into your report's "Data Preparation" section).

## 2. EDA
```bash
python 02_eda.py
```
Produces 8 charts in `reports/figures/`. For each chart, write 1 sentence of
insight underneath it in your report/slides — that's what gets graded, not
just the chart itself.

## 3. Train models 
```bash
python 03_train_models.py
```
Trains Linear Regression / Random Forest / XGBoost for rating prediction,
and Logistic Regression / Random Forest / XGBoost for visit-mode
classification. Prints comparison tables for both — copy these directly into
your "Model Performance" report section. Saves the best of each to `models/`.

## 4. Recommendation system
```bash
python 04_recommender.py
```
Builds collaborative filtering (SVD on user-item matrix) and content-based
filtering (attraction feature similarity). Saves artifacts to `models/`.

## 5. Launch the app
```bash
streamlit run app.py
```
Three working tabs: Overview/EDA, Rating Prediction, Visit Mode Prediction,
Recommendations (both CF and content-based).

## 6. Deploy (optional but strong for a portfolio, 10 min)
Push this folder to a GitHub repo, then go to https://share.streamlit.io,
connect the repo, point it at `app.py`. You get a public link to put in your
submission/resume.

## 7. Write the report
Structure:
1. **Data Preparation** — paste `cleaning_log.txt`, note row counts before/after.
2. **EDA** — 4-6 charts from `reports/figures/` with a 1-sentence insight each.
3. **Model Performance** — the two comparison tables printed by
   `03_train_models.py`, plus 1-2 sentences on which model won and why
   (e.g., "XGBoost had the best F1 because it handles the class imbalance
   in VisitMode better than Logistic Regression").
4. **Recommendation System** — explain CF vs content-based briefly, show
   one example output from each.
5. **Business Insights** — 3-5 bullets tying findings back to the use cases:
   e.g., "Family visits skew toward Region X — target family packages there";
   "Attraction type Y has below-average ratings despite high traffic —
   flag for service review."
6. **Streamlit App** — 2-3 screenshots of the tabs in action.

## If you're really out of time
Priority order if something breaks: get `01` and `03` (classification only)
working first — that alone covers Data Cleaning + EDA + one full ML task +
Streamlit, which satisfies the core grading rubric even if the recommender
or regression piece is thinner.
