"""
STEP 3 — Regression (predict Rating) + Classification (predict VisitMode)
Run after 01_clean_and_merge.py. Saves best models + encoders to models/
and prints a comparison table for both tasks (paste into your report).
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, precision_recall_fscore_support,
                              classification_report)
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

IN_PATH = "data/processed/master_df.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)
print(f"Loaded: {df.shape}")

TARGET_RATING = "Rating"
TARGET_VISITMODE = "VisitMode" if "VisitMode" in df.columns else "VisitModeId"

# ---------------- Choose feature columns ----------------
# ADJUST THIS: add/remove columns based on what's actually in your master_df.
CANDIDATE_FEATURES = [
    "Continent", "Region", "Country", "CityName",
    "AttractionType", "VisitYear", "VisitMonth",
]
FEATURES = [c for c in CANDIDATE_FEATURES if c in df.columns]
print(f"Using features: {FEATURES}")

model_df = df[FEATURES + [TARGET_RATING, TARGET_VISITMODE]].dropna()

# Encode categoricals with LabelEncoder, save encoders for the Streamlit app
encoders = {}
X = model_df[FEATURES].copy()
for c in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders[c] = le

joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

# =========================================================
# REGRESSION: predict Rating
# =========================================================
print("\n=== REGRESSION: predicting Rating ===")
y_reg = model_df[TARGET_RATING]
Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}
if HAS_XGB:
    reg_models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

reg_results = []
best_reg, best_reg_score = None, -np.inf
for name, m in reg_models.items():
    m.fit(Xtr, ytr)
    preds = m.predict(Xte)
    r2 = r2_score(yte, preds)
    mae = mean_absolute_error(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    reg_results.append({"model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
    print(f"{name:15s}  R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    if r2 > best_reg_score:
        best_reg, best_reg_score = m, r2

print("\nRegression comparison table:")
print(pd.DataFrame(reg_results).to_string(index=False))
joblib.dump(best_reg, os.path.join(MODEL_DIR, "regression_model.pkl"))
print(f"Saved best regressor -> models/regression_model.pkl")

# =========================================================
# CLASSIFICATION: predict VisitMode
# =========================================================
print("\n=== CLASSIFICATION: predicting VisitMode ===")
y_clf_raw = model_df[TARGET_VISITMODE].astype(str)
clf_le = LabelEncoder()
y_clf = clf_le.fit_transform(y_clf_raw)
joblib.dump(clf_le, os.path.join(MODEL_DIR, "visitmode_label_encoder.pkl"))

Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
}
if HAS_XGB:
    clf_models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, eval_metric="mlogloss")

clf_results = []
best_clf, best_clf_score = None, -np.inf
for name, m in clf_models.items():
    m.fit(Xtr, ytr)
    preds = m.predict(Xte)
    acc = accuracy_score(yte, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="weighted", zero_division=0)
    clf_results.append({"model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    print(f"{name:20s}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    if f1 > best_clf_score:
        best_clf, best_clf_score = m, f1

print("\nClassification comparison table:")
print(pd.DataFrame(clf_results).to_string(index=False))

print("\nDetailed report for best model:")
print(classification_report(yte, best_clf.predict(Xte), target_names=clf_le.classes_.astype(str), zero_division=0))

joblib.dump(best_clf, os.path.join(MODEL_DIR, "classification_model.pkl"))
print(f"Saved best classifier -> models/classification_model.pkl")

print("\nAll models + encoders saved in models/. Ready for the Streamlit app.")
