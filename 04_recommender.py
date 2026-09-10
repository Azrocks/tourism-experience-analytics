import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

IN_PATH = "data/processed/master_df.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)
ATTRACTION_COL = "Attraction" if "Attraction" in df.columns else "AttractionId"
ATYPE_COL = "AttractionType" if "AttractionType" in df.columns else None
CITY_COL = "CityName" if "CityName" in df.columns else None

print("=== Collaborative Filtering ===")
ratings = df[["UserId", ATTRACTION_COL, "Rating"]].drop_duplicates(subset=["UserId", ATTRACTION_COL])
user_item = ratings.pivot_table(index="UserId", columns=ATTRACTION_COL, values="Rating").fillna(0)
print(f"User-item matrix shape: {user_item.shape}")

n_components = min(20, min(user_item.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item.values)
item_factors = svd.components_.T  


pred_matrix = user_factors @ item_factors.T
pred_df = pd.DataFrame(pred_matrix, index=user_item.index, columns=user_item.columns)

joblib.dump(pred_df, os.path.join(MODEL_DIR, "cf_predicted_ratings.pkl"))
joblib.dump(user_item, os.path.join(MODEL_DIR, "cf_user_item_matrix.pkl"))
print("Saved collaborative filtering artifacts.")

def recommend_cf(user_id, n=5):
    """Recommend attractions the user hasn't rated yet, ranked by predicted rating."""
    if user_id not in pred_df.index:
        return []
    already_rated = user_item.loc[user_id]
    already_rated = set(already_rated[already_rated > 0].index)
    scores = pred_df.loc[user_id].drop(labels=already_rated, errors="ignore")
    return scores.sort_values(ascending=False).head(n).index.tolist()

sample_user = user_item.index[0]
print(f"Sample CF recommendation for user {sample_user}: {recommend_cf(sample_user)}")

print("\n=== Content-Based Filtering ===")
attr_cols = [c for c in [ATYPE_COL, CITY_COL] if c is not None]
attractions = df[[ATTRACTION_COL] + attr_cols].drop_duplicates(subset=[ATTRACTION_COL]).set_index(ATTRACTION_COL)

if attr_cols:
    ohe = OneHotEncoder(handle_unknown="ignore")
    attr_features = ohe.fit_transform(attractions[attr_cols].astype(str)).toarray()
    sim_matrix = cosine_similarity(attr_features)
    sim_df = pd.DataFrame(sim_matrix, index=attractions.index, columns=attractions.index)
    joblib.dump(sim_df, os.path.join(MODEL_DIR, "cb_similarity_matrix.pkl"))
    print("Saved content-based similarity matrix.")

    def recommend_cb(attraction_name, n=5):
        """Recommend attractions similar to a given one the user liked."""
        if attraction_name not in sim_df.index:
            return []
        return sim_df[attraction_name].drop(labels=[attraction_name]).sort_values(ascending=False).head(n).index.tolist()

    sample_attraction = attractions.index[0]
    print(f"Sample CB recommendation similar to '{sample_attraction}': {recommend_cb(sample_attraction)}")
else:
    print("No attraction feature columns found for content-based filtering — check ATYPE_COL/CITY_COL.")

print("\nRecommendation artifacts ready in models/ for the Streamlit app.")
