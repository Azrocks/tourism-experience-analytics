"""
Tourism Experience Analytics — Streamlit App
Run with: streamlit run app.py
Loads models/artifacts saved by 03_train_models.py and 04_recommender.py.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = "models"
DATA_PATH = "data/processed/master_df.csv"

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

    .hero {
        background: linear-gradient(135deg, #0F766E 0%, #134E4A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; font-weight: 700;}
    .hero p {margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem;}

    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 8px 8px 0 0;
        padding: 0 1.2rem;
        font-weight: 600;
    }

    div.stButton > button {
        background-color: #0F766E;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #0B5A54;
        color: white;
    }

    .result-card {
        background-color: #F0FDFA;
        border: 1px solid #99F6E4;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    artifacts = {}
    for name in ["regression_model", "classification_model", "encoders", "features",
                 "visitmode_label_encoder", "cf_predicted_ratings", "cf_user_item_matrix",
                 "cb_similarity_matrix"]:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        artifacts[name] = joblib.load(path) if os.path.exists(path) else None
    return artifacts

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

artifacts = load_artifacts()
df = load_data()

st.markdown("""
<div class="hero">
    <h1>🧳 Tourism Experience Analytics</h1>
    <p>Rating prediction · Visit mode classification · Personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_rating, tab_visitmode, tab_recs = st.tabs(
    ["📊 Overview", "⭐ Predict Rating", "🧑‍🤝‍🧑 Predict Visit Mode", "🎯 Recommendations"]
)

def geo_filtered(frame, **filters):
    """Return frame filtered by the given column=value pairs, ignoring None values."""
    for col, val in filters.items():
        if val is not None and col in frame.columns:
            frame = frame[frame[col] == val]
    return frame

def render_input_grid(features, encoders, key_prefix):
    """
    Renders feature inputs in up to 2 rows.
    Continent -> Region -> Country -> CityName cascade based on the actual data,
    so each dropdown only shows values consistent with the ones picked before it.
    Other features (AttractionType, VisitYear, VisitMonth, ...) render as plain inputs.
    """
    geo_order = [c for c in ["Continent", "Region", "Country", "CityName"] if c in features]
    other_feats = [f for f in features if f not in geo_order]

    inputs = {}

    # Row 1 — geographic cascade
    if geo_order:
        cols = st.columns(len(geo_order))
        working = df
        for i, feat in enumerate(geo_order):
            options = sorted(working[feat].dropna().unique().tolist())
            if not options:
                options = list(encoders[feat].classes_) if feat in encoders else []
            with cols[i]:
                choice = st.selectbox(feat, options, key=f"{key_prefix}_{feat}")
            inputs[feat] = choice
            working = geo_filtered(working, **{feat: choice})

    # Row 2 — remaining features (attraction type, year, month, ...)
    if other_feats:
        cols = st.columns(len(other_feats))
        for i, feat in enumerate(other_feats):
            with cols[i]:
                if feat in encoders:
                    options = list(encoders[feat].classes_)
                    inputs[feat] = st.selectbox(feat, options, key=f"{key_prefix}_{feat}")
                else:
                    default = int(df[feat].median()) if feat in df.columns else 0
                    inputs[feat] = st.number_input(feat, value=default, key=f"{key_prefix}_{feat}")

    return inputs

def encode_row(features, encoders, inputs):
    row = []
    for feat in features:
        if feat in encoders:
            row.append(encoders[feat].transform([inputs[feat]])[0])
        else:
            row.append(inputs[feat])
    return row

with tab_overview:
    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", f"{len(df):,}")
    c2.metric("Unique users", f"{df['UserId'].nunique():,}" if "UserId" in df.columns else "-")
    c3.metric("Avg rating", round(df["Rating"].mean(), 2) if "Rating" in df.columns else "-")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        if "Continent" in df.columns:
            st.markdown("**Visits by Continent**")
            st.bar_chart(df["Continent"].value_counts())
    with col2:
        if "AttractionType" in df.columns:
            st.markdown("**Average Rating by Attraction Type**")
            st.bar_chart(df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False))

    if "Attraction" in df.columns:
        st.markdown("**Top 10 Most-Visited Attractions**")
        st.dataframe(df["Attraction"].value_counts().head(10), use_container_width=True)

with tab_rating:
    st.subheader("Predict the rating a user might give")
    if artifacts["regression_model"] is None:
        st.warning("Regression model not found — run 03_train_models.py first.")
    else:
        features = artifacts["features"]
        encoders = artifacts["encoders"]
        inputs = render_input_grid(features, encoders, key_prefix="reg")

        if st.button("Predict Rating"):
            row = encode_row(features, encoders, inputs)
            pred = artifacts["regression_model"].predict([row])[0]
            st.markdown(f"""
            <div class="result-card">
                <span style="font-size:0.9rem;color:#0F766E;font-weight:600;">PREDICTED RATING</span><br>
                <span style="font-size:2rem;font-weight:700;">{pred:.2f} / 5</span>
            </div>
            """, unsafe_allow_html=True)

with tab_visitmode:
    st.subheader("Predict likely visit mode")
    if artifacts["classification_model"] is None:
        st.warning("Classification model not found — run 03_train_models.py first.")
    else:
        features = artifacts["features"]
        encoders = artifacts["encoders"]
        inputs = render_input_grid(features, encoders, key_prefix="clf")

        if st.button("Predict Visit Mode"):
            row = encode_row(features, encoders, inputs)
            model = artifacts["classification_model"]
            le = artifacts["visitmode_label_encoder"]
            pred_idx = model.predict([row])[0]
            pred_label = le.inverse_transform([pred_idx])[0]

            st.markdown(f"""
            <div class="result-card">
                <span style="font-size:0.9rem;color:#0F766E;font-weight:600;">PREDICTED VISIT MODE</span><br>
                <span style="font-size:2rem;font-weight:700;">{pred_label}</span>
            </div>
            """, unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([row])[0]
                proba_df = pd.DataFrame({"VisitMode": le.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
                st.write("")
                st.markdown("**Probability by visit mode**")
                st.bar_chart(proba_df.set_index("VisitMode"))

with tab_recs:
    st.subheader("Personalized attraction recommendations")
    method = st.radio(
        "Method", ["Collaborative Filtering (existing user)", "Content-Based (similar to an attraction)"],
        horizontal=True,
    )

    if method.startswith("Collaborative") and artifacts["cf_predicted_ratings"] is not None:
        pred_df = artifacts["cf_predicted_ratings"]
        user_item = artifacts["cf_user_item_matrix"]
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.selectbox("Select User ID", pred_df.index.tolist())
        with col2:
            n = st.slider("Number of recommendations", 3, 10, 5)
        if st.button("Get Recommendations", key="cf_btn"):
            already_rated = user_item.loc[user_id]
            already_rated = set(already_rated[already_rated > 0].index)
            scores = pred_df.loc[user_id].drop(labels=already_rated, errors="ignore")
            top = scores.sort_values(ascending=False).head(n)
            st.dataframe(
                top.reset_index().rename(columns={0: "Predicted Score", "index": "Attraction"}),
                use_container_width=True, hide_index=True,
            )

    elif method.startswith("Content") and artifacts["cb_similarity_matrix"] is not None:
        sim_df = artifacts["cb_similarity_matrix"]
        col1, col2 = st.columns(2)
        with col1:
            attraction = st.selectbox("Pick an attraction you liked", sim_df.index.tolist())
        with col2:
            n = st.slider("Number of recommendations", 3, 10, 5, key="cb_slider")
        if st.button("Get Recommendations", key="cb_btn"):
            top = sim_df[attraction].drop(labels=[attraction]).sort_values(ascending=False).head(n)
            st.dataframe(
                top.reset_index().rename(columns={0: "Similarity Score", "index": "Attraction"}),
                use_container_width=True, hide_index=True,
            )
    else:
        st.warning("Recommendation artifacts not found — run 04_recommender.py first.")

st.divider()
st.caption("Tourism Experience Analytics · Regression, Classification & Recommendation System")
