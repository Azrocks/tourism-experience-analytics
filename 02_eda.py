"""
STEP 2 — Exploratory Data Analysis
Run after 01_clean_and_merge.py. Saves PNG charts to reports/figures/
for your documentation, and prints the key numeric insights to paste into your report.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
IN_PATH = "data/processed/master_df.csv"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)
print(f"Loaded master_df: {df.shape}")

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()
    print(f"Saved {name}")

# ---- Guess column names (adjust if yours differ) ----
COL_CONTINENT = "Continent" if "Continent" in df.columns else None
COL_COUNTRY   = "Country" if "Country" in df.columns else None
COL_ATYPE     = "AttractionType" if "AttractionType" in df.columns else None
COL_ATTRACTION= "Attraction" if "Attraction" in df.columns else None
COL_VISITMODE = "VisitMode" if "VisitMode" in df.columns else None
COL_RATING    = "Rating"
COL_MONTH     = "VisitMonth" if "VisitMonth" in df.columns else None

# 1. Users by continent
if COL_CONTINENT:
    plt.figure(figsize=(8,5))
    df[COL_CONTINENT].value_counts().plot(kind="bar", color="teal")
    plt.title("Visits by Continent"); plt.ylabel("Number of visits")
    savefig("01_visits_by_continent.png")

# 2. Top 10 countries
if COL_COUNTRY:
    plt.figure(figsize=(9,5))
    df[COL_COUNTRY].value_counts().head(10).plot(kind="barh", color="steelblue")
    plt.title("Top 10 Countries by Visits"); plt.xlabel("Number of visits")
    savefig("02_top_countries.png")

# 3. Attraction type popularity + avg rating
if COL_ATYPE:
    agg = df.groupby(COL_ATYPE)[COL_RATING].agg(["mean", "count"]).sort_values("count", ascending=False)
    print("\n=== Attraction type: avg rating & visit count ===")
    print(agg.head(15))
    plt.figure(figsize=(9,5))
    agg["mean"].head(15).plot(kind="bar", color="orange")
    plt.title("Average Rating by Attraction Type"); plt.ylabel("Avg Rating")
    savefig("03_avgrating_by_type.png")

# 4. Top 10 attractions by visit count (min threshold to avoid noise)
if COL_ATTRACTION:
    counts = df[COL_ATTRACTION].value_counts()
    top = counts[counts >= 5].head(10)
    plt.figure(figsize=(9,5))
    top.plot(kind="barh", color="seagreen")
    plt.title("Top 10 Most-Visited Attractions (min 5 visits)")
    savefig("04_top_attractions.png")

# 5. VisitMode vs Continent heatmap
if COL_VISITMODE and COL_CONTINENT:
    ct = pd.crosstab(df[COL_CONTINENT], df[COL_VISITMODE])
    plt.figure(figsize=(9,6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Visit Mode vs Continent")
    savefig("05_visitmode_vs_continent.png")

# 6. Seasonality
if COL_MONTH:
    plt.figure(figsize=(9,5))
    df[COL_MONTH].value_counts().sort_index().plot(kind="line", marker="o")
    plt.title("Visits by Month"); plt.xlabel("Month"); plt.ylabel("Number of visits")
    savefig("06_seasonality.png")

# 7. Rating distribution
plt.figure(figsize=(7,5))
df[COL_RATING].plot(kind="hist", bins=5, color="purple", edgecolor="black")
plt.title("Rating Distribution"); plt.xlabel("Rating")
savefig("07_rating_distribution.png")

# 8. Correlation heatmap on numeric columns
num_df = df.select_dtypes(include="number")
if num_df.shape[1] > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix (numeric features)")
    savefig("08_correlation_matrix.png")

print("\nEDA complete. Charts are in reports/figures/ — pull these + 1-2 sentence")
print("takeaways per chart directly into your report/presentation.")
