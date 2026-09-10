import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "transaction": "Transaction.xlsx",
    "user": "User.xlsx",
    "city": "City.xlsx",
    "type": "Type.xlsx",
    "mode": "Mode.xlsx",
    "continent": "Continent.xlsx",
    "country": "Country.xlsx",
    "region": "Region.xlsx",
    "item": "Item.xlsx",
}

def load(name):
    path = os.path.join(RAW_DIR, FILES[name])
    df = pd.read_excel(path)
    print(f"{name:12s} shape={df.shape}  cols={list(df.columns)}")
    return df

print("=== Loading raw tables ===")
tx      = load("transaction")
user    = load("user")
city    = load("city")
atype   = load("type")
mode    = load("mode")
cont    = load("continent")
country = load("country")
region  = load("region")
item    = load("item")

def strip_title(s):
    if pd.api.types.is_string_dtype(s):
        return s.str.strip().str.title()
    return s

def clean_ids(df, id_cols):
    """Ensure id columns are consistent int type where possible."""
    for c in id_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

for df in [tx, user, city, atype, mode, cont, country, region, item]:
    for c in df.select_dtypes(include="object").columns:
        df[c] = strip_title(df[c])

print("\n=== Missing values before cleaning ===")
for name, df in [("transaction", tx), ("user", user), ("item", item)]:
    print(name, df.isnull().sum().to_dict())

if "Rating" in tx.columns:
    tx = tx.dropna(subset=["Rating"])
    tx["Rating"] = pd.to_numeric(tx["Rating"], errors="coerce")
    tx = tx[(tx["Rating"] >= 1) & (tx["Rating"] <= 5)]   

visit_mode_col = "VisitMode" if "VisitMode" in tx.columns else "VisitModeId"
tx = tx.dropna(subset=[visit_mode_col])

if "VisitYear" in tx.columns and "VisitMonth" in tx.columns:
    tx["VisitYear"] = pd.to_numeric(tx["VisitYear"], errors="coerce").astype("Int64")
    tx["VisitMonth"] = pd.to_numeric(tx["VisitMonth"], errors="coerce").astype("Int64")
    tx = tx.dropna(subset=["VisitYear", "VisitMonth"])

tx = tx.drop_duplicates()

print(f"\nTransaction shape after cleaning: {tx.shape}")

if "CityName" in city.columns:
    city["CityName"] = city["CityName"].str.strip().str.title()
    city = city.drop_duplicates(subset=["CityName", "CountryId"]) if "CountryId" in city.columns else city.drop_duplicates()

print("\n=== Merging tables ===")
df = tx.merge(user, on="UserId", how="left")

if "CityId" in df.columns and "CityId" in city.columns:
    df = df.merge(city, on="CityId", how="left", suffixes=("", "_city"))

if "CountryId" in df.columns and "CountryId" in country.columns:
    df = df.merge(country, on="CountryId", how="left", suffixes=("", "_country"))

if "RegionId" in df.columns and "RegionId" in region.columns:
    df = df.merge(region, on="RegionId", how="left", suffixes=("", "_region"))

if "ContinentId" in df.columns and "ContinentId" in cont.columns:
    df = df.merge(cont, on="ContinentId", how="left", suffixes=("", "_continent"))

if "AttractionId" in df.columns and "AttractionId" in item.columns:
    df = df.merge(item, on="AttractionId", how="left", suffixes=("", "_item"))

if "AttractionTypeId" in df.columns and "AttractionTypeId" in atype.columns:
    df = df.merge(atype, on="AttractionTypeId", how="left", suffixes=("", "_type"))

if "VisitModeId" in df.columns and "VisitModeId" in mode.columns:
    df = df.merge(mode, on="VisitModeId", how="left", suffixes=("", "_mode"))

print(f"Master shape: {df.shape}")
print("\n=== Missing values after merge (top 15) ===")
print(df.isnull().sum().sort_values(ascending=False).head(15))

protected = {"Rating", visit_mode_col}
for c in df.select_dtypes(include="object").columns:
    if c not in protected:
        df[c] = df[c].fillna("Unknown")

out_path = os.path.join(OUT_DIR, "master_df.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned master dataset -> {out_path}  shape={df.shape}")

with open(os.path.join(OUT_DIR, "cleaning_log.txt"), "w") as f:
    f.write("DATA CLEANING SUMMARY\n")
    f.write(f"Transactions after cleaning: {tx.shape[0]} rows\n")
    f.write(f"Master merged dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write("Steps: stripped/title-cased text fields, dropped rows with missing/invalid Rating,\n")
    f.write("dropped rows with missing VisitMode, removed duplicate transactions,\n")
    f.write("left-joined user/city/country/region/continent/item/type/mode lookup tables,\n")
    f.write("filled remaining categorical gaps with 'Unknown'.\n")
print("Wrote cleaning_log.txt (paste into your report).")
