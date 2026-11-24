"""
PROJECT 1: Data Cleaning + Exploratory Analysis
Dataset: Bengaluru House Price Data (Kaggle: "Bengaluru House price data")
File needed: Bengaluru_House_Data.csv  (put it next to this script)

Business question:
    "What drives house prices in Bengaluru, and which areas offer the best value per sq ft?"

Run:  python project1_cleaning_eda.py
Outputs: cleaned_bengaluru.csv + charts in ./charts/
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("Bengaluru_House_Data.csv")
print("Raw shape:", df.shape)
print(df.isna().sum())

# ---------------------------------------------------------------- 2. CLEAN
# 2a. Drop columns with too many missing values / low value
df = df.drop(columns=["society", "availability"])

# 2b. Missing values
df["location"] = df["location"].fillna("Unknown").str.strip()
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(0)
df = df.dropna(subset=["size"])

# 2c. 'size' looks like "2 BHK" or "4 Bedroom" -> extract the number
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(int)

# 2d. 'total_sqft' is messy: "1133 - 1384", "34.46Sq. Meter", "4125Perch"
def sqft_to_number(x):
    x = str(x).strip()
    if "-" in x:  # range -> average
        parts = x.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan  # unit-based values (Sq. Meter, Perch...) dropped for simplicity

df["total_sqft"] = df["total_sqft"].apply(sqft_to_number)
df = df.dropna(subset=["total_sqft"])

# 2e. Price is in lakhs (1 lakh = 100,000 INR) -> create price per sqft in INR
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# 2f. Duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# 2g. Outliers: unrealistic rooms-per-sqft and price per sqft (per-location z-score filter)
df = df[df["total_sqft"] / df["bhk"] >= 300]          # < 300 sqft per bedroom is suspicious
df = df[df["bath"] <= df["bhk"] + 2]                  # more than BHK+2 bathrooms is suspicious


# Per-location mean/std (works on every pandas version); locations with < 10 listings are kept as-is
grp = df.groupby("location")["price_per_sqft"]
loc_mean = grp.transform("mean")
loc_std = grp.transform("std")
loc_n = grp.transform("size")
keep = (loc_n < 10) | (
    (df["price_per_sqft"] > loc_mean - loc_std) & (df["price_per_sqft"] <= loc_mean + loc_std)
)
df = df[keep]

# 2h. Too many rare locations -> group into "Other"
loc_counts = df["location"].value_counts()
rare = loc_counts[loc_counts <= 10].index
df["location_grouped"] = df["location"].where(~df["location"].isin(rare), "Other")

print("Clean shape:", df.shape)
df.to_csv("cleaned_bengaluru.csv", index=False)

# ---------------------------------------------------------------- 3. EDA
print(df[["total_sqft", "bhk", "bath", "price", "price_per_sqft"]].describe())

# Chart 1: distribution of price per sqft
plt.figure(figsize=(8, 5))
plt.hist(df["price_per_sqft"], bins=50, color="#2E86AB", edgecolor="white")
plt.title("Distribution of Price per Sq Ft (INR)")
plt.xlabel("Price per sq ft")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig("charts/p1_price_per_sqft_hist.png", dpi=150)
plt.close()

# Chart 2: top 10 most expensive locations (min 15 listings so it's not noise)
loc_stats = (
    df[df["location_grouped"] != "Other"]
    .groupby("location_grouped")["price_per_sqft"]
    .agg(["median", "count"])
    .query("count >= 15")
    .sort_values("median", ascending=False)
)
loc_stats.head(10).sort_values("median").plot.barh(
    y="median", legend=False, figsize=(8, 5), color="#E4572E"
)
plt.title("Top 10 Most Expensive Locations (median price/sq ft)")
plt.xlabel("INR per sq ft")
plt.tight_layout()
plt.savefig("charts/p1_top_locations.png", dpi=150)
plt.close()

# Chart 3: correlation heatmap (matplotlib only)
corr = df[["total_sqft", "bhk", "bath", "balcony", "price"]].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("charts/p1_correlation.png", dpi=150)
plt.close()

# Chart 4: price vs area, by BHK
plt.figure(figsize=(8, 5))
for bhk in sorted(df["bhk"].unique())[:4]:
    sub = df[df["bhk"] == bhk]
    plt.scatter(sub["total_sqft"], sub["price"], s=8, alpha=0.5, label=f"{bhk} BHK")
plt.xlim(0, 5000)
plt.ylim(0, 500)
plt.xlabel("Total sq ft")
plt.ylabel("Price (lakhs INR)")
plt.title("Price vs Area by BHK")
plt.legend()
plt.tight_layout()
plt.savefig("charts/p1_price_vs_area.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 4. FINDINGS (print for your README)
best_value = (
    loc_stats.sort_values("median").head(5).index.tolist()
)
print("\n=== KEY FINDINGS (edit these into your README) ===")
print("Most expensive area:", loc_stats.index[0], f"(~INR {loc_stats['median'].iloc[0]:,.0f}/sqft)")
print("Cheapest areas with 15+ listings:", best_value)
print("Corr(area, price):", round(corr.loc["total_sqft", "price"], 2))
print("Corr(bath, price):", round(corr.loc["bath", "price"], 2))
