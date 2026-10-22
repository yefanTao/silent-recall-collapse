"""
Step 0b: Clean the WDC computers subset.
- Normalize text fields, prices, brands.
- Filter to labeled records with cluster_size >= 3.
- Save cleaned dataframe.

Expects: computers_raw.pkl (from step0_explore.py)
Produces: computers_cleaned.pkl
"""
import pandas as pd
import numpy as np
import re, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_products")

df = pd.read_pickle(os.path.join(DATA_DIR, "computers_raw.pkl"))
print(f"Loaded {len(df)} raw computer offers")

# ------------------------------------------------------------------
# 1. Text cleaning
# ------------------------------------------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())

df["title_clean"] = df["title"].apply(clean_text)
df["description_clean"] = df["description"].apply(clean_text)
df["brand_clean"] = df["brand"].apply(clean_text)

# ------------------------------------------------------------------
# 2. Price normalization
# ------------------------------------------------------------------
def clean_price(price):
    if pd.isna(price):
        return np.nan
    nums = re.findall(r"[\d]+\.?\d*", str(price).replace(",", ""))
    return float(nums[0]) if nums else np.nan

df["price_clean"] = df["price"].apply(clean_price)

# ------------------------------------------------------------------
# 3. Brand normalization
# ------------------------------------------------------------------
BRAND_MAP = {
    "hewlett-packard": "hp", "hewlett packard": "hp", "hp enterprise": "hp",
    "hp inc": "hp", "hp inc.": "hp",
    "dell": "dell", "dell inc": "dell",
    "lenovo": "lenovo", "apple": "apple", "mac": "apple",
    "asus": "asus", "acer": "acer", "microsoft": "microsoft",
    "samsung": "samsung", "toshiba": "toshiba", "sony": "sony",
    "msi": "msi", "razer": "razer", "lg": "lg", "intel": "intel",
}

def normalize_brand(brand):
    b = clean_text(brand)
    if not b:
        return "unknown"
    for key, val in BRAND_MAP.items():
        if key in b:
            return val
    return b

df["brand_norm"] = df["brand_clean"].apply(normalize_brand)

# ------------------------------------------------------------------
# 4. Filter: cluster_size >= 3
# ------------------------------------------------------------------
sizes = df.groupby("cluster_id").size()
valid = sizes[sizes >= 3].index
df_valid = df[df["cluster_id"].isin(valid)].copy().reset_index(drop=True)

print(f"\nAfter filtering (cluster_size >= 3):")
print(f"  Offers:   {len(df_valid)}")
print(f"  Entities: {df_valid['cluster_id'].nunique()}")
print(f"  Avg offers/entity: {df_valid.groupby('cluster_id').size().mean():.1f}")

# Quick field stats on cleaned data
print(f"\nField coverage in cleaned data:")
for col in ["title_clean", "brand_norm", "price_clean", "description_clean"]:
    if col == "price_clean":
        filled = df_valid[col].notna().sum()
    else:
        filled = (df_valid[col].str.strip() != "").sum() if col != "price_clean" else 0
    print(f"  {col:25s}  {filled/len(df_valid):6.2%} filled")

out = os.path.join(DATA_DIR, "computers_cleaned.pkl")
df_valid.to_pickle(out)
print(f"\nSaved: {out}")
