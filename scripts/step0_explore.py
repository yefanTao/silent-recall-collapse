"""
Step 0a: Explore the WDC LSPC v2 offers corpus — computers subset.

The main corpus is offers_corpus_english_v2.json.gz with schema:
  id, title, description, brand, price, specTableContent,
  keyValuePairs, category, cluster_id, identifiers
"""
import gzip, json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_products")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CORPUS = os.path.join(DATA_DIR, "offers_corpus_english_v2.json.gz")

# ------------------------------------------------------------------
# 1. Load & filter to computers (with progress)
# ------------------------------------------------------------------
print("Loading & filtering to Computers_and_Accessories ...")

rows = []
total_count = 0

with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
    for line in f:
        total_count += 1
        if total_count % 500_000 == 0:
            print(f"\r  {total_count:>12,} lines read | "
                  f"{len(rows):,} computers kept", end="", flush=True)

        rec = json.loads(line)
        if rec.get("category") == "Computers_and_Accessories":
            rows.append(rec)

print(f"\r  {total_count:>12,} lines read | "
      f"{len(rows):,} computers kept  [DONE]")
print()

df = pd.DataFrame(rows)
print(f"\nTotal offers (all categories): {total_count:,}")
print(f"Computers offers:              {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# ------------------------------------------------------------------
# 2. Basic statistics
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("BASIC STATISTICS — Computers")
print("=" * 60)
print(f"Total offers:    {len(df):,}")
print(f"Unique entities: {df['cluster_id'].nunique():,}")

sizes = df.groupby("cluster_id").size()
print(f"\nOffers per entity:")
print(sizes.describe())

# ------------------------------------------------------------------
# 3. Field missing rates
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FIELD MISSING RATES")
print("=" * 60)
for col in ["title", "description", "brand", "price",
            "specTableContent", "keyValuePairs",
            "cluster_id", "identifiers"]:
    if col not in df.columns:
        continue
    na = df[col].isna().sum()
    empty = 0
    if df[col].dtype == object:
        empty = (df[col].astype(str).str.strip().isin(["", "None", "nan"])).sum()
    missing = max(na, empty)
    print(f"  {col:25s}  {missing/len(df):6.2%} missing  ({missing:,}/{len(df):,})")

# ------------------------------------------------------------------
# 4. Entity size distribution
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(sizes.clip(upper=50), bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Offers per entity (clipped at 50)")
axes[0].set_ylabel("Count")
axes[0].set_title("Entity Size Distribution — Computers")

sizes.value_counts().sort_index().plot(loglog=True, marker=".", ax=axes[1], linewidth=0)
axes[1].set_xlabel("Entity size")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Entity Size (log-log)")

plt.tight_layout()
out = os.path.join(FIG_DIR, "entity_size_dist.png")
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# ------------------------------------------------------------------
# 5. Sample records
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("SAMPLE RECORDS (5 random)")
print("=" * 60)
for _, row in df.sample(5, random_state=42).iterrows():
    d = row.dropna().to_dict()
    for k in d:
        d[k] = str(d[k])[:120]
    print(json.dumps(d, indent=2))
    print("---")

# Save computers subset
out_pkl = os.path.join(DATA_DIR, "computers_raw.pkl")
df.to_pickle(out_pkl)
print(f"\nSaved computers subset: {out_pkl} ({len(df):,} rows)")
