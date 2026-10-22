"""Encode cleaned WDC text with MiniLM; rows match wdc_full_cleaned.pkl."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data", "wdc_full")
EMB_CACHE = os.path.join(DATA_DIR, "embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    if os.path.exists(EMB_CACHE):
        print(f"{EMB_CACHE} already exists; nothing to do")
        return
    df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl"))
    texts = (df["title_clean"].fillna("") + " "
           + df["brand_clean"].fillna("") + " "
           + df["description_clean"].fillna("")).str.strip().tolist()
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embedding {len(texts):,} texts with {MODEL_NAME} on {dev}")
    model = SentenceTransformer(MODEL_NAME, device=dev)
    emb = model.encode(texts, batch_size=2048, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    np.save(EMB_CACHE, emb)
    print(f"saved {EMB_CACHE} shape {emb.shape}")

if __name__ == "__main__":
    main()
