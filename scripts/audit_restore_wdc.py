"""Memory-bounded reconstruction of WDC rows for the audit.

Stage rows in SQLite rather than materializing the entire 26M-offer raw corpus.
Parsing and filters match wdc_full_explore.py / wdc_full_clean.py for fields
used by exp_tie_robustness.py. Does not generate scores or overwrite archives.
"""
import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(data):
    output = data / "wdc_full_cleaned.pkl"
    if output.exists():
        log(f"Already exists: {output}")
        return
    source = data / "offers_english.json.gz"
    if source.stat().st_size != 4076708488:
        raise ValueError("WDC source download incomplete or different release; verify before processing")
    dbpath = data / "audit_raw.sqlite"
    con = sqlite3.connect(dbpath)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA cache_size=-131072")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY,value INTEGER)")
    done = con.execute("SELECT value FROM progress WHERE key='complete'").fetchone()
    if not done:
        if con.execute("SELECT name FROM sqlite_master WHERE name='offers'").fetchone():
            raise RuntimeError("Interrupted staging DB; preserve it and use a new directory or explicitly restart staging")
        con.execute("CREATE TABLE offers (cluster_id TEXT, source_domain TEXT, title TEXT, brand TEXT, description TEXT, price TEXT)")
        batch = []
        lines = 0
        started = time.perf_counter()
        digest = hashlib.sha256()
        with source.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        with gzip.open(source, "rb") as f:
            for line in f:
                lines += 1
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                cid = rec.get("cluster_id", "")
                if not cid:
                    continue
                props = {}
                for p in rec.get("schema.org_properties", []):
                    for key, value in p.items():
                        value = str(value).strip("[] ").strip()
                        if value and value != "nan":
                            props[key.lstrip("/")] = value
                try:
                    domain = urlparse(rec.get("url", "")).netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                except Exception:
                    domain = ""
                batch.append((str(cid), domain, props.get("name", ""), props.get("brand", ""),
                              props.get("description", ""), props.get("price", "")))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO offers VALUES (?,?,?,?,?,?)", batch)
                    con.commit()
                    batch.clear()
                if lines % 500000 == 0:
                    log(f"Parsed {lines:,} lines; {time.perf_counter()-started:.0f}s")
        if batch:
            con.executemany("INSERT INTO offers VALUES (?,?,?,?,?,?)", batch)
        con.execute("INSERT INTO progress VALUES ('complete',?)", (lines,))
        con.commit()
        (data / "audit_source.json").write_text(json.dumps({
            "url": "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/offers_english.json.gz",
            "bytes": source.stat().st_size, "sha256": digest.hexdigest(), "lines": lines,
            "parse_seconds": time.perf_counter() - started,
            "fields": ["cluster_id", "source_domain", "title", "brand", "description", "price"],
            "note": "Same parse rules and row order as original scripts; unused metadata fields omitted"
        }, indent=2), encoding="utf-8")
    log("Building cluster index on disk")
    con.execute("CREATE INDEX IF NOT EXISTS by_cluster ON offers(cluster_id)")
    con.commit()
    log("Selecting >=5 offers and >=2 distinct sources per entity")
    groups = con.execute("SELECT cluster_id,count(*) FROM offers GROUP BY cluster_id "
                         "HAVING count(*)>=5 AND min(source_domain)<>max(source_domain) ORDER BY cluster_id").fetchall()
    sizes = pd.Series([g[1] for g in groups], index=[g[0] for g in groups])
    selected = list(sizes.index)
    if int(sizes.sum()) > 2_000_000:
        selected, total = [], 0
        for cid, count in sizes.sort_values(ascending=False).items():
            if total + count > 2_000_000:
                break
            selected.append(cid)
            total += count
    log(f"Keeping {len(selected):,} entities, {int(sizes.loc[selected].sum()):,} rows")
    con.execute("CREATE TEMP TABLE selected (cluster_id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO selected VALUES (?)", [(cid,) for cid in selected])
    log("Loading filtered rows in original input order")
    query = "SELECT o.* FROM offers o JOIN selected s ON o.cluster_id=s.cluster_id ORDER BY o.rowid"
    df = pd.read_sql_query(query, con)
    con.close()
    for col in ["title", "description", "brand"]:
        df[col + "_clean"] = df[col].fillna("").astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df["price_clean"] = pd.to_numeric(df.price.astype(str).str.replace(",", "", regex=False)
                                       .str.extract(r"(\d+\.?\d*)", expand=False), errors="coerce")
    df.reset_index(drop=True).to_pickle(output)
    log(f"Saved {output}, rows={len(df):,}")


def embed(data, batch):
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    output = data / "embeddings.npy"
    if output.exists():
        log(f"Already exists: {output}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU runtime unavailable; verify before regenerating million-row embeddings")
    df = pd.read_pickle(data / "wdc_full_cleaned.pkl")
    texts = (df.title_clean.fillna("") + " " + df.brand_clean.fillna("") + " "
             + df.description_clean.fillna("")).str.strip().tolist()
    revision = HfApi().model_info("sentence-transformers/all-MiniLM-L6-v2").sha
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=revision, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    log(f"Encoding {len(texts):,} rows on {torch.cuda.get_device_name(0)}, revision={revision}, batch={batch}")
    start = time.perf_counter()
    # Sort globally like SentenceTransformer, then encode multiples of batch size
    # into a mapped output. Avoid retaining a million Python embedding objects.
    order = np.argsort([-model._input_length(text) for text in texts])
    partial = data / "embeddings.partial.npy"
    emb = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                  shape=(len(texts), model.get_sentence_embedding_dimension()))
    chunk_size = batch * 32
    for start_row in range(0, len(texts), chunk_size):
        rows = order[start_row:start_row + chunk_size]
        vectors = model.encode([texts[i] for i in rows], batch_size=batch,
                               normalize_embeddings=True, show_progress_bar=False,
                               convert_to_numpy=True)
        emb[rows] = vectors
        emb.flush()
        if start_row % (chunk_size * 8) == 0:
            log(f"Encoded {min(start_row + len(rows), len(texts)):,}/{len(texts):,}; "
                f"{time.perf_counter()-start:.1f}s")
    shape = list(emb.shape)
    del emb
    partial.replace(output)
    metadata = {"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": revision,
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "batch_size": batch, "n_rows": len(texts), "shape": shape,
                "encoding_chunks": chunk_size, "global_length_sort": True,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "seconds": time.perf_counter()-start,
                "note": "Regenerated embeddings, original embedding revision not archived; not byte-identical reproduction"}
    (data / "audit_embedding.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Saved {output}, {metadata['seconds']:.1f}s")


def warmup(data, batch):
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    texts = []
    with gzip.open(data / "offers_english.json.gz", "rb") as f:
        while len(texts) < 2048:
            rec = json.loads(f.readline())
            props = {}
            for p in rec.get("schema.org_properties", []):
                for key, value in p.items():
                    value = str(value).strip("[] ").strip()
                    if value and value != "nan":
                        props[key.lstrip("/")] = value
            text = " ".join(re.sub(r"\s+", " ", props.get(col, "").lower().strip())
                            for col in ["name", "brand", "description"]).strip()
            texts.append(text)
    revision = HfApi().model_info("sentence-transformers/all-MiniLM-L6-v2").sha
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=revision, device="cuda")
    model.encode(texts[:32], batch_size=32, normalize_embeddings=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    info = {"model_revision": revision, "rows": len(texts), "seconds": elapsed,
            "rows_per_second": len(texts)/elapsed, "batch": batch,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "dimension": vectors.shape[1], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "note": "First 2048 public corpus rows; throughput may differ for full sorted corpus"}
    (data / "audit_gpu_warmup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(json.dumps(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "embed", "warmup"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "wdc_full")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data)
    elif args.stage == "embed":
        embed(args.data, args.batch)
    else:
        warmup(args.data, args.batch)
