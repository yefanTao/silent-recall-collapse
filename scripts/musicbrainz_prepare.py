"""
Build data/musicbrainz/musicbrainz_clean.pkl from the raw Leipzig ER benchmark file.

Source: the "MusicBrainz 200K" corpus (musicbrainz-200-A01.csv.dapo) from the
Leipzig DB-group benchmark datasets for entity resolution:
  https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
(200K song records from 5 sources; CID is the gold cluster id.)

Place the downloaded file at data/musicbrainz/musicbrainz-200-A01.csv.dapo and run:
  python scripts/musicbrainz_prepare.py

Output schema (consumed by exp_c4_generalize.py):
  all raw columns (as strings) + record_id (=TID), cluster_id (=CID),
  source (=SourceID), and text = lowercased "title artist album year language
  number length" with empty fields dropped and whitespace collapsed.
NOTE: the raw file contains literal "null" strings; they are kept verbatim
(keep_default_na=False), matching the corpus used in the paper. This script
regenerates the paper's musicbrainz_clean.pkl byte-for-byte on the text column
(verified on all 193,750 rows).
"""
import os
import pickle
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DDIR = os.path.join(ROOT, "data", "musicbrainz")
RAW  = os.path.join(DDIR, "musicbrainz-200-A01.csv.dapo")
OUT  = os.path.join(DDIR, "musicbrainz_clean.pkl")

TEXT_FIELDS = ["title", "artist", "album", "year", "language", "number", "length"]

def main():
    if not os.path.exists(RAW):
        raise SystemExit(
            f"missing {RAW}\nDownload musicbrainz-200-A01.csv.dapo from the Leipzig "
            "DB-group ER benchmark page (see module docstring) and place it there.")
    # keep_default_na=False: the raw file uses literal "null" strings that must
    # survive into the text column (pandas would otherwise turn them into NaN).
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    df["record_id"] = df["TID"]
    df["cluster_id"] = df["CID"]
    df["source"] = df["SourceID"]

    def mk_text(row):
        parts = [str(row[f]).strip() for f in TEXT_FIELDS if str(row[f]).strip() != ""]
        return " ".join(" ".join(parts).lower().split())

    df["text"] = df.apply(mk_text, axis=1)
    pickle.dump(df, open(OUT, "wb"))
    print(f"wrote {OUT}: {len(df)} rows, {df['cluster_id'].nunique()} clusters")

if __name__ == "__main__":
    main()
