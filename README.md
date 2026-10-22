# Reproducing the experiments

Run commands from this directory. Use Python 3.12 for the main experiments.

## Install and test

Create a virtual environment and install PyTorch for your CPU/CUDA configuration, then:

```bash
python -m pip install -r requirements-audit.txt
python -m pip install "./faiss_contrib[plot,test]"
python -m pytest --import-mode=importlib faiss_contrib/test_entity_index.py -q
entity-index-demo --dataset fashion-mnist --dup-rate 0.5 --no-plot
entity-index-demo --dataset glove-25 --dup-rate 0.5 --no-plot
```

The two demos download public vectors on first use and run on CPU.

## Redraw the paper figures

```bash
python scripts/plot_paper_revision.py
python scripts/run_archived_experiment.py exp_synthesis --output outputs/rerun/synthesis
```

These commands use the included numerical results. The five figures and their input
hashes are written to `outputs/figures/`. Archived measurements are in
`outputs/validation/` and `outputs/pvldb_audit/`.

## Prepare data for new runs

Bulk datasets and model weights are downloaded separately:

| Dataset | Preparation, in order |
| --- | --- |
| Full WDC | `bash scripts/wdc_full_download.sh`; `python scripts/audit_restore_wdc.py --stage prepare`; `python scripts/audit_restore_wdc.py --stage embed --batch 256` (CUDA) |
| WDC-Computers | `bash scripts/step0_download.sh`; `python scripts/step0_explore.py`; `python scripts/step0_clean.py` |
| MusicBrainz | Download the [provider file](https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-200-A01.csv.dapo) into `data/musicbrainz/`, then `python scripts/musicbrainz_prepare.py` |
| MS MARCO | Install `datasets`; `exp_msmarco` downloads and embeds the corpus on first run |
| SIFT | Save the [training shard](https://huggingface.co/datasets/open-vdb/sift-128-euclidean/resolve/main/train/train-00001-of-00001.parquet) as `data/sift/train-00001-of-00001.parquet` |
| GIST | Save the [first training shard](https://huggingface.co/datasets/open-vdb/gist-960-euclidean/resolve/main/train/train-00001-of-00003.parquet) as `data/gist/train-00001-of-00003.parquet` |

WDC source and restored-embedding metadata are in `artifact_provenance/wdc/`.
The original WDC embedding revision was not recorded; restored embeddings are a
separate replication. Graph construction and timing also vary with runtime and threading.

## Run experiments

The wrapper accepts an `exp_*.py` module name without `.py`, preserves its numerical
defaults, and redirects results to a new directory. For example:

```bash
python scripts/run_archived_experiment.py exp_dup_vs_dim --output outputs/rerun/dupdim
python scripts/run_archived_experiment.py exp_tie_robustness --output outputs/rerun/wdc_ties --set ONLY=wdc_sweep,real_dedup
python scripts/audit_pvldb.py --suite jitter --output outputs/rerun/jitter
python scripts/exp_nsg_build.py --output outputs/rerun/nsg --timeout-seconds 60
```

The NSG timeout is a bounded observation, not proof of nontermination.

| Paper result | Producers (names following `exp_`) |
| --- | --- |
| Figure 1 | `indeg_dist` |
| Figure 2 | `regime`, `megahub`, `hub_mechanism`, `wdc_controls`, `wdc_collateral`, `pure_ann`, `small_exact_groups`, `tie_robustness`, then `synthesis` |
| Figure 3 | `dup_vs_dim` |
| Figure 4 | `diskann_deep`, via the native wrapper below |
| Figure 5 | `pareto`, `pareto_dedup` |
| Tables 1, 2, 4, 5 (natural), 6 | `tie_robustness`; prepare `computers`, `msmarco`, and `embedding_interaction` caches first |
| Table 3 | `audit_pvldb.py --suite jitter`; `audit_jitter_precision.py` |
| Table 5 (injection) | `msmarco` |
| Table 7 | `audit_pareto_runtime.py`, three construction seeds |
| Table 8 | `scale_clustered` |

For Figure 2's separate WDC-control corrections, run `tie_robustness` with
`--set ONLY=wdc_controls --set OUTNAME=tie_wdc_controls.json`.
The synthesis and plot scripts read `outputs/validation/`: to plot new results,
work in a separate copy and put the corresponding new JSON files there.
Preserve each experiment's metric: clean-base ID overlap, full-gallery tie-aware
recall, and entity hit rates measure different outcomes. Some natural-corpus
experiments include self matches and resample queries after reduction.

For the five-model matrix, prepare WDC and MusicBrainz, then run the commands below
for each dataset (`wdc`, `musicbrainz`). Repeat with seed 43 and label `rerun43`:

```bash
python scripts/audit_c4_runtime.py --dataset wdc --run-label rerun42 --seed 42 --revisions-file outputs/pvldb_audit/2026-10-10/c4_verified/wdc/model-revisions.json
python scripts/audit_c4_runtime.py --dataset musicbrainz --run-label rerun42 --seed 42 --revisions-file outputs/pvldb_audit/2026-10-10/c4_verified/musicbrainz/model-revisions.json
python scripts/analyze_c4_entities.py --run-label rerun42 --family-size 80
```

For Table 7, run `audit_pareto_runtime.py` with seeds 0, 1, 2 and distinct
`--output` directories. Summarize them, in seed order, with
`summarize_pareto_runtime.py --runs DIR0 DIR1 DIR2 --output outputs/rerun/pareto_summary.json`.

For native graph runs, use a separate Python 3.11 environment:

```bash
python -m pip install -r requirements-native.txt
python scripts/audit_native_runtime.py --experiment hnswlib_check --seed 0 --run-label rerun_native
python scripts/audit_native_runtime.py --experiment diskann --seed 0 --run-label rerun_native
python scripts/audit_native_runtime.py --experiment diskann_deep --seed 0 --run-label rerun_native
```

Repeat with seeds 1 and 2. Use new output directories or labels for new runs.
