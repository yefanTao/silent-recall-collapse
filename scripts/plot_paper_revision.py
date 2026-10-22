"""Redraw selected manuscript figures from archived numbers; never run experiments.

Original outputs/figures and outputs/validation remain unchanged. Revised PNGs
are saved separately, with checksums of the JSON inputs for provenance.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/validation"
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
USED = {}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 320, "axes.axisbelow": True,
})
RED, BLUE, GREEN = "#b54139", "#2166ac", "#348652"


def load(name):
    path = SRC / (name + ".json")
    USED[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def finish(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=.18, which="major", linewidth=.5)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def degree_summary():
    p = load("indeg_dist")
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9), layout="constrained")
    x = np.arange(3)
    for key, label, color, shift in [("clean", "Clean", BLUE, -.18),
                                     ("collapsed", "+18K sink", RED, .18)]:
        axs[0].bar(x + shift, [p[key][k] for k in ["mean", "p99", "max_indeg"]],
                   width=.34, color=color, label=label)
    axs[0].set(yscale="log", ylabel="Layer-0 in-degree", xticks=x,
               xticklabels=["Mean", "P99", "Max"], title="(a) Degree")
    axs[0].legend(frameon=False, loc="upper left")
    for key, color, shift in [("clean", BLUE, -.18), ("collapsed", RED, .18)]:
        axs[1].bar(np.arange(2) + shift,
                   [100 * p[key][k] for k in ["top0.01pct_edge_share", "top1pct_edge_share"]],
                   width=.34, color=color)
    axs[1].set(yscale="log", ylabel="Incoming edges (%)", xticks=[0, 1],
               xticklabels=["Top 0.01%", "Top 1%"], title="(b) Edge share")
    finish(fig, "indeg_dist.png")


def duplication():
    p = load("dup_vs_dim")
    fig, ax = plt.subplots(figsize=(3.4, 2.25), layout="constrained")
    # Connect points in plotted x order as visual guides, not rank trajectories.
    for key, marker, color, label in [
            ("arm1_duplication", "o", RED, "Duplicate injection"),
            ("arm2_dimensionality", "s", BLUE, "Clean rank sweep")]:
        rows = sorted(p[key], key=lambda r: r["max_indeg"])
        ax.plot([r["max_indeg"] for r in rows], [r["hnsw"] for r in rows],
                marker=marker, color=color, lw=1.3, markersize=3, label=label)
    ax.set(xscale="log", xlabel="Maximum HNSW base-layer in-degree",
           ylabel="HNSW ID recall@10", xlim=(22, 14000), ylim=(.65, 1.025),
           yticks=[.7, .8, .9, 1.0])
    ax.legend(frameon=False, loc="upper right")
    finish(fig, "dup_vs_dim.png")






def entry_capture():
    p = load("diskann_deep")
    rs = p["points"]
    x = [r["sink_frac"] for r in rs]
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.2), layout="constrained")
    axs[0].plot(x, [r["diskann_recall"] for r in rs], "o-", color=GREEN,
                markersize=3, lw=1, label="Vamana recall")
    axs[0].set(ylabel="Recall@10", ylim=(-.04, 1.02), title="(a) Entry-point capture")
    twin = axs[0].twinx()
    twin.plot(x, [r["max_indeg"] for r in rs], "s--", color="#777777",
              markersize=2.5, lw=.8, label="Max in-degree")
    twin.set_ylabel("Maximum in-degree", color="#555555")
    handles, labels = axs[0].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axs[0].legend(handles + twin_handles, labels + twin_labels,
                  frameon=False, loc="center left")
    for key, label, color in [("centroid_sink_cos", "Centroid vs. sink", BLUE),
                               ("medoid_sink_cos", "Medoid vs. sink", GREEN)]:
        axs[1].plot(x, [r[key] for r in rs], "o-", color=color,
                    markersize=3, lw=1, label=label)
    axs[1].set(ylabel="Cosine similarity", xlabel="Injected sink / clean base",
               title="(b) Centroid and entry-point alignment")
    axs[1].legend(frameon=False, loc="lower right")
    for ax in axs:
        ax.set_xscale("symlog", linthresh=.001)
        ax.set_xticks([0, .001, .01, .05], ["0", ".001", ".01", ".05"])
        ax.axvline(.003, color=RED, ls=":", lw=1)
    finish(fig, "diskann_deep.png")


def pareto():
    p, dedup = load("pareto"), load("pareto_dedup")
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
    series = [
        ("Full-HNSW", p["full_hnsw"], "recall", RED, "o", False),
        ("Dedup, single label", dedup["dedup_hnsw"], "recall_singlelabel", "#a67b52", "D", False),
        ("Dedup, multiple labels", dedup["dedup_hnsw"], "recall_multilabel", "#a67b52", "D", True),
        ("Reduced-HNSW", p["reduced_hnsw"], "recall", GREEN, "o", True),
        ("Full-IVF", p["full_ivf"], "recall", BLUE, "s", False),
        ("Reduced-IVFPQ", p["reduced_ivfpq"], "recall", "#7955a1", "^", True)]
    for label, rs, key, color, marker, filled in series:
        r = max(rs, key=lambda r: r[key])
        ax.scatter(r["bytes"] / 1e6, r[key], s=32, marker=marker,
                   facecolors=color if filled else "white", edgecolors=color,
                   label=label, zorder=3)
    ax.set(xscale="log", xlabel="Serialized index size (MB, decimal)",
           ylabel="Own-entity hit@10", ylim=(.68, .91), xlim=(3, 3000))
    ax.legend(frameon=False, loc="lower left", ncol=1)
    finish(fig, "pareto.png")


def synthesis():
    p = load("synthesis_indegree")
    assert len(p["points"]) == p["n_conditions"] == 72
    fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
    groups = sorted({r["source"] for r in p["points"]})
    reds = ["#7f1d1d", "#b54139", "#d97160", "#d59679"]
    for group, color, marker in zip(groups, reds, ["o", "^", "s", "D"]):
        rows = [r for r in p["points"] if r["source"] == group]
        ax.scatter([r["indeg_max"] for r in rows], [r["hnsw"] for r in rows],
                   color=color, marker=marker, s=15, label="HNSW: " + group.replace("synthetic: ", ""),
                   alpha=.85, linewidths=0)
    ax.scatter([r["indeg_max"] for r in p["points"]],
               [r["ivf"] for r in p["points"]], marker="x", color=BLUE,
               s=15, linewidths=.7, label="IVF (all conditions)", alpha=.7)
    ax.set(xscale="log", xlabel="Maximum base-layer in-degree (HNSW)",
           ylabel="Recall@10", ylim=(.4, 1.035))
    ax.legend(frameon=False, loc="lower left", fontsize=6)
    finish(fig, "synthesis_indegree.png")


if __name__ == "__main__":
    degree_summary()
    duplication()
    entry_capture()
    pareto()
    synthesis()
    (OUT / "provenance.json").write_text(json.dumps({
        "source_sha256": USED,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Archived values only; no experiments rerun. See manuscript captions for protocol."
    }, indent=2), encoding="utf-8")
    print(f"Wrote 5 manuscript figures to {OUT}")
