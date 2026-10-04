"""
Per-generation diagnostics for the pure-rejection resampling constraint
handler (Resampling / Resampling_AL sim_types).

Structurally mirrors ``arnold_diagnostics.py``, but for the rejection path.
The strategy fills ``strategy.resample_diag_buffer`` with one record per
INFEASIBLE DRAW consumed in ``resample_infeasibles_rejection`` - the initial
infeasible offspring PLUS every rejected redraw - and stashes slot-level
counts in ``strategy._last_resample_stats``.  This module

* drains that buffer at end-of-generation (clearing it), tagging the counts
  with the generation number,
* appends to a stable on-disk CSV (``resample_per_gen.csv``) - written EVERY
  generation so the run is crash-safe and analysable mid-flight,
* exposes plot functions meant to be called ONCE after the run completes
  (the CSV holds the full history, so plotting is a stateless
  read-from-disk side-effect).

"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Reuse the Chocat/Arnold constraint-label mapping so the heatmap rows read
# as physical-variable bounds (e.g. "p4 <= p4_max") and pair each variable's
# lower/upper bound on adjacent rows - identical to the Arnold heatmap.
from cht_diagnostics import _displayed_constraints


# CSV schema

PER_GEN_FIELDS = [
    "generation",
    "n_infeasible_slots",   # offspring slots infeasible on their FIRST draw
    "n_infeasible_draws",   # EVERY infeasible draw (initial + rejected redraws)
    "n_redraws",            # total resample draws performed this gen
    "n_unresolved",         # slots still infeasible after the cap (dropped)
    "n_lambda",
    "n_constraints",        # length of the raw g vector (for the heatmap)
    # JSON list (length n_constraints): how many infeasible DRAWS violated
    # each constraint j this generation.  Sums active_js across every record.
    "per_constraint_violation_count",
]


# Buffer drain -> CSV append

def _jsonify(v):
    """JSON-encode list-valued cells; pass scalars through; "" for None."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v


def _summarise(gen: int, records: list[dict], stats: dict,
               n_lambda: int, n_constraints: int) -> dict:
    """Reduce a generation's per-draw records to one summary row.

    ``per_constraint_violation_count`` aggregates the heatmap source: for
    every infeasible-draw record, each index in ``active_js`` increments its
    counter.  Length is fixed at ``n_constraints`` so constraints never
    violated this gen still occupy a (zero) row in the heatmap.
    """
    counts = np.zeros(max(n_constraints, 0), dtype=int)
    for r in records:
        for j in (r.get("active_js") or []):
            if 0 <= int(j) < len(counts):
                counts[int(j)] += 1

    return {
        "generation":                     gen,
        "n_infeasible_slots":             int(stats.get("n_infeasible_slots", 0)),
        "n_infeasible_draws":             len(records),
        "n_redraws":                      int(stats.get("n_redraws", 0)),
        "n_unresolved":                   int(stats.get("n_unresolved", 0)),
        "n_lambda":                       int(n_lambda),
        "n_constraints":                  int(n_constraints),
        "per_constraint_violation_count": counts.tolist(),
    }


def append_per_gen_row(csv_path: Path, gen: int, records: list[dict],
                       stats: dict, n_lambda: int,
                       n_constraints: int) -> dict:
    """Append the per-gen summary row, writing a header on first call.

    Returns the (un-serialised) row dict.
    """
    csv_path = Path(csv_path)
    row = _summarise(gen, records, stats, n_lambda, n_constraints)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_GEN_FIELDS)
        if is_new:
            writer.writeheader()
        out = dict(row)
        out["per_constraint_violation_count"] = _jsonify(
            row["per_constraint_violation_count"])
        writer.writerow(out)
    return row


def drain_and_persist(strategy, gen: int, out_dir: Path,
                      n_lambda: int) -> list[dict]:
    """One-call helper invoked by main.py after each generation.

    Drains ``strategy.resample_diag_buffer`` (clearing it), reads the paired
    ``strategy._last_resample_stats``, appends the per-gen CSV row, and
    returns the drained records.  Cheap when the buffer is empty (non-
    resample sim_types), so it can be called unconditionally.

    The header is written eagerly on the first call regardless of whether
    records exist, so post-hoc tooling can tell "resample mode ran but
    produced no infeasibles" from "wrong sim_type / no file".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(getattr(strategy, "resample_diag_buffer", []))
    if hasattr(strategy, "resample_diag_buffer"):
        strategy.resample_diag_buffer.clear()
    stats = dict(getattr(strategy, "_last_resample_stats", {}) or {})

    # n_constraints sizes the per-constraint heatmap vector.  Fall back to
    # the max index actually seen if the strategy didn't expose it.
    n_constraints = getattr(strategy, "n_constraints", None)
    if not n_constraints:
        seen = [int(j) for r in records for j in (r.get("active_js") or [])]
        n_constraints = (max(seen) + 1) if seen else 0

    append_per_gen_row(out_dir / "resample_per_gen.csv", gen, records,
                       stats, n_lambda, n_constraints)
    return records


# CSV read helpers

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_list(s):
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


# Plot 1 - per-constraint violation heatmap (raw counts)

def plot_constraint_violation_heatmap(out_dir: Path) -> Path | None:
    """Heatmap of infeasible-draw counts, gen x constraint.

    Source: ``per_constraint_violation_count`` from resample_per_gen.csv -
    the number of infeasible DRAWS (initial offspring + every rejected
    redraw) that violated each constraint, per generation.  Rows are
    collapsed/relabelled to the 12 physical-variable bounds via the shared
    ``_displayed_constraints`` mapping (identical layout to the Arnold
    heatmap, so the two figures can be compared side by side).
    """
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "resample_per_gen.csv")
    if not rows:
        return None

    gens, counts_by_gen = [], []
    n_raw = None
    for r in rows:
        cc = _to_list(r.get("per_constraint_violation_count"))
        if cc is None:
            continue
        gens.append(int(r["generation"]))
        counts_by_gen.append(np.asarray(cc, dtype=float))
        if n_raw is None:
            n_raw = len(cc)
    if not gens or not n_raw:
        return None

    indices, labels = _displayed_constraints(n_raw)
    n_disp = len(indices)
    mat = np.zeros((n_disp, len(gens)), dtype=float)
    for col, cc in enumerate(counts_by_gen):
        mat[:, col] = cc[indices]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[gens[0], gens[-1], -0.5, n_disp - 0.5],
                   cmap="magma", vmin=0.0)
    plt.colorbar(im, ax=ax, label="# infeasible draws violating constraint")
    ax.set_yticks(np.arange(n_disp))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Resampling: per-constraint violation count (all draws)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Constraint")
    fig.tight_layout()
    fig_path = out_dir / "resample_constraint_heatmap.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# Plot 2 - infeasible draws / redraws per generation

def plot_infeasible_draws(out_dir: Path) -> Path | None:
    """Per generation: infeasible slots, total infeasible draws, and how many
    slots stayed unresolved after the rejection cap.  The rejection analogue
    of the Arnold infeasibility-rate plot - shows how much sampling effort the
    rejection loop spent on infeasibles each generation."""
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "resample_per_gen.csv")
    if not rows:
        return None

    gens        = np.array([int(r["generation"]) for r in rows])
    slots       = np.array([_to_float(r.get("n_infeasible_slots")) or 0.0 for r in rows])
    draws       = np.array([_to_float(r.get("n_infeasible_draws")) or 0.0 for r in rows])
    unresolved  = np.array([_to_float(r.get("n_unresolved")) or 0.0 for r in rows])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(gens, draws, color="#b30", label="infeasible draws (all)")
    ax.plot(gens, slots, color="#06a", label="infeasible slots (first draw)")
    ax.plot(gens, unresolved, color="#888", linestyle="--",
            label="unresolved after cap")
    ax.set_title("Resampling: infeasible draws per generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "resample_infeasible_draws.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_all(out_dir: Path) -> list[Path]:
    """Render every resample diagnostic figure once, at end of run.

    Returns the paths actually written (skips any whose CSV source is empty).
    """
    out_dir = Path(out_dir)
    return [
        p for p in (
            plot_constraint_violation_heatmap(out_dir),
            plot_infeasible_draws(out_dir),
        )
        if p is not None
    ]
