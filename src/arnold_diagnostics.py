"""
Per-generation diagnostics for the Arnold & Hansen 2012 constraint-handling
technique (ArnoldCHT / ArnoldCHT_AL sim_types).

Structurally mirrors ``cht_diagnostics.py``: the strategy fills
``strategy.arnold_diag_buffer`` with one record per infeasible offspring
consumed in ``apply_arnold_infeasibility`` (Eq. 6 v_j filter + Eq. 7
subtractive Cholesky update).  This module

* drains that buffer at end-of-generation, tagging each record with the
  generation number,
* appends to two stable on-disk CSVs (``arnold_per_call.csv`` and
  ``arnold_per_gen.csv``) - written EVERY generation so the run is
  crash-safe and analysable mid-flight,
* exposes four standalone plot functions, each producing ONE figure, meant
  to be called ONCE after the run completes (the CSVs hold the full
  history, so plotting is a stateless read-from-disk side-effect).

The four figures, blending the Chocat and Arnold diagnostic styles:

  1. ``plot_constraint_violation_heatmap`` - gen x constraint heatmap of how
     often each constraint was violated (Chocat-style, % of λ).
  2. ``plot_infeasibility_rate``          - n_infeasible / λ per generation
     (Arnold-style).
  3. ``plot_mean_vj``                      - mean ‖v_j‖ per active constraint
     over generations (Arnold-style).
  4. ``plot_condition_by_lineage``        - κ(C) trajectory, one curve per
     lineage (Chocat-style).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Reuse the Chocat constraint-label mapping so the heatmap rows read as
# physical-variable bounds (e.g. "p4 <= p4_max") rather than raw g indices.
# It also collapses the 18-element raw g vector to the 12 meaningful rows.
from cht_diagnostics import _displayed_constraints


# CSV schemas

PER_CALL_FIELDS = [
    "generation", "parent_idx", "lineage_id",
    "m_active",                 # number of constraints active in this call
    "A_delta_fro",              # ‖A_new − A‖_F of the Eq. 7 step
    "condition_number_after",   # κ(C) of the parent's covariance post-update
    "shrink_applied", "psd_fallback",
    # JSON-encoded list columns (downstream tooling json.loads() them):
    "active_js",                # constraint indices violated (finite g_j > 0)
    "v_norms",                  # ‖v_j‖ for each active j, aligned with active_js
]

PER_GEN_FIELDS = [
    "generation",
    "n_arnold_calls",           # records this gen (= infeasible offspring seen)
    "n_infeasible",             # infeasible offspring this gen (slots w/o a candidate)
    "n_lambda",
    "infeasibility_rate",       # n_infeasible / λ
    "n_psd_fallback",
    "n_no_active_js",           # infeasibles with no finite-positive g_j
    "mean_m_active",
    "mean_A_delta_fro",
    "max_A_delta_fro",
    "any_shrink_applied",
    "n_constraints",            # length of the raw g vector (for the heatmap)
    # JSON list (length n_constraints): how many offspring violated each
    # constraint j this generation.  Sums active_js across the gen's records.
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


def append_per_call_rows(csv_path: Path, gen: int, records: list[dict]) -> None:
    """Append the drained per-call records, writing a header on first call."""
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CALL_FIELDS)
        if is_new:
            writer.writeheader()
        for rec in records:
            row = {"generation": gen}
            for k in PER_CALL_FIELDS[1:]:
                row[k] = _jsonify(rec.get(k))
            writer.writerow(row)


def _summarise(gen: int, records: list[dict], n_infeasible: int,
               n_lambda: int, n_constraints: int) -> dict:
    """Reduce a generation's per-call records to one summary row.

    ``per_constraint_violation_count`` aggregates the heatmap source: for
    every record, each constraint index in ``active_js`` increments its
    counter.  Length is fixed at ``n_constraints`` so constraints that were
    never violated this gen still occupy a (zero) row in the heatmap.
    """
    fros, m_actives = [], []
    n_psd_fallback = 0
    n_no_active_js = 0
    any_shrink = False
    counts = np.zeros(max(n_constraints, 0), dtype=int)
    for r in records:
        f = r.get("A_delta_fro")
        if f is not None:
            fros.append(float(f))
        ma = r.get("m_active")
        if ma is not None:
            m_actives.append(int(ma))
            if int(ma) == 0:
                n_no_active_js += 1
        if r.get("psd_fallback"):
            n_psd_fallback += 1
        if r.get("shrink_applied"):
            any_shrink = True
        for j in (r.get("active_js") or []):
            if 0 <= int(j) < len(counts):
                counts[int(j)] += 1

    def _mean(xs): return float(np.mean(xs)) if xs else None
    def _max(xs):  return float(np.max(xs))  if xs else None

    return {
        "generation":                     gen,
        "n_arnold_calls":                 len(records),
        "n_infeasible":                   n_infeasible,
        "n_lambda":                       n_lambda,
        "infeasibility_rate":             (n_infeasible / n_lambda) if n_lambda else None,
        "n_psd_fallback":                 n_psd_fallback,
        "n_no_active_js":                 n_no_active_js,
        "mean_m_active":                  _mean(m_actives),
        "mean_A_delta_fro":               _mean(fros),
        "max_A_delta_fro":                _max(fros),
        "any_shrink_applied":             any_shrink,
        "n_constraints":                  int(n_constraints),
        "per_constraint_violation_count": counts.tolist(),
    }


def append_per_gen_row(csv_path: Path, gen: int, records: list[dict],
                       n_infeasible: int, n_lambda: int,
                       n_constraints: int) -> dict:
    """Append the per-gen summary row.  Returns the row dict."""
    csv_path = Path(csv_path)
    row = _summarise(gen, records, n_infeasible, n_lambda, n_constraints)
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
                      n_infeasible: int, n_lambda: int) -> list[dict]:
    """One-call helper invoked by main.py after each generation.

    Drains ``strategy.arnold_diag_buffer`` (clearing it), writes both CSVs,
    and returns the drained records.  Cheap when the buffer is empty
    (non-Arnold sim_types), so it can be called unconditionally if desired.

    The CSV headers are written eagerly on the first call regardless of
    whether records exist, so post-hoc tooling can tell "Arnold mode ran
    but produced no infeasibles" from "wrong sim_type / no file".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(getattr(strategy, "arnold_diag_buffer", []))
    if hasattr(strategy, "arnold_diag_buffer"):
        strategy.arnold_diag_buffer.clear()
    # n_constraints sizes the per-constraint heatmap vector.  Fall back to
    # the max index actually seen if the strategy didn't expose it.
    n_constraints = getattr(strategy, "n_constraints", None)
    if not n_constraints:
        seen = [int(j) for r in records for j in (r.get("active_js") or [])]
        n_constraints = (max(seen) + 1) if seen else 0

    append_per_call_rows(out_dir / "arnold_per_call.csv", gen, records)
    append_per_gen_row(out_dir / "arnold_per_gen.csv", gen, records,
                       n_infeasible, n_lambda, n_constraints)
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


# Plot 1 - per-constraint violation heatmap (Chocat-style)

def plot_constraint_violation_heatmap(out_dir: Path) -> Path | None:
    """Heatmap of how often each constraint is violated, gen x constraint.

    Source: ``per_constraint_violation_count`` from arnold_per_gen.csv,
    normalised by λ so each cell reads "% of this generation's offspring
    that violated constraint j".  Rows are collapsed/relabelled to the 12
    physical-variable bounds via the shared Chocat ``_displayed_constraints``
    mapping.
    """
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "arnold_per_gen.csv")
    if not rows:
        return None

    gens, counts_by_gen, lambda_by_gen = [], [], []
    n_raw = None
    for r in rows:
        cc = _to_list(r.get("per_constraint_violation_count"))
        if cc is None:
            continue
        g = int(r["generation"])
        gens.append(g)
        counts_by_gen.append(np.asarray(cc, dtype=float))
        lambda_by_gen.append(_to_float(r.get("n_lambda")) or 0.0)
        if n_raw is None:
            n_raw = len(cc)
    if not gens or not n_raw:
        return None

    indices, labels = _displayed_constraints(n_raw)
    n_disp = len(indices)
    pct = np.zeros((n_disp, len(gens)), dtype=float)
    for col, (cc, lam) in enumerate(zip(counts_by_gen, lambda_by_gen)):
        if lam > 0:
            pct[:, col] = 100.0 * cc[indices] / lam

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    im = ax.imshow(pct, aspect="auto", origin="lower",
                   extent=[gens[0], gens[-1], -0.5, n_disp - 0.5],
                   cmap="magma", vmin=0.0)
    plt.colorbar(im, ax=ax, label="% of λ offspring violating constraint")
    ax.set_yticks(np.arange(n_disp))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Arnold: per-constraint violation frequency")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Constraint")
    fig.tight_layout()
    fig_path = out_dir / "arnold_constraint_heatmap.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# Plot 2 - infeasibility rate per generation (Arnold-style)

def plot_infeasibility_rate(out_dir: Path) -> Path | None:
    """n_infeasible / λ per generation - how much of each batch the Arnold
    path "spends" on infeasibles (which contribute no selection candidate)."""
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "arnold_per_gen.csv")
    if not rows:
        return None

    gens = np.array([int(r["generation"]) for r in rows])
    rate = np.array([_to_float(r.get("infeasibility_rate")) or 0.0
                     for r in rows])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(gens, rate, color="#0a4")
    ax.set_title("Arnold: infeasibility rate per generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("n_infeasible / λ")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "arnold_infeasibility_rate.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# Plot 3 - mean ‖v_j‖ per active constraint (Arnold-style)

def plot_mean_vj(out_dir: Path) -> Path | None:
    """One curve per constraint: mean ‖v_j‖ over the calls where j was active,
    per generation.  Answers "is the v_j low-pass filter actually learning
    a stable boundary direction for each constraint?"."""
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "arnold_per_call.csv")
    if not rows:
        return None

    # Pool ‖v_j‖ by (generation, j): average across parents/calls.
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    for r in rows:
        g = int(r["generation"])
        active = _to_list(r.get("active_js")) or []
        norms = _to_list(r.get("v_norms")) or []
        for j, vn in zip(active, norms):
            key = (g, int(j))
            sums[key] = sums.get(key, 0.0) + float(vn)
            counts[key] = counts.get(key, 0) + 1

    js_seen = sorted({k[1] for k in sums})
    if not js_seen:
        return None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    cmap = plt.get_cmap("tab20")
    for k, j in enumerate(js_seen):
        xs = sorted(g for (g, jj) in sums if jj == j)
        ys = [sums[(g, j)] / counts[(g, j)] for g in xs]
        ax.plot(xs, ys, label=f"j={j}", color=cmap(k % 20), linewidth=1.2)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Arnold: mean ‖v_j‖ per active constraint")
    ax.set_xlabel("Generation")
    ax.set_ylabel("‖v_j‖")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "arnold_mean_vj.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# Plot 4 - condition number κ(C) by lineage (Chocat-style)

def plot_condition_by_lineage(out_dir: Path) -> Path | None:
    """κ(C) over time, one polyline per lineage.

    Each curve tracks one individual's covariance from creation to
    displacement.  A rising κ along a curve means the Eq. 7 shrinkage is
    making that lineage's search anisotropic (squeezing the boundary
    direction) - the intended behaviour; runaway κ flags over-shrinkage.
    """
    out_dir = Path(out_dir)
    rows = _read_csv(out_dir / "arnold_per_call.csv")
    if not rows:
        return None

    by_lineage: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        lid = r.get("lineage_id")
        cond = _to_float(r.get("condition_number_after"))
        if lid in (None, "") or cond is None:
            continue
        by_lineage.setdefault(int(lid), []).append((int(r["generation"]), cond))
    if not by_lineage:
        return None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    lineages = sorted(by_lineage)
    cmap = plt.get_cmap("viridis")
    n = max(len(lineages), 1)
    for k, lid in enumerate(lineages):
        pts = sorted(by_lineage[lid])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=0.8, alpha=0.7, color=cmap(k / max(n - 1, 1)))
    ax.set_yscale("log")
    ax.set_title("Arnold: condition number κ(C) by lineage")
    ax.set_xlabel("Generation")
    ax.set_ylabel("κ(C)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_path = out_dir / "arnold_condition_by_lineage.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# One-shot driver - call ONCE after the run completes

def plot_all(out_dir: Path) -> list[Path]:
    """Generate all four Arnold diagnostic figures from the on-disk CSVs.

    Each plotter writes its own standalone figure and returns its path (or
    None if it had no data).  Returns the list of figures actually written.
    """
    paths = [
        plot_constraint_violation_heatmap(out_dir),
        plot_infeasibility_rate(out_dir),
        plot_mean_vj(out_dir),
        plot_condition_by_lineage(out_dir),
    ]
    return [p for p in paths if p is not None]
