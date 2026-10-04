"""
Per-generation diagnostics for the Chocat 2015 constraint-handling technique.

The strategy fills ``strategy.cht_diag_buffer`` with one record per
(parent, phase) CHT invocation.  This module:

* drains the buffer at end-of-generation, tagging each record with the
  generation number,
* appends them to two on-disk CSVs (``cht_per_call.csv`` and
  ``cht_per_gen.csv``),
* every ``plot_interval`` generations, regenerates a six-panel summary
  PNG that visualises whether the CHT is doing what it should (volume
  preserved, anisotropy growing, infeasibility falling, eigenvector
  alignment with the boundary).

CSV schemas are stable so downstream analysis can rely on them.
"""

from __future__ import annotations

import csv
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# CSV schemas

PER_CALL_FIELDS = [
    "generation", "phase", "parent_idx", "lineage_id",
    "n_violators", "shrink_applied", "psd_fallback",
    "log_det_C_before", "log_det_C_after",
    "condition_number_before", "condition_number_after",
    "min_eigenvalue_after", "max_eigenvalue_after",
    "effective_pool_weight",
    "violation_axis_angle_deg",
    # JSON-encoded list-valued columns: stored as strings so downstream
    # tooling (pandas) can json.loads() them on read.
    "eigenvalues_before", "eigenvalues_after",
    "principal_axis_before", "principal_axis_after",
    "mean_violation_direction",
    "per_constraint_active_count",
]

PER_GEN_FIELDS = [
    "generation",
    "n_cht_calls",
    "n_resample_iterations",
    "n_infeasible_post_resample",
    "n_lambda",
    "infeasibility_rate",
    "n_psd_fallback",
    # Aggregates across all (call, parent) records this generation.
    "mean_log_det_drift",       # mean(log_det_after - log_det_before)
    "max_abs_log_det_drift",    # max |drift| - sanity for volume preservation
    "mean_condition_number_after",
    "max_condition_number_after",
    "mean_violation_axis_angle_deg",
    "mean_effective_pool_weight",
    "any_shrink_applied",
]


# Buffer drain -> CSV append

def _jsonify(v):
    """JSON-encode list-valued cells; pass through scalars as-is."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v


def append_per_call_rows(csv_path: Path, gen: int, records: list[dict]) -> None:
    """Append the drained per-call records to cht_per_call.csv.

    Writes a header row on first call (when the file does not yet exist).
    """
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


def _summarise(gen: int, records: list[dict],
                n_resample_iterations: int,
                n_infeasible_post_resample: int,
                n_lambda: int) -> dict:
    """Reduce a generation's worth of per-call records to one summary row.

    Per-parent variability is collapsed by mean/max so the per-gen CSV
    stays small and easy to plot.  The richer per-call signals are still
    available in cht_per_call.csv for any analysis that needs them.
    """
    drifts, conds, angles, pool_ws = [], [], [], []
    n_psd_fallback = 0
    any_shrink = False
    for r in records:
        a, b = r.get("log_det_C_after"), r.get("log_det_C_before")
        if a is not None and b is not None:
            drifts.append(a - b)
        c = r.get("condition_number_after")
        if c is not None:
            conds.append(c)
        ang = r.get("violation_axis_angle_deg")
        if ang is not None:
            angles.append(ang)
        pw = r.get("effective_pool_weight")
        if pw is not None:
            pool_ws.append(pw)
        if r.get("psd_fallback"):
            n_psd_fallback += 1
        if r.get("shrink_applied"):
            any_shrink = True

    def _mean(xs):  return float(np.mean(xs)) if xs else None
    def _max(xs):   return float(np.max(xs))  if xs else None
    def _absmax(xs): return float(np.max(np.abs(xs))) if xs else None

    return {
        "generation":                     gen,
        "n_cht_calls":                    len(records),
        "n_resample_iterations":          n_resample_iterations,
        "n_infeasible_post_resample":     n_infeasible_post_resample,
        "n_lambda":                       n_lambda,
        "infeasibility_rate":             (n_infeasible_post_resample / n_lambda) if n_lambda else None,
        "n_psd_fallback":                 n_psd_fallback,
        "mean_log_det_drift":             _mean(drifts),
        "max_abs_log_det_drift":          _absmax(drifts),
        "mean_condition_number_after":    _mean(conds),
        "max_condition_number_after":     _max(conds),
        "mean_violation_axis_angle_deg":  _mean(angles),
        "mean_effective_pool_weight":     _mean(pool_ws),
        "any_shrink_applied":             any_shrink,
    }


def append_per_gen_row(csv_path: Path, gen: int, records: list[dict],
                        n_resample_iterations: int,
                        n_infeasible_post_resample: int,
                        n_lambda: int) -> dict:
    """Append the per-gen summary row.  Returns the row dict for caller use."""
    csv_path = Path(csv_path)
    row = _summarise(gen, records, n_resample_iterations,
                     n_infeasible_post_resample, n_lambda)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_GEN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


# Augmented Lagrangian diagnostics (CHT_AL sim_type)
# AL telemetry is structurally simpler than the per-call CHT records: one
# row per generation, capturing γ, μ and the proxy (f, g_al) used for the
# update.  We use a separate file (al_per_gen.csv) and a separate drain
# helper rather than wedging AL columns into cht_per_gen.csv, because AL
# state is independent of CHT state and we want both visible side-by-side
# in plots without forcing a join.

AL_PER_GEN_FIELDS = [
    "generation",
    "count",                 # pycma's internal AL update counter
    "is_initialized",        # True once pycma's _initialized array is full
    "f_proxy_scalar",        # cheap-proxy aggregate of f at parent centroid
    "g_al_proxy",            # cheap-proxy g_AL at parent centroid (JSON list)
    "lam",                   # Lagrangian coefficients (JSON list)
    "mu",                    # penalty coefficients      (JSON list)
    "al_pen_proxy",          # AL penalty at the proxy point
    # Diag-2: per-generation population g_al statistics, computed across
    # the *real* (non-sentinel) parents that fed the proxy.  Lets us see
    # whether the mean proxy is masking bimodality / wide distribution.
    "g_al_min",              # min g_al across real parents this gen
    "g_al_max",              # max g_al across real parents this gen
    "g_al_std",              # std deviation of g_al across real parents
    "n_feasible_parents",    # count of real (non-sentinel) parents
    # Diag-3: scaling-drift ratio.  Grows when μ_AL grows faster than
    # |F| shrinks.  Large values mean the penalty dominates the
    # objective in the augmented fitness.
    "pen_to_f_ratio",
    # AL update-proxy summary, so the
    # front-vs-all-parents A/B is self-documenting per generation.
    "n_proxy_set",           # points actually summarised (front size or all)
]


def append_al_per_gen_rows(csv_path: Path, gen: int, records: list[dict]) -> None:
    """Append one CSV row per AL diagnostic record (typically one per gen).

    Writes a header row on first call (when the file does not yet exist).
    """
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AL_PER_GEN_FIELDS)
        if is_new:
            writer.writeheader()
        for rec in records:
            row = {"generation": gen}
            for k in AL_PER_GEN_FIELDS[1:]:
                row[k] = _jsonify(rec.get(k))
            writer.writerow(row)


def drain_and_persist_al(strategy, gen: int, out_dir: Path) -> list[dict]:
    """Drain ``strategy.al_diag_buffer`` and append to al_per_gen.csv.

    Mirrors ``drain_and_persist`` for CHT.  Cheap when the buffer is
    empty (sim_type != CHT_AL), so it can be called unconditionally.

    The CSV header is written eagerly on the first call regardless of
    whether records exist - so post-hoc analysis tools can detect "AL
    mode was enabled but never produced data" by finding a header-only
    file, rather than mistaking a missing file for "wrong sim_type".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(getattr(strategy, "al_diag_buffer", []))
    if hasattr(strategy, "al_diag_buffer"):
        strategy.al_diag_buffer.clear()
    csv_path = out_dir / "al_per_gen.csv"
    # Called unconditionally so the header is written on the first
    # generation even when records is empty (e.g. the AL hasn't
    # bootstrapped yet).  append_al_per_gen_rows is a no-op for an
    # empty records list on subsequent calls.
    append_al_per_gen_rows(csv_path, gen, records)
    return records


def _nondominated_pairs(pairs):
    """Return the non-dominated subset of a list of (f0, f1) tuples.

    Minimisation in both objectives.  O(N²) - only used at end-of-run
    on small lists (final-population N=12, archive N<=100).
    """
    n = len(pairs)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if (pairs[j][0] <= pairs[i][0] and pairs[j][1] <= pairs[i][1]
                    and (pairs[j][0] < pairs[i][0]
                         or pairs[j][1] < pairs[i][1])):
                keep[i] = False
                break
    return [pairs[i] for i in range(n) if keep[i]]


def _diversity_for_set(pairs):
    """Compute Deb's Δ on a set of 2-objective (f0, f1) tuples.

    Returns a dict with: ``n_nd``, ``ext_0``, ``ext_1``, ``d_bar``,
    ``spacing`` (Schott), ``delta`` (Deb).

    The Δ formula uses *self-extremes* for d_f and d_l (= 0) - so
    in-isolation Δ collapses to a uniformity measure
    Σ|d_i - d̄| / ((N-1)-d̄) rather than a coverage measure.  Cross-run
    coverage is captured by the ``ext_*`` fields and by post-hoc
    cross-run analysis.

    Sentinel filter: drop any pair where both components equal 1.0
    (SPARK/PITOT3 failure marker).  Caller must pass *raw scaled*
    objectives, not augmented-Lagrangian fitness.
    """
    import math

    valid = [(float(a), float(b)) for (a, b) in pairs
             if not (a >= 0.99999 and b >= 0.99999)]
    nd = _nondominated_pairs(valid)
    out = {"n_nd": len(nd),
           "ext_0": 0.0, "ext_1": 0.0,
           "d_bar": float("nan"),
           "spacing": float("nan"),
           "delta": float("nan")}
    if len(nd) < 2:
        return out

    nd_sorted = sorted(nd, key=lambda p: p[0])
    f0s = [p[0] for p in nd_sorted]
    f1s = [p[1] for p in nd_sorted]
    out["ext_0"] = max(f0s) - min(f0s)
    out["ext_1"] = max(f1s) - min(f1s)

    dists = [
        math.sqrt((nd_sorted[k + 1][0] - nd_sorted[k][0]) ** 2
                  + (nd_sorted[k + 1][1] - nd_sorted[k][1]) ** 2)
        for k in range(len(nd_sorted) - 1)
    ]
    d_bar = sum(dists) / len(dists)
    out["d_bar"] = d_bar
    out["spacing"] = math.sqrt(
        sum((d - d_bar) ** 2 for d in dists) / len(dists)
    )
    num = sum(abs(d - d_bar) for d in dists)
    den = (len(dists)) * d_bar
    out["delta"] = num / den if den > 0 else float("nan")
    return out


def write_diversity_metrics(out_path, final_parents, archive):
    """Write Δ and crowding-related metrics for the final population
    and the external archive to ``out_path``.

    Both inputs are sequences carrying (f0, f1) pairs in raw fitness
    space:
      * ``final_parents``: DEAP individuals - we read ``fitness.values``
        and ``_feasible`` to filter.
      * ``archive``: archive dicts from ``strategy.external_archive``,
        each with a ``fitness`` tuple.

    Both metric blocks are computed and dumped as a human-readable
    text file.  Designed to be called once at end of run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pop_pairs = []
    for ind in final_parents:
        if not getattr(ind, "_feasible", True):
            continue
        if not ind.fitness.valid:
            continue
        if len(ind.fitness.values) != 2:
            # Diversity metric only defined for the 2-objective CHT_AL
            # set-up; legacy 3-objective runs skip this block.
            return
        # Skip sentinels (PITOT3 or SPARK).  The internal
        # _diversity_for_set function also drops (1, 1) points via its
        # own fitness-based filter, which catches SPARK sentinels; the
        # explicit flag check here additionally catches PITOT3-only
        # sentinels (real fit, fake constraint state).
        if (getattr(ind, "_pitot3_sentinel", False)
                or getattr(ind, "_spark_sentinel", False)):
            continue
        pop_pairs.append(tuple(ind.fitness.values))

    arc_pairs = [tuple(m["fitness"]) for m in archive
                 if len(m["fitness"]) == 2]

    pop_metrics = _diversity_for_set(pop_pairs)
    arc_metrics = _diversity_for_set(arc_pairs)

    def _fmt(v):
        if isinstance(v, float):
            if v != v:  # NaN
                return "n/a"
            return f"{v:.6f}"
        return str(v)

    with out_path.open("w") as f:
        f.write("# Diversity metrics - Deb's Δ on the non-dominated set\n")
        f.write("# Lower Δ = more uniform spread along the front.\n")
        f.write("# Δ uses per-set extremes (d_f = d_l = 0); for cross-run\n")
        f.write("# coverage compare ext_0 and ext_1 directly.\n")
        f.write("\n[Final population]\n")
        for k in ("n_nd", "ext_0", "ext_1", "d_bar", "spacing", "delta"):
            f.write(f"  {k} = {_fmt(pop_metrics[k])}\n")
        f.write("\n[External archive]\n")
        f.write(f"  total_size = {len(archive)}\n")
        for k in ("n_nd", "ext_0", "ext_1", "d_bar", "spacing", "delta"):
            f.write(f"  {k} = {_fmt(arc_metrics[k])}\n")


def drain_and_persist(strategy, gen: int, out_dir: Path,
                       n_resample_iterations: int,
                       n_infeasible_post_resample: int,
                       n_lambda: int) -> list[dict]:
    """One-call helper invoked by main.py after each generation.

    Drains ``strategy.cht_diag_buffer`` (clearing it), writes both CSVs,
    and returns the drained records so the caller can keep them around
    for plotting if it wants to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(strategy.cht_diag_buffer)
    strategy.cht_diag_buffer.clear()

    append_per_call_rows(out_dir / "cht_per_call.csv", gen, records)
    append_per_gen_row(
        out_dir / "cht_per_gen.csv", gen, records,
        n_resample_iterations, n_infeasible_post_resample, n_lambda,
    )
    return records


# Plotting

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


def _displayed_constraints(n_constraints: int) -> tuple[list[int], list[str]]:
    """Pick the subset of the raw g vector to show on the heatmap, with labels.

    The raw constraint vector built by src/problem/feasibility.py for n=6
    design variables has length 18:

      j=0..5   physical-space constraints
      j=6..11  lower box bounds  (x[i] >= 1)
      j=12..17 upper box bounds  (x[i] <= 2)

    Mapping the 12 displayed rows onto the raw g vector requires care
    because the normalised->physical un-transformation in
    src/problem/transforms.py is *coupled* for two variables:

      p4   (i=2):   p4 = (x[2]-1)-1176.01-driver_p + 14.62-driver_p
      res_p (i=4): res_p = (x[4]-1)-(bounds[4][1] - driver_p) + driver_p

    For p4 the box upper x[2] <= 2 corresponds to p4 <= 1190.63-driver_p
    (i.e. compression_ratio <= 70), which is NOT the hard physical
    ceiling p4 <= bounds[2][1].  At high driver_p, p4 can blow past the
    hard ceiling while x[2] is still well within [1, 2].  Sourcing the
    "p4 <= p4_max" row from j=14 would therefore log it as never
    violated even though j=5 (the actual ceiling check) fires
    thousands of times.  We source from j=5 instead.

    For p4 and res_p there is also no *fixed* lower bound - the lower
    limit is coupled to driver_p.  We label those rows accordingly
    instead of the misleading "var >= var_min".

    All other variables (%He, driver_p, D_throat, L_buffer) have linear
    decoupled un-transformations, so their box bounds *are* equivalent
    to their physical bounds.  res_p's *upper* bound is similarly fixed
    (= bounds[4][1] = 8 MPa).

    Caveat - what gets dropped: comp_ratio upper (j=4 / j=14) is the
    only constraint that fires under x[2] > 2 specifically; we don't
    show it as a separate row.  Since j=4/j=14 has 0 firings in
    practice (the algorithm never pushes x[2] past 2 in normalised
    space), this is a non-issue, but worth knowing if a future run
    behaves differently.

    Returns
    -------
    indices : list of int
        Indices into the raw g vector to include, in display order.
    labels : list of str
        Human-readable labels matching `indices` row-for-row.

    Falls back to `(range(n), str(i))` for any length other than 18 so
    this module stays decoupled from the specific problem if the
    constraint layout ever changes.
    """
    if n_constraints != 18:
        return list(range(n_constraints)), [str(i) for i in range(n_constraints)]
    # Lower-bound rows: box-lower j=6..11, except where a more honest
    # physical label exists for coupled variables.
    lower_indices = [6, 7, 8, 9, 10, 11]
    lower_labels  = [
        "%He ≥ %He_min",
        "driver_p ≥ driver_p_min",
        "p4 ≥ 14.62·driver_p",      # coupled - no fixed p4_min
        "D_throat ≥ D_throat_min",
        "res_p ≥ driver_p",         # coupled - no fixed res_p_min
        "L_buf ≥ L_buf_min",
    ]
    # Upper-bound rows: box-upper j=12..17, except j=14 -> j=5 because
    # the box upper for p4 is driver-coupled, not the hard ceiling.
    upper_indices = [12, 13, 5, 15, 16, 17]
    upper_labels  = [
        "%He ≤ %He_max",
        "driver_p ≤ driver_p_max",
        "p4 ≤ p4_max",              # j=5: hard physical ceiling
        "D_throat ≤ D_throat_max",
        "res_p ≤ res_p_max",
        "L_buf ≤ L_buf_max",
    ]
    # Interleave so each variable's lower/upper pair are adjacent rows
    # (lower then upper - with the heatmap's origin="lower" this puts the
    # "var >= min" row directly beneath its "var <= max" row).  Reads far
    # more naturally than the old "all lowers, then all uppers" split.
    paired_indices: list[int] = []
    paired_labels:  list[str] = []
    for lo_i, lo_l, up_i, up_l in zip(
        lower_indices, lower_labels, upper_indices, upper_labels
    ):
        paired_indices += [lo_i, up_i]
        paired_labels  += [lo_l, up_l]
    return paired_indices, paired_labels


def plot_cht_diagnostics(out_dir: Path, current_gen: int) -> Path | None:
    """Generate the six-panel CHT diagnostic figure from the on-disk CSVs.

    Reading from CSV (rather than holding the entire history in memory)
    means this function can be called as a pure side-effect at any
    plotting cadence without changing the main-loop's data structures.

    Returns the path of the saved figure, or None if there is no data
    yet.
    """
    out_dir = Path(out_dir)
    per_call_rows = _read_csv(out_dir / "cht_per_call.csv")
    per_gen_rows  = _read_csv(out_dir / "cht_per_gen.csv")

    if not per_call_rows or not per_gen_rows:
        return None

    # Build per-(gen, parent) tidy arrays
    gens_pc, parents_pc, lineages_pc = [], [], []
    log_det_after, cond_after, angle_pc, pool_w_pc = [], [], [], []
    # Eigenvalue spectrum snapshots for parent 0.  We pull from
    # resample_iter_0 (not post_eval) because post_eval only fires in the
    # rare gens where the resample loop did not feasibilise the full
    # offspring batch - that's an order of magnitude fewer records and
    # they cluster in the early run.  resample_iter_0 fires every gen
    # that has any infeasible at all, so it covers the bulk of the run.
    eigs_after_one_parent_by_gen = {}   # {gen: vp_after} for parent 0
    # Per-constraint counts keyed by gen.  We only keep one snapshot per
    # gen, taken from the resample_iter_0 phase: at that phase every
    # parent's CHT call sees the *same* pool of original infeasibles, so
    # any one parent's count is the per-gen ground truth.  This avoids
    # the mu-fold double-counting that summing across parents/phases
    # produces, and gives a clean denominator (n_lambda) for percentages.
    constraint_counts_iter0_by_gen = {}
    for r in per_call_rows:
        g = int(r["generation"])
        p = int(r["parent_idx"])
        # Lineage column is optional for backward compatibility with
        # CSVs written before lineage tracking was added.  When missing
        # we fall through to colouring by parent_idx, preserving the
        # prior plot behaviour for legacy data.
        lid_raw = r.get("lineage_id")
        lid = int(lid_raw) if lid_raw not in (None, "") else None
        gens_pc.append(g); parents_pc.append(p); lineages_pc.append(lid)
        log_det_after.append(_to_float(r["log_det_C_after"]))
        cond_after.append(_to_float(r["condition_number_after"]))
        angle_pc.append(_to_float(r["violation_axis_angle_deg"]))
        pool_w_pc.append(_to_float(r["effective_pool_weight"]))
        if p == 0 and r.get("phase") == "resample_iter_0":
            ev = _to_list(r["eigenvalues_after"])
            if ev is not None:
                eigs_after_one_parent_by_gen[g] = ev
        if (r.get("phase") == "resample_iter_0"
                and g not in constraint_counts_iter0_by_gen):
            cc = _to_list(r["per_constraint_active_count"])
            if cc is not None:
                constraint_counts_iter0_by_gen[g] = np.array(cc, dtype=float)

    gens_pc = np.array(gens_pc); parents_pc = np.array(parents_pc)
    log_det_after = np.array(log_det_after, dtype=float)
    cond_after    = np.array(cond_after,    dtype=float)
    angle_pc      = np.array(angle_pc,      dtype=float)

    # Per-lineage grouping for trajectory panels (1, 2).  When the CSV has
    # the lineage_id column (post-instrumentation runs), group by lineage
    # so each curve tracks one individual from creation to displacement.
    # Legacy CSVs (no lineage_id) fall back to parent_idx so old runs
    # still render - just with the slot-reassignment artefacts you'd
    # already expect.
    has_lineage = all(lid is not None for lid in lineages_pc) and len(lineages_pc) > 0
    lineage_pc = (np.array(lineages_pc, dtype=int) if has_lineage
                  else parents_pc)

    # Per-gen series
    pg_gens, pg_infeas, pg_resample, pg_psd = [], [], [], []
    lambda_by_gen = {}
    for r in per_gen_rows:
        gen = int(r["generation"])
        pg_gens.append(gen)
        pg_infeas.append(_to_float(r["infeasibility_rate"]))
        pg_resample.append(_to_float(r["n_resample_iterations"]))
        pg_psd.append(_to_float(r["n_psd_fallback"]))
        nl = _to_float(r["n_lambda"])
        if nl is not None and nl > 0:
            lambda_by_gen[gen] = nl
    pg_gens = np.array(pg_gens)
    pg_infeas = np.array(pg_infeas, dtype=float)
    pg_resample = np.array(pg_resample, dtype=float)

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
    fig.suptitle(f"CHT diagnostics through generation {current_gen}", fontsize=14)

    unique_lineages = sorted(set(lineage_pc.tolist()))

    # Pick a colormap: tab10 cycles for slot-id (<= ~12 distinct hues),
    # viridis varies smoothly for lineage IDs (potentially hundreds, and
    # creation-order is meaningful - older lineages on one end, newer on
    # the other).
    if has_lineage:
        cmap = plt.get_cmap("viridis")
        # Normalise lineage IDs onto [0, 1] by their position in creation
        # order so the colour scale is uniform regardless of how many
        # lineages survived.
        n_lineages = max(len(unique_lineages), 1)
        colour_for = lambda k: cmap(k / max(n_lineages - 1, 1))
        group_label = "lineage"
    else:
        cmap = plt.get_cmap("tab10")
        colour_for = lambda k: cmap(k % 10)
        group_label = "parent slot"

    def _plot_per_lineage(ax, y):
        """Draw one polyline per lineage, sorted by generation within each.

        Sorting matters: per-call rows can interleave gens across phases,
        so unsorted plotting would draw zig-zag lines that visually
        misrepresent the trajectory.
        """
        for k, lid in enumerate(unique_lineages):
            mask = lineage_pc == lid
            if not np.any(mask):
                continue
            xs = gens_pc[mask]; ys = y[mask]
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], lw=0.7, alpha=0.7,
                    color=colour_for(k))

    # 1) log det(C) trajectory per lineage (post-CHT)
    # Each curve tracks one individual: it starts when the lineage was
    # created, ends when selection displaced it.  Step 5 of
    # _chtCovarianceUpdate preserves det(C) within a single call, so any
    # gradual decline along a curve is the rank-mu_MO,succ update doing
    # its job between generations (not a CHT-side bug).  See
    # cht_per_gen.csv columns mean_log_det_drift / max_abs_log_det_drift
    # for the per-call invariance diagnostic.
    ax = axes[0, 0]
    _plot_per_lineage(ax, log_det_after)
    ax.set_title(f"log det(C) trajectory (post-CHT, by {group_label})")
    ax.set_xlabel("Generation"); ax.set_ylabel("log det(C) after CHT")
    ax.grid(alpha=0.3)

    # 2) Condition number per lineage
    ax = axes[0, 1]
    _plot_per_lineage(ax, cond_after)
    ax.set_yscale("log")
    ax.set_title(f"Condition number κ(C) (by {group_label})")
    ax.set_xlabel("Generation"); ax.set_ylabel("κ(C)")
    ax.grid(alpha=0.3, which="both")

    # 3) Eigenvalue spectrum heatmap for parent 0 (post_eval phase only)
    ax = axes[0, 2]
    if eigs_after_one_parent_by_gen:
        gs = sorted(eigs_after_one_parent_by_gen.keys())
        spectrum = np.array([eigs_after_one_parent_by_gen[g] for g in gs]).T  # (n_dim, n_gen)
        # log-scale colour: eigenvalues span many orders of magnitude
        im = ax.imshow(np.log10(spectrum + 1e-300), aspect="auto",
                       origin="lower",
                       extent=[gs[0], gs[-1], 0, spectrum.shape[0]],
                       cmap="viridis")
        plt.colorbar(im, ax=ax, label="log10(eigenvalue)")
        ax.set_title("Eigenvalue spectrum (parent 0, resample_iter_0)")
        ax.set_xlabel("Generation"); ax.set_ylabel("Eigenvalue index")
    else:
        ax.text(0.5, 0.5, "no parent-0 post_eval data yet",
                ha="center", va="center", transform=ax.transAxes)

    # 4) Infeasibility rate + resample iterations on twin axes
    ax = axes[1, 0]
    ax.plot(pg_gens, pg_infeas, color="C3", label="infeasibility rate")
    ax.set_xlabel("Generation"); ax.set_ylabel("infeasibility rate", color="C3")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(pg_gens, pg_resample, color="C0", lw=0.8, alpha=0.7,
             label="resample iters")
    ax2.set_ylabel("# resample iterations", color="C0")
    ax.set_title("Infeasibility & resample iterations")

    # 5) Per-constraint activity heatmap (% of LAMBDA initial offspring)
    # Counts come from the resample_iter_0 phase, normalised by n_lambda
    # so each cell reads as "% of this generation's initial offspring
    # that violated constraint j".  _displayed_constraints() collapses
    # the raw 18-element g vector to 12 physical-variable bounds.
    ax = axes[1, 1]
    if constraint_counts_iter0_by_gen:
        gs = sorted(constraint_counts_iter0_by_gen.keys())
        n_raw = len(constraint_counts_iter0_by_gen[gs[0]])
        indices, labels = _displayed_constraints(n_raw)
        n_displayed = len(indices)
        pct = np.zeros((n_displayed, len(gs)), dtype=float)
        for col, gn in enumerate(gs):
            denom = lambda_by_gen.get(gn)
            if denom:
                full_counts = constraint_counts_iter0_by_gen[gn]
                pct[:, col] = 100.0 * full_counts[indices] / denom
        im = ax.imshow(pct, aspect="auto", origin="lower",
                       extent=[gs[0], gs[-1], -0.5, n_displayed - 0.5],
                       cmap="magma", vmin=0.0)
        plt.colorbar(im, ax=ax,
                     label="% of LAMBDA offspring violating constraint")
        ax.set_yticks(np.arange(n_displayed))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Per-constraint activity (resample_iter_0)")
        ax.set_xlabel("Generation"); ax.set_ylabel("Constraint")
    else:
        ax.text(0.5, 0.5, "no per-constraint data yet",
                ha="center", va="center", transform=ax.transAxes)

    # 6) Principal axis vs mean violation direction angle
    ax = axes[1, 2]
    valid = np.isfinite(angle_pc)
    if np.any(valid):
        ax.scatter(gens_pc[valid], angle_pc[valid], s=4, alpha=0.4,
                   c=parents_pc[valid], cmap="tab10")
        ax.axhline(90, color="k", ls="--", lw=0.8, alpha=0.6,
                   label="orthogonal (target)")
        ax.set_ylim(0, 92)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Angle: principal axis vs mean violation direction")
    ax.set_xlabel("Generation"); ax.set_ylabel("angle (deg)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"cht_diagnostics_gen_{current_gen:04d}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
