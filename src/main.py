"""
Main entry point for the X2 free-piston driver multi-objective optimisation.

Reads config/experiments.yaml and runs one experiment per list entry, each in
a fresh subprocess so the memory a run accumulates is reclaimed between them.

Command line:
    python3 src/main.py [--seed-npz PATH]

    --seed-npz          a .npz from init_population_l1d.py. Its ``x_norm``
                        array replaces the random initial population. Omit it
                        to sample the initial population at random.
    --experiment-index  internal. The dispatcher re-execs itself with this
                        to run a single entry; it is not meant to be passed
                        by hand.

Each YAML entry resolves to a 10-tuple, assembled at the bottom of this file:

    (sim_type, pop_size, step_size, p4_treatment, al_tol,
     cht_gamma, features, arnold_beta, arnold_cc, n_gen)

sim_type selects the constraint handling:

    'ArnoldCHT_AL'     Arnold and Hansen 2012 covariance adaptation plus an
                       Augmented Lagrangian on the shock-speed shortfall.
                       An infeasible offspring is consumed before evaluation
                       to shrink its parent's covariance along the violated
                       direction, and the slot is then dropped. This is the
                       configuration the distributed runs used.
    'CHT_AL'           as above but with the Chocat 2015 covariance shrinkage
                       in place of Arnold's, and infeasibles resampled from a
                       tightened distribution rather than dropped.
    'ArnoldCHT'        the Arnold covariance adaptation alone, with the
                       shock speed as a third objective rather than a
                       constraint.
    'CovarianceCHT'    the Chocat covariance shrinkage alone, likewise
                       3-objective.
    'Resampling_AL'    rejection resampling from the unchanged distribution,
                       the baseline the active adaptations are measured
                       against.
    'ParentValue', 'ElitistCrossover', 'RandomCrossover', 'Penalty'
                       earlier repair-based and penalty-based treatments,
                       kept as comparison baselines.

p4_treatment enforces the upper p4 bound after mutation. It must be None for
the CHT family, which replaces bound repair entirely.

Output layout
-------------
Each experiment creates a fresh numbered folder,

    <repo>/Results/<category>/<prefix>_NNNN/

numbered from the largest existing NNNN plus one, with one subfolder per
output type. The AL strategies are 2-objective and write their own
al_diagnostics/ directory; see OUTPUT_FOLDERS_AL below.

Module-level setup
------------------
The DEAP creator types are registered at import. creator.create mutates a
global registry, so it has to run exactly once per process, and importing
this module is what guarantees that.
"""

import gc
import os
import pathlib
import resource
import subprocess
import sys
import time
import multiprocessing
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from deap import base, creator, tools

from algorithm.toolbox   import Toolbox
from algorithm.hypervolume import HyperVolume
from algorithm.cmaes     import (
    StrategyMultiObjective,
    is_al_active, is_cht_active, cht_method, is_resample_active,
)
from problem.config      import (
    APPROX_IDEAL, APPROX_NADIR,
    APPROX_IDEAL_2D, APPROX_NADIR_2D,
    BOUNDS,
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_lower, p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_lower, reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.transforms  import variable_transformation, variable_untransformation, unnormalise_fitness
from problem.evaluate    import evaluate, set_logbook
# Re-exported constant so the sentinel detector below can compare
# delta_vs1 against the same value the evaluator uses on failure.
from problem.evaluate    import _PITOT3_FAILURE_SENTINEL
from problem.feasibility import evaluate_constraints, is_feasible
from problem            import l1d_job
from plotting            import (
    plot_objective_space,
    plot_objective_space_3d,
    plot_objective_space_heatmap,
    plot_holdtime_impactspeed_2d,
)
from results_io          import (
    setup_run_directory,
    setup_subfolders,
    write_population_csv,
)
from cht_diagnostics     import drain_and_persist as _cht_drain_and_persist
from cht_diagnostics     import drain_and_persist_al as _al_drain_and_persist
from cht_diagnostics     import plot_cht_diagnostics as _cht_plot
from cht_diagnostics     import write_diversity_metrics as _write_diversity_metrics
# Single-source the diversity metric (Deb's Δ) used both for the
# end-of-run text dump and the new per-generation diversity plot.
from cht_diagnostics     import _diversity_for_set
from al_plots            import (
    plot_archive_pareto,
    plot_epsilon_and_gal,
    plot_diversity,
)
from arnold_diagnostics  import drain_and_persist as _arnold_drain_and_persist
from arnold_diagnostics  import plot_all as _arnold_plot_all
from resample_diagnostics import drain_and_persist as _resample_drain_and_persist
from resample_diagnostics import plot_all as _resample_plot_all


def _quiet_l1d_worker():
    """Pool initializer: silence L1d streaming in every worker process.

    Mirrors init_population_l1d._init_worker.  With λ workers evaluating in
    parallel, letting each l1d4 subprocess inherit the parent's stdout would
    interleave their per-step progress into unreadable noise, so we route it
    to spooled tempfiles instead (see STREAM_L1D_OUTPUT in problem/l1d_job.py).
    Must be a module-level function so multiprocessing can pickle it.
    """
    l1d_job.STREAM_L1D_OUTPUT = False


# DEAP type registration  (runs once on import)

# FitnessMulti: all three objectives are minimised (weights = -1).
# The evaluator returns normalised values, so minimising maps to:
#   delta_vs1   -> 0 is best
#   hold_time   -> 0 is best (we negate: longer hold time = smaller normalised value)
#   impact_speed -> 0 is best (same negation logic)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual",   list, fitness=creator.FitnessMulti,
               ind_number=int, sim_type=str, bounds=list)

# 2-objective variants used by the CHT_AL sim_type, where delta_vs1 has
# been moved from a Pareto objective to an Augmented-Lagrangian constraint.
# Registered unconditionally so the namespace is always populated; main()
# picks which class to instantiate at run-time based on sim_type.
creator.create("FitnessMulti2D", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual2D",   list, fitness=creator.FitnessMulti2D,
               ind_number=int, sim_type=str, bounds=list)

# Module-level singletons

toolbox = Toolbox()
toolbox.register("evaluate", evaluate)

# Reference point for HV computation: the all-zeros point in normalised
# space.  Instantiated as 3-D for legacy sim_types; main() rebinds the
# name to a 2-D HyperVolume when is_al_active(sim_type).
pop_hypervolumes = HyperVolume(np.array((0, 0, 0)))

normalised = True

# Output structure

RESULTS_CATEGORY = "parent_value_with_recomb"
RUN_PREFIX       = "pv_w_rec"
SAVE_INTERVAL    = 10

# Legacy 3-objective output structure (delta_vs1 + hold_time + impact_speed).
OUTPUT_FOLDERS = [
    "pareto_3d",
    "pareto_heatmap",
    "pareto_dvs1_holdtime",
    "pareto_dvs1_impactspeed",
    "pareto_holdtime_impactspeed",
    "population",
    # Surviving μ parent set per save trigger - the post-selection Pareto
    # front, distinct from "population" which records the pre-selection
    # offspring batch (and so includes individuals that selection then
    # discards).  Direct artefact for "what is the current Pareto front".
    "parents",
    "convergence",
    "summary",
    # CHT diagnostics (CSVs + per-SAVE_INTERVAL summary plots).  Created
    # for every run; only populated when sim_type == 'CovarianceCHT'.
    "cht_diagnostics",
    # Arnold CHT diagnostics (per-gen CSVs + four end-of-run figures).
    # Created for every run; only populated when cht_method == 'arnold'.
    "arnold_diagnostics",
    # Per-generation strategy state (σ and psucc per parent slot).
    # Populated for ALL sim_types so the σ death-spiral hypothesis
    # can be verified independently of constraint-handling choice.
    "strategy_diagnostics",
]

# CHT_AL output structure: 2-objective (no 3-D Pareto plot, no
# delta_vs1-vs-* plots - delta_vs1 is now a constraint), plus a new
# al_diagnostics directory mirroring cht_diagnostics for AL telemetry.
RESULTS_CATEGORY_AL = "al_cht_recomb"
RUN_PREFIX_AL       = "al_cht"
OUTPUT_FOLDERS_AL = [
    "pareto_holdtime_impactspeed",
    "population",
    "parents",
    "convergence",
    "summary",
    "cht_diagnostics",
    "arnold_diagnostics",
    "resample_diagnostics",
    "strategy_diagnostics",
    "al_diagnostics",
    # End-of-run post-processing figures (.eps): archive Pareto scatter,
    # ε-schedule + per-individual g_AL, and per-generation diversity.
    # Populated once at end of run for the AL family; see al_plots.py.
    "postprocessing",
]


def _run_constants(sim_type):
    """Resolve sim_type-specific run constants in one place.

    Returns
    -------
    (results_category, run_prefix, output_folders, Individual_cls,
     ideal_point, nadir_point) :
        Individual_cls is the DEAP class to instantiate for each
        offspring (Individual for 3-objective, Individual2D for AL).
        ideal_point / nadir_point are used by the unnormalisation step
        in the summary writer; they match the dimensionality of the
        selected Individual_cls.
    """
    if is_al_active(sim_type):
        # Both CHT_AL and ArnoldCHT_AL use the 2-objective (hold_time,
        # impact) tree with delta_vs1 handled by the Augmented Lagrangian.
        return (
            RESULTS_CATEGORY_AL, RUN_PREFIX_AL, OUTPUT_FOLDERS_AL,
            creator.Individual2D,
            APPROX_IDEAL_2D, APPROX_NADIR_2D,
        )
    return (
        RESULTS_CATEGORY, RUN_PREFIX, OUTPUT_FOLDERS,
        creator.Individual,
        APPROX_IDEAL, APPROX_NADIR,
    )


def _detect_sentinels(ind, fit, g_al):
    """Set `_pitot3_sentinel` and `_spark_sentinel` flags on `ind`.

    Called once per individual right after heavy evaluation.  The two
    sentinel modes are independent:

    - **PITOT3 sentinel**: the shock-speed solver hit its failure
      marker, encoded as ``delta_vs1 = 3585 m/s``.  Recovered from
      ``g_al + ind.al_tol``.  This case can leave the per-individual
      ``fitness`` *real* (SPARK succeeded), so detecting it on fitness
      alone misses these individuals.

    - **SPARK sentinel**: the hold-time / impact-speed solver hit its
      failure markers (raw 0 and 350), normalised to (1, 1).  Detected
      from the fitness tuple.

    Either flag should cause exclusion from the AL proxy mean, the AL
    bootstrap, the donor's psucc / σ update, the external archive, and
    the non-sentinel HV trace.

    Both flags default to False when the corresponding input is None
    (e.g. legacy 3-objective mode where fit_2d / g_al aren't computed
    in the CHT_AL shape).
    """
    if fit is None:
        ind._spark_sentinel = False
    else:
        ind._spark_sentinel = all(v >= 0.99999 for v in fit)

    if g_al is None:
        ind._pitot3_sentinel = False
    else:
        # Threshold 1 m/s below the exact 3585 sentinel; a genuinely
        # measured delta_vs1 only reaches 3585 when vs1 -> 0, which already
        # returns the sentinel, so this never false-positives on a real run.
        al_tol = getattr(ind, "al_tol", 100.0)
        delta_vs = float(np.asarray(g_al)[0]) + al_tol
        ind._pitot3_sentinel = delta_vs >= (_PITOT3_FAILURE_SENTINEL - 1.0)


def _is_sentinel(ind):
    """Return True if `ind` carries either a PITOT3 or SPARK sentinel
    flag.  Convenience wrapper for filter sites."""
    return (getattr(ind, "_pitot3_sentinel", False)
            or getattr(ind, "_spark_sentinel", False))


def _git_stamp():
    """Return the repo's short commit SHA (``-dirty`` suffix if the working
    tree has uncommitted changes), or ``"unknown"`` if git is unavailable.

    Captured once at run time and written into summary/output.txt so every
    result folder is self-identifying: the exact code + working-tree state
    that produced it.  Best-effort - a missing git or non-repo checkout must
    never abort a multi-hour optimisation, hence the broad except.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _features_oneline(features):
    """Render a features dict as one compact, greppable, sorted line.

    Empty / None -> ``"(none)"``.  Matches the ``key = value`` idiom of the
    surrounding output.txt so a whole run's config sits on two lines
    (git commit + features) rather than being scattered across the log.
    """
    if not features:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(features.items()))


def _cheap_al_proxy(strategy):
    """Return (F̄, ḡ_AL, stats) cheap proxy from the current parent set.

    The AL coefficient adaptation in pycma expects values "at the
    distribution mean".  In MO-CMA-ES there is no single mean; per the
    user-confirmed design we average over parents that survived
    selection.  Zero extra heavy evaluations.

    Parents whose fitness.values are at
    the (1, 1) reference (heavy-evaluator failures) are excluded from
    the proxy mean.  Their g_al is a sentinel-implied value
    (PITOT3 -> +3485 m/s, i.e. 3585 - al_tol) that does not reflect the true geometric
    state of the population; including them skews the mean and
    pollutes pycma's CDF-based μ-update.

    Returns (None, None, {}) when no real parent remains for proxying
    (all parents are sentinel-failures - should be rare).

    Also returns a stats dict capturing the per-generation population g_al
    distribution (min, max, std, count) over the full clean parent set.
    """
    # Clean candidate set: valid fitness, has g_al, non-sentinel. Either
    # kind of sentinel would contaminate the aggregate.
    candidates = [
        p for p in strategy.parents
        if p.fitness.valid
        and getattr(p, "_g_al", None) is not None
        and not _is_sentinel(p)
    ]
    if not candidates:
        return None, None, {}

    # Distribution stats over the FULL clean set, so the spread is visible
    # even when the proxy below is front-restricted.
    g_all = np.stack([np.asarray(p._g_al, dtype=float) for p in candidates],
                     axis=0)

    proxy_set = candidates

    g_arr = np.stack([np.asarray(p._g_al, dtype=float) for p in proxy_set],
                     axis=0)                     # (n_proxy_set, m)
    F_arr = np.asarray([sum(p.fitness.values) for p in proxy_set],
                       dtype=float)

    g_proxy = np.mean(g_arr, axis=0)

    stats = {
        "g_al_min":           float(np.min(g_all)),
        "g_al_max":           float(np.max(g_all)),
        "g_al_std":           float(np.std(g_all)),
        "n_feasible_parents": int(len(candidates)),
        "F_proxy_abs_max":    float(np.max(np.abs(F_arr))),
        "n_proxy_set":        int(len(proxy_set)),
    }
    return float(np.mean(F_arr)), g_proxy, stats


def _store_raw_delta_vs1(ind, g_al):
    """Cache the schedule-independent delta_vs1 measurement on ``ind``.

    ``g_al`` is ``[delta_vs1 - al_tol]`` computed at evaluation with the
    individual's birth-generation ``al_tol``, so the underlying physics
    measurement is ``delta_vs1 = g_al[0] + ind.al_tol``.  Caching it lets
    ``strategy.refresh_al_constraints`` recompute ``g_al`` against a moving
    ``al_tol`` schedule each generation without re-running L1d (Task 2).

    Set to ``None`` when there is no measurement (box/phys-infeasible
    individual - ``g_al is None``) so the refresh skips it cleanly.
    """
    if g_al is None:
        ind._raw_delta_vs1 = None
        return
    ind._raw_delta_vs1 = float(np.asarray(g_al)[0]) + getattr(ind, "al_tol", 100.0)


# Snapshot helpers

def _hv_contributions(population, ref=None):
    """Per-individual HV contribution = HV(pop) − HV(pop ∖ {i}).

    The HyperVolume class expects "to maximise" inputs, so we negate the
    fitness values (which are all to-minimise) before passing them in.

    The reference point dimensionality is auto-detected from the first
    feasible individual's fitness tuple - 3-D for legacy sim_types, 2-D
    for CHT_AL.  Caller may override via ``ref``.

    Infeasible individuals (fitness.valid is False) have no fitness and
    therefore no HV contribution; they appear in the returned list as
    None and are omitted from the HV computation entirely.
    """
    if len(population) == 0:
        return []
    feasible_idx = [i for i, ind in enumerate(population) if ind.fitness.valid]
    if not feasible_idx:
        return [None] * len(population)
    fits_neg = np.array([list(population[i].fitness.values) for i in feasible_idx]) * -1
    if ref is None:
        ref = np.zeros(fits_neg.shape[1])
    hv = HyperVolume(ref)
    full = hv.compute(fits_neg)
    contribs_feasible = []
    for k in range(len(feasible_idx)):
        partial = hv.compute(np.delete(fits_neg, k, axis=0))
        contribs_feasible.append(full - partial)
    out = [None] * len(population)
    for k, i in enumerate(feasible_idx):
        out[i] = contribs_feasible[k]
    return out


def _build_pop_row(ind, gen, slot_idx, sigma_used, parent_idx, hv_contribution, bounds):
    """One CSV row for one individual.

    Infeasible individuals (no valid fitness) emit None for every objective
    column so downstream analysis can distinguish "not evaluated" from a
    real zero.  The 'feasible' and 'max_g' columns surface the constraint
    state for post-hoc CHT diagnostics.

    The fitness dimensionality is auto-detected from the individual's
    fitness tuple.  For CHT_AL (2-D fitness) the legacy delta_vs1 columns
    are recovered from ``ind._g_al`` (the AL constraint vector, =
    delta_vs1 - al_tol) so the CSV schema stays stable for downstream
    analysis tools - the columns are identical, just sourced differently.
    """
    raw_vars    = variable_untransformation(ind, bounds)
    scaled_vars = list(ind)

    is_2d_fitness = (
        ind.fitness.valid and len(ind.fitness.values) == 2
    )

    if ind.fitness.valid:
        scaled_objs_raw = list(ind.fitness.values)
        if is_2d_fitness:
            # CHT_AL: fitness is (hold_time, impact_speed) - delta_vs1 is
            # not in fitness.values but is recoverable from g_al.
            raw_2d  = list(unnormalise_fitness(ind.fitness.values,
                                               APPROX_IDEAL_2D, APPROX_NADIR_2D))
            g_al    = getattr(ind, "_g_al", None)
            al_tol  = getattr(ind, "al_tol", 100.0)
            raw_dvs = (float(g_al[0]) + al_tol) if g_al is not None else None
            raw_objs    = [raw_dvs, raw_2d[0], raw_2d[1]]
            # delta_vs1 has no normalisation in 2-D mode; emit None.
            scaled_objs = [None, scaled_objs_raw[0], scaled_objs_raw[1]]
        else:
            scaled_objs = scaled_objs_raw
            raw_objs    = list(unnormalise_fitness(ind.fitness.values,
                                                   APPROX_IDEAL, APPROX_NADIR))
    else:
        scaled_objs = [None, None, None]
        raw_objs    = [None, None, None]

    g = getattr(ind, "_g", None)
    feasible = getattr(ind, "_feasible", None)
    max_g = float(np.max(g)) if g is not None else None

    return {
        "generation":            gen,
        "ind_number":            slot_idx,
        "parent_idx":            parent_idx,
        "chosen":                False,
        "offspring_ind_number":  None,
        "sigma":                 sigma_used,
        "hv_contribution":       hv_contribution,
        "feasible":              feasible,
        "max_g":                 max_g,
        "raw_pct_he":            raw_vars[0],
        "raw_driver_p":          raw_vars[1],
        "raw_p4":                raw_vars[2],
        "raw_d_throat":          raw_vars[3],
        "raw_reservoir_p":       raw_vars[4],
        "raw_buffer_length":     raw_vars[5],
        "scaled_pct_he":         scaled_vars[0],
        "scaled_driver_p":       scaled_vars[1],
        "scaled_p4":             scaled_vars[2],
        "scaled_d_throat":       scaled_vars[3],
        "scaled_reservoir_p":    scaled_vars[4],
        "scaled_buffer_length":  scaled_vars[5],
        "raw_delta_vs1":         raw_objs[0],
        "raw_hold_time":         raw_objs[1],
        "raw_impact_speed":      raw_objs[2],
        "scaled_delta_vs1":      scaled_objs[0],
        "scaled_hold_time":      scaled_objs[1],
        "scaled_impact_speed":   scaled_objs[2],
    }


def _make_snapshot(gen, population, sigmas_per_slot, parent_idx_per_slot, bounds):
    """Build snapshot rows for one generation.

    Each individual is also tagged with (_origin_gen, _snapshot_idx) so that
    later generations can locate this row when they need to fill in the
    'offspring_ind_number' or 'chosen' columns.
    """
    contributions = _hv_contributions(population)
    rows = []
    for i, ind in enumerate(population):
        row = _build_pop_row(
            ind, gen, i,
            sigmas_per_slot[i],
            parent_idx_per_slot[i],
            contributions[i],
            bounds,
        )
        ind._origin_gen   = gen
        ind._snapshot_idx = i
        rows.append(row)
    return rows


def _write_parents_csv(out_dir, gen, strategy, bounds):
    """Write one CSV per save trigger recording the surviving μ parent set.

    Distinct from ``population_gen_*.csv``: the offspring snapshot
    records the *pre-selection* candidate batch (which includes sentinel
    offspring that selection then drops).  This file records the
    *post-selection* parents - the actual search distribution that
    seeds the next generation.

    One row per slot in ``strategy.parents``.  Columns include
    fitness (the missing piece in strategy_per_gen.csv), constraint
    state, sentinel flags, σ and psucc per slot, and design variables
    in both raw and scaled form.
    """
    import csv
    from pathlib import Path

    fieldnames = [
        "generation", "parent_slot",
        "lineage_id",
        "sigma", "psucc",
        "feasible",
        "pitot3_sentinel", "spark_sentinel",
        "g_al",
        # Raw (physical) design variables
        "raw_pct_he", "raw_driver_p", "raw_p4",
        "raw_d_throat", "raw_reservoir_p", "raw_buffer_length",
        # Scaled (algorithm-space [1, 2]) design variables
        "scaled_pct_he", "scaled_driver_p", "scaled_p4",
        "scaled_d_throat", "scaled_reservoir_p", "scaled_buffer_length",
        # Objectives (raw + scaled).  delta_vs1 only meaningful for
        # legacy 3-objective mode; CHT_AL records it via g_al instead.
        "raw_delta_vs1", "raw_hold_time", "raw_impact_speed",
        "scaled_delta_vs1", "scaled_hold_time", "scaled_impact_speed",
    ]

    rows = []
    for i, ind in enumerate(strategy.parents):
        raw_vars    = variable_untransformation(ind, bounds)
        scaled_vars = list(ind)

        is_2d = ind.fitness.valid and len(ind.fitness.values) == 2
        if ind.fitness.valid:
            scaled_objs_raw = list(ind.fitness.values)
            if is_2d:
                raw_2d = list(unnormalise_fitness(
                    ind.fitness.values, APPROX_IDEAL_2D, APPROX_NADIR_2D,
                ))
                g_al = getattr(ind, "_g_al", None)
                al_tol = getattr(ind, "al_tol", 100.0)
                raw_dvs = (float(g_al[0]) + al_tol) if g_al is not None else None
                raw_objs    = [raw_dvs, raw_2d[0], raw_2d[1]]
                scaled_objs = [None, scaled_objs_raw[0], scaled_objs_raw[1]]
            else:
                raw_objs = list(unnormalise_fitness(
                    ind.fitness.values, APPROX_IDEAL, APPROX_NADIR,
                ))
                scaled_objs = scaled_objs_raw
        else:
            raw_objs    = [None, None, None]
            scaled_objs = [None, None, None]

        g_al = getattr(ind, "_g_al", None)
        g_al_val = float(np.asarray(g_al)[0]) if g_al is not None else None

        rows.append({
            "generation":          gen,
            "parent_slot":         i,
            "lineage_id":          getattr(ind, "lineage_id", None),
            "sigma":               float(strategy.sigmas[i]),
            "psucc":               float(strategy.psucc[i]),
            "feasible":            getattr(ind, "_feasible", None),
            "pitot3_sentinel":     bool(getattr(ind, "_pitot3_sentinel", False)),
            "spark_sentinel":      bool(getattr(ind, "_spark_sentinel", False)),
            "g_al":                g_al_val,
            "raw_pct_he":          raw_vars[0],
            "raw_driver_p":        raw_vars[1],
            "raw_p4":              raw_vars[2],
            "raw_d_throat":        raw_vars[3],
            "raw_reservoir_p":     raw_vars[4],
            "raw_buffer_length":   raw_vars[5],
            "scaled_pct_he":       scaled_vars[0],
            "scaled_driver_p":     scaled_vars[1],
            "scaled_p4":           scaled_vars[2],
            "scaled_d_throat":     scaled_vars[3],
            "scaled_reservoir_p":  scaled_vars[4],
            "scaled_buffer_length":scaled_vars[5],
            "raw_delta_vs1":       raw_objs[0],
            "raw_hold_time":       raw_objs[1],
            "raw_impact_speed":    raw_objs[2],
            "scaled_delta_vs1":    scaled_objs[0],
            "scaled_hold_time":    scaled_objs[1],
            "scaled_impact_speed": scaled_objs[2],
        })

    path = Path(out_dir) / f"parents_gen_{gen:04d}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in fieldnames})


def _mark_chosen(gen_snapshots, chosen):
    """Set chosen=True for every snapshot row whose individual survived
    selection.  Idempotent - once True, stays True."""
    for ind in chosen:
        if hasattr(ind, '_origin_gen') and hasattr(ind, '_snapshot_idx'):
            snap = gen_snapshots.get(ind._origin_gen)
            if snap is not None and 0 <= ind._snapshot_idx < len(snap):
                snap[ind._snapshot_idx]["chosen"] = True


def _fill_offspring(gen_snapshots, parents_at_generate, offspring):
    """Write each new offspring's ind_number into its parent's snapshot row.

    Only the *first* offspring is recorded per row.  A long-surviving parent
    that produces one offspring per generation will have its first-gen
    offspring recorded; subsequent ones are tracked instead in any snapshot
    that captures the parent again (e.g. via failure-substitution).
    """
    for off in offspring:
        if not hasattr(off, '_ps'):
            continue
        tag, p_idx = off._ps
        if tag != "o" or p_idx is None or p_idx >= len(parents_at_generate):
            continue
        parent = parents_at_generate[p_idx]
        if not (hasattr(parent, '_origin_gen') and hasattr(parent, '_snapshot_idx')):
            continue
        snap = gen_snapshots.get(parent._origin_gen)
        if snap is None or not (0 <= parent._snapshot_idx < len(snap)):
            continue
        row = snap[parent._snapshot_idx]
        if row.get("offspring_ind_number") is None:
            row["offspring_ind_number"] = off.ind_number


def _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders,
                  strategy=None, bounds=None):
    """Write per-generation plots and population CSVs.

    All known snapshots are re-written every save trigger so that lazily-
    filled fields (chosen, offspring_ind_number) propagate to disk as the
    information becomes available.

    Plots are routed by fitness dimensionality (auto-detected from
    fitness_history).  3-D fitness goes to the legacy 5-plot bundle;
    2-D fitness (CHT_AL) goes to a single hold_time-vs-impact_speed
    plot - there is no third axis to scatter on.

    When ``strategy`` is supplied, also writes the surviving μ parent
    set to parents_gen_NNNN.csv and threads the parents' fitness into
    the 2-D Pareto plot so the purple highlight shows the post-selection
    Pareto front rather than the last offspring batch.
    """
    pop_dir = folders["population"]
    for g, rows in gen_snapshots.items():
        write_population_csv(pop_dir / f"population_gen_{g:04d}.csv", rows)

    # Surviving parents CSV.  Skipped if no strategy passed (legacy
    # callers) or the parents folder isn't in the layout.
    parent_fitness = None
    if strategy is not None and "parents" in folders:
        _write_parents_csv(folders["parents"], bookshelf_gen, strategy, bounds)
        parent_fitness = [
            tuple(ind.fitness.values)
            for ind in strategy.parents
            if ind.fitness.valid
        ]

    is_2d = (
        len(fitness_history) > 0
        and len(fitness_history[0]) == 2
    )

    if is_2d:
        # CHT_AL: only one Pareto plot (the 2-D one) - delta_vs1 is no
        # longer a Pareto axis; it is logged in the AL diagnostics csv.
        plot_holdtime_impactspeed_2d(
            fitness_history,
            MU=MU, gen=bookshelf_gen,
            out_dir=folders["pareto_holdtime_impactspeed"],
            parent_fitness=parent_fitness,
        )
    else:
        # Legacy 3-objective plots (unchanged behaviour).
        plot_objective_space(fitness_history, 'delta_vs1', 'hold_time',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_dvs1_holdtime"])
        plot_objective_space(fitness_history, 'delta_vs1', 'impact_speed',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_dvs1_impactspeed"])
        plot_objective_space(fitness_history, 'hold_time', 'impact_speed',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_holdtime_impactspeed"])
        plot_objective_space_3d(fitness_history,
                                MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_3d"])
        plot_objective_space_heatmap(fitness_history,
                                     MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_heatmap"])

    # Belt-and-braces: each plot_* function calls plt.close() but only
    # on the current figure.  plt.close('all') guarantees no pyplot
    # state survives a save burst, which over hundreds of generations
    # would otherwise compound into a noticeable RSS drift.
    plt.close('all')


def _append_strategy_per_gen_row(out_dir, gen, strategy):
    """Append one row of σ, psucc and lineage state to strategy_per_gen.csv.

    Called once per generation immediately after toolbox.update(), so
    self.sigmas / self.psucc reflect the post-update parent set (i.e.
    the parents that will seed the *next* generate() call).

    Schema: generation, mu, summary stats (mean/min/max for σ and
    psucc), then per-slot lineage_<i>, sigma_<i>, psucc_<i>.  μ is
    constant within a run, so the header is fixed at first write.
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_per_gen.csv"

    sigmas = list(strategy.sigmas)
    psucc  = list(strategy.psucc)
    mu     = len(sigmas)
    lineage_ids = [
        getattr(p, "_lineage_id", None) for p in strategy.parents
    ]

    fieldnames = (
        ["generation", "mu",
         "mean_sigma", "min_sigma", "max_sigma",
         "mean_psucc", "min_psucc", "max_psucc",
         "gens_silent", "sigma_floor_active",
         "archive_size",
         "archive_eps_culled_gen",
         "archive_eps_culled_total",
         # Diag-1 (split): PITOT3 vs SPARK vs both, plus the union count
         # for ranking.  A parent can be both (PITOT3 + SPARK joint
         # failure); n_both counts that overlap.
         "n_pitot3_sentinels",
         "n_spark_sentinels",
         "n_both_sentinels",
         "n_sentinels"]            # union: pitot3 + spark - both
        + [f"lineage_{i}" for i in range(mu)]
        + [f"sigma_{i}"   for i in range(mu)]
        + [f"psucc_{i}"   for i in range(mu)]
    )

    # Silent-streak length, reported per generation.
    silent = getattr(strategy, "_gens_silent_count", 0)
    floor_active = False

    # Diag-1 (split): differentiate PITOT3 sentinels (delta_vs1 = 3585;
    # fit may be real) from SPARK sentinels (fit ~= (1, 1); g_al may be
    # real).  Reading flags rather than re-computing keeps the
    # definitions consistent with the filters in _cheap_al_proxy etc.
    n_pitot3 = 0
    n_spark  = 0
    n_both   = 0
    for p in strategy.parents:
        pitot3 = getattr(p, "_pitot3_sentinel", False)
        spark  = getattr(p, "_spark_sentinel",  False)
        if pitot3:
            n_pitot3 += 1
        if spark:
            n_spark += 1
        if pitot3 and spark:
            n_both += 1
    n_sent = n_pitot3 + n_spark - n_both    # union, no double counting

    row = {
        "generation": gen,
        "mu":         mu,
        "mean_sigma": float(np.mean(sigmas)),
        "min_sigma":  float(np.min(sigmas)),
        "max_sigma":  float(np.max(sigmas)),
        "mean_psucc": float(np.mean(psucc)),
        "min_psucc":  float(np.min(psucc)),
        "max_psucc":  float(np.max(psucc)),
        "gens_silent":         silent,
        "sigma_floor_active":  bool(floor_active),
        "archive_size":        len(getattr(strategy, "external_archive", [])),
        "archive_eps_culled_gen":
            getattr(strategy, "archive_eps_culled_gen", 0),
        "archive_eps_culled_total":
            getattr(strategy, "archive_eps_culled_total", 0),
        "n_pitot3_sentinels":  n_pitot3,
        "n_spark_sentinels":   n_spark,
        "n_both_sentinels":    n_both,
        "n_sentinels":         n_sent,
    }
    for i in range(mu):
        row[f"lineage_{i}"] = lineage_ids[i]
        row[f"sigma_{i}"]   = float(sigmas[i])
        row[f"psucc_{i}"]   = float(psucc[i])

    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# Main evolution loop

def main(experiment_type, seed_population=None):
    """Run one experiment.

    seed_population : array-like (n, 6) or None
        If given, rows are used as the initial population in NORMALISED
        [1, 2]^6 space (e.g. ``x_norm`` from init_population_l1d.py).  When
        there are at least MU rows the first MU are used; fewer are padded
        with random feasible individuals.  None -> the legacy random
        uniform initialisation.
    """
    s1 = time.time()

    # Experiment parameters
    N           = 6
    pop_size    = experiment_type[1]
    MU, LAMBDA  = pop_size, pop_size
    # Number of generations.  Read from the config's per-experiment `n_gen`
    # (tuple slot 9); older tuples without it fall back to the historic 350.
    NGEN        = experiment_type[9] if len(experiment_type) > 9 else 350
    sim_type    = experiment_type[0]
    p4_treatment = experiment_type[3]
    step_size   = experiment_type[2]
    # AL constraint tolerance (m/s on delta_vs1).  Only consumed when
    # is_al_active(sim_type); legacy sim_types ignore it.  Default 100 m/s
    # per the user-confirmed setting.  experiment_type may be a 4-tuple
    # for legacy YAML entries; fall back to the default in that case.
    al_tol = experiment_type[4] if len(experiment_type) > 4 else 100.0
    # CHT covariance shrinkage strength.  None -> let the strategy fall
    # back to its dimension-dependent default (0.5/(n+2)).  Consumed by
    # both CovarianceCHT and CHT_AL.
    cht_gamma = experiment_type[5] if len(experiment_type) > 5 else None
    # Anti-degeneration feature toggles.  Each key in this dict opts in
    # to one of the experimental fixes for the σ death spiral; all
    # default off so omitting the field reproduces baseline behaviour.
    # See the YAML header for the full schema.
    features = experiment_type[6] if len(experiment_type) > 6 else {}
    # Arnold & Hansen 2012 coefficients (ArnoldCHT / ArnoldCHT_AL only).
    # None -> let the strategy fall back to the paper defaults
    # (β = 0.1/(n+2), c_c = 1/(n+2)).  Ignored by non-Arnold sim_types.
    arnold_beta = experiment_type[7] if len(experiment_type) > 7 else None
    arnold_cc   = experiment_type[8] if len(experiment_type) > 8 else None

    print(f"Step Size = {step_size}")
    print(f'Pop Size = {pop_size}\n')
    if is_al_active(sim_type):
        print(f'AL tolerance (delta_vs1 ≤): {al_tol} m/s')
    if cht_method(sim_type) == 'chocat':
        print(f'cht_gamma = {cht_gamma if cht_gamma is not None else "default (0.5/(n+2))"}')
    if cht_method(sim_type) == 'arnold':
        print(f'arnold_beta = {arnold_beta if arnold_beta is not None else "default (0.1/(n+2))"}, '
              f'arnold_cc = {arnold_cc if arnold_cc is not None else "default (1/(n+2))"}')
    if features:
        active_features = [k for k, v in features.items() if v not in (None, False, 0)]
        if active_features:
            print(f'Active features: {active_features}')
        else:
            print('Active features: none (baseline)')
    else:
        print('Active features: none (baseline)')

    # Output directory layout
    # Per-sim_type constants: legacy modes write a 3-objective tree;
    # CHT_AL writes a 2-objective tree with an extra al_diagnostics dir.
    (results_category, run_prefix, output_folders,
     Individual_cls, ideal_point, nadir_point) = _run_constants(sim_type)
    run_dir = setup_run_directory(results_category, run_prefix)
    folders = setup_subfolders(run_dir, output_folders)
    print(f"Run directory: {run_dir}")

    # Per-mode HV calculator.  Rebinds the global ``pop_hypervolumes``
    # name to a local 2-D / 3-D variant - the global lookup at the
    # bottom of the loop will hit this local instead.  No global access
    # leaks because no other site in main.py reads pop_hypervolumes.
    #
    # Two parallel HV calculators are kept:
    #   * ``pop_hypervolumes``        - ref at (0, ..., 0) in fit_neg space
    #                                   (= IDEAL of fit_2d).  This is a
    #                                   utopia-distance metric: every
    #                                   point contributes fit_2d[0] *
    #                                   fit_2d[1], so LOWER HV = front
    #                                   closer to the ideal.  Does not
    #                                   reward front spread.  Retained
    #                                   for back-compat with the existing
    #                                   convergence_data.txt format.
    #   * ``pop_hypervolumes_nadir``  - ref at (1, ..., 1) in fit_2d space
    #                                   directly (no negation).  Standard
    #                                   MOO convention: HIGHER HV = better
    #                                   (more area dominated below ref);
    #                                   rewards both convergence toward
    #                                   ideal AND front spread.  This is
    #                                   the headline convergence metric.
    n_obj = 2 if is_al_active(sim_type) else 3
    pop_hypervolumes       = HyperVolume(np.zeros(n_obj))
    pop_hypervolumes_nadir = HyperVolume(np.ones(n_obj))

    # Logbook initialisation
    gen_counter = 0
    toolbox.logbook.add('generation',          gen_counter)
    toolbox.logbook.add('fixer_count',         {f"{gen_counter}": 0})
    toolbox.logbook.add('time_spent_fixing',   0)
    toolbox.logbook.add("time taken",          0)
    toolbox.logbook.add("No. individuals that failed objective tests", 0)
    toolbox.logbook.add("No. individuals that produced no hold time",  0)
    toolbox.logbook.add("No. individuals that failed constraint tests", 0)
    toolbox.logbook.add("hypervolume",         [0 for _ in range(1, NGEN + 1)])
    # Standard-MOO HV with ref at the nadir corner (1,...,1) in fit_2d
    # space.  Higher = better; rewards both convergence and spread.
    # This is the headline convergence metric; ``hypervolume`` above is
    # the legacy utopia-distance variant retained for back-compat.
    toolbox.logbook.add("hypervolume_nadir",   [0 for _ in range(1, NGEN + 1)])
    # Diag-4: HV computed only over non-sentinel parents (those whose
    # heavy evaluators succeeded - fitness ≠ (1, 1)).  Removes the
    # sentinel-rate confounder when comparing HV trajectories across
    # runs.  Same length as "hypervolume", zero-padded.
    toolbox.logbook.add("hypervolume_nonsentinel",
                        [0 for _ in range(1, NGEN + 1)])
    # Per-generation count of offspring that passed the feasibility check
    # (and were therefore evaluated by SPARK + PITOT3).  A persistently
    # low number signals stagnation - the search ellipsoid is wider than
    # the feasible region so almost every offspring is rejected.  Index
    # is zero-based on generation (slot k holds gen k+1's count).
    toolbox.logbook.add("feasible_offspring_count", [0 for _ in range(1, NGEN + 1)])
    # CovarianceCHT only: how many CHT-and-resample iterations were
    # needed each generation before either every offspring became
    # feasible or the cap was hit.  0 means the first batch was already
    # all-feasible; max_iterations means the cap was reached.
    toolbox.logbook.add("resample_iterations", [0 for _ in range(1, NGEN + 1)])

    # Give the evaluate module a reference to the logbook so it can update
    # counters without accessing a global toolbox.
    set_logbook(toolbox.logbook)

    # Design-variable bounds
    bounds = BOUNDS

    # Population initialisation
    def pop_init(MU):
        percent_he_list     = np.random.uniform(he_lower,            he_upper,            (MU, 1))
        D_throat_list       = np.random.uniform(D_throat_lower,      D_throat_upper,      (MU, 1))
        driver_p_list       = np.random.uniform(driver_p_lower,      driver_p_upper,      (MU, 1))
        buffer_length_list  = np.random.uniform(buffer_length_lower, buffer_length_upper, (MU, 1))

        p4_list = np.zeros_like(percent_he_list)
        for i in range(MU):
            if 1190.63 * driver_p_list[i] < p4_upper:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], 1190.63 * driver_p_list[i])
            else:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], p4_upper)

        reservoir_p_list = [np.random.uniform(driver_p_list[i], reservoir_upper) for i in range(MU)]

        return [
            [percent_he_list[i][0], driver_p_list[i][0], p4_list[i][0],
             D_throat_list[i][0],   reservoir_p_list[i][0], buffer_length_list[i][0]]
            for i in range(MU)
        ]

    i = 0
    if seed_population is not None:
        # Seeded start: rows are already in normalised [1, 2]^6 space, so
        # they are NOT re-transformed.  Use the first MU; if too few were
        # supplied, pad with random feasible individuals so the per-parent
        # state arrays still have length MU.
        seed = [list(map(float, row))
                for row in np.asarray(seed_population, dtype=float)]
        if len(seed) >= MU:
            init_pop_transformed = seed[:MU]
            if len(seed) > MU:
                print(f"Seed population has {len(seed)} individuals; using the "
                      f"first {MU} to match pop_size.")
        else:
            pad = variable_transformation(pop_init(MU - len(seed)), bounds)
            init_pop_transformed = seed + list(pad)
            print(f"Seed population has {len(seed)} individuals (< MU={MU}); "
                  f"padded with {MU - len(seed)} random feasible individuals.")
        print(f"Seeded initial population from provided individuals (MU={MU}).")
    else:
        init_pop_untransformed = pop_init(MU)
        init_pop_transformed   = variable_transformation(init_pop_untransformed, bounds)

    # The strategy assumes every parent it starts with is feasible - Pareto
    # selection later relies on every parent having a valid fitness, and
    # length(self.parents) must equal mu so per-parent state arrays don't
    # shrink and break later generate() calls.  pop_init enforces some
    # constraints by construction (driver_p < reservoir_p, p4 within
    # bounds) but not the compression-ratio range, so a small fraction
    # (~1%) of initial individuals fail the feasibility check.  Resample
    # any infeasible slots one at a time until the whole pop is feasible.
    from problem.feasibility import evaluate_constraints, is_feasible
    _MAX_RESAMPLE_ATTEMPTS = 1000
    for slot in range(MU):
        attempt = 0
        while not is_feasible(evaluate_constraints(init_pop_transformed[slot], bounds)):
            attempt += 1
            if attempt > _MAX_RESAMPLE_ATTEMPTS:
                raise RuntimeError(
                    f"Could not generate a feasible initial individual for slot "
                    f"{slot} after {_MAX_RESAMPLE_ATTEMPTS} attempts.  Check "
                    f"that pop_init's sampling ranges are consistent with "
                    f"problem.feasibility.evaluate_constraints."
                )
            # pop_init(1) returns a length-1 list; replace just this slot.
            init_pop_transformed[slot] = variable_transformation(pop_init(1), bounds)[0]

    # Use the dimension-appropriate Individual class - Individual2D for
    # CHT_AL (2-objective fitness), Individual for the legacy 3-objective
    # sim_types.  Picked once via _run_constants() above so this is the
    # only branch needed.
    population = [Individual_cls(x) for x in init_pop_transformed]
    initial_population = population

    for ind in population:
        ind.ind_number = i
        ind.bounds     = bounds
        # al_tol is read by problem.evaluate.evaluate() when computing
        # g_al = delta_vs1 - al_tol in CHT_AL mode.  Setting it on every
        # individual (regardless of sim_type) is harmless: legacy paths
        # never consult it.
        ind.al_tol = al_tol
        i += 1

    # Silence L1d subprocess streaming for the whole run.  The initial
    # population is evaluated serially here in the PARENT (before the Pool
    # exists), so the flag must be set in the parent too - not just in the
    # Pool initializer.  On Linux's default fork start-method the workers
    # spawned below also inherit this False; the explicit initializer makes
    # it robust under spawn as well.
    l1d_job.STREAM_L1D_OUTPUT = False

    for ind in population:
        ind.sim_type   = sim_type
        ind.normalised = normalised
        fit, g, g_al = toolbox.evaluate(ind)
        ind._g = g
        ind._g_al = g_al
        _store_raw_delta_vs1(ind, g_al)   # Task 2: cache delta_vs1 for refresh
        ind._feasible = fit is not None
        if ind._feasible:
            ind.fitness.values = fit
        # Detect PITOT3 / SPARK sentinels for downstream filters.
        # Only meaningful in CHT_AL mode (legacy modes route failures
        # through a different sentinel-recovery path).
        if is_al_active(sim_type):
            _detect_sentinels(ind, fit, g_al)
        else:
            ind._pitot3_sentinel = False
            ind._spark_sentinel = False

    # Strategy and multiprocessing setup
    # n_constraints is the length of the box+physical constraint vector;
    # the Arnold modes need it to size one v_j accumulator per constraint
    # per parent.  Compute once from a feasible seed (every initial
    # individual is feasible by construction here).
    n_constraints = len(evaluate_constraints(initial_population[0], bounds))
    strategy = StrategyMultiObjective(
        population, sigma=step_size,
        mu=MU, lambda_=LAMBDA,
        sim_type=sim_type, p4_treatment=p4_treatment,
        bounds=bounds,
        al_tol=al_tol,                # consumed only when AL family
        cht_gamma=cht_gamma,          # None -> strategy default (0.5/(n+2)); Chocat only
        arnold_beta=arnold_beta,      # None -> paper default 0.1/(n+2); Arnold only
        arnold_cc=arnold_cc,          # None -> paper default 1/(n+2); Arnold only
        n_constraints=n_constraints,  # required by the Arnold modes
        features=features,            # anti-degeneration toggles (see YAML header)
        logbook=toolbox.logbook,      # injected - no global access inside cmaes.py
    )
    toolbox.register("generate", strategy.generate, Individual_cls)
    toolbox.register("update",   strategy.update)

    # Bootstrap AL coefficients from the initial population's data.
    # Idempotent: pycma's set_coefficients short-circuits once
    # _initialized is fully True; we still call it again every generation
    # below until it is, to refine on additional samples.
    #
    # Filter sentinel individuals (evaluator failures
    # encoded as fitness ~= (1, 1)) from the bootstrap sample.  Their
    # g_al is a sentinel-implied value (PITOT3 -> +3485 m/s, i.e.
    # 3585 - al_tol) that inflates iqr(G) and biases the initial μ_AL
    # too small.
    if is_al_active(sim_type):
        F_pop, G_AL = [], []
        for ind in population:
            if not ind._feasible:
                continue
            if not ind.fitness.valid:
                continue
            if _is_sentinel(ind):
                continue   # PITOT3 or SPARK sentinel - exclude from bootstrap
            F_pop.append(sum(ind.fitness.values))
            G_AL.append(ind._g_al)
        if F_pop:
            strategy.init_al(F_pop, G_AL)
            print(f"AL bootstrapped on {len(F_pop)} real parents: "
                  f"lam={strategy.al.lam}, mu={strategy.al.mu}")

    # maxtasksperchild caps the number of evaluations a worker handles
    # before the Pool kills and respawns it.  This bounds per-worker
    # memory creep from PITOT3 / SPARK / gdtk.gas, all of which retain
    # state across calls (Lua VMs, cached gas-model objects, GasState /
    # Driver / Tube instances).  Without this, after a few hundred
    # generations the workers' RSS sums up to all available system RAM
    # and the kernel OOM-killer terminates the parent process - visible
    # as "Killed" followed by a flood of worker BrokenPipeErrors.
    #
    # 50 is a balance: large enough that the worker startup cost
    # (loading PITOT3, gdtk, etc.) doesn't dominate the per-eval cost,
    # small enough that any single worker's heap stays bounded.  At
    # pop_size=12, each worker handles ~4 generations before being
    # recycled.
    pool = multiprocessing.Pool(maxtasksperchild=50,
                                initializer=_quiet_l1d_worker)
    toolbox.register("map", pool.map)

    # Snapshot bookkeeping
    gen_snapshots = {}
    gen_snapshots[0] = _make_snapshot(
        0, population,
        sigmas_per_slot=[step_size] * MU,
        parent_idx_per_slot=[None] * MU,
        bounds=bounds,
    )

    # Seed fitness_history with the initial population so the "every individual
    # ever sampled" plots include the starting points, not just offspring.
    fitness_history = [tuple(ind.fitness.values) for ind in population]

    # Per-generation history for the end-of-run post-processing figures
    # (AL family only; stays empty and unused otherwise).  Each is a small
    # list appended to once per generation from the elitist parent set -
    # negligible memory/CPU next to the heavy evaluations.
    #   epsilon_history   : [(gen, al_tol), ...]        - ε schedule
    #   gal_history       : [(gen, [g_al, ...]), ...]   - per-parent g_AL
    #   diversity_history : [(gen, delta), ...]         - Deb's Δ of the front
    epsilon_history   = []
    gal_history       = []
    diversity_history = []

    # Evolution
    for gen in range(NGEN):
        toolbox.logbook.bookshelf['generation'] += 1
        bookshelf_gen = toolbox.logbook.bookshelf['generation']
        print('\n')
        print('*' * 30)
        print(f"Generation {bookshelf_gen}")
        print('*' * 30)

        # Snapshot strategy state BEFORE generate(): update() mutates
        # self.sigmas in-place, so we need a frozen view of which step size
        # was used to mutate each parent into its offspring this generation.
        sigmas_at_generate  = list(strategy.sigmas)
        parents_at_generate = list(strategy.parents)

        parents    = population
        population = toolbox.generate()

        # Each new offspring's _ps now points at its parent.  Walk back to
        # the parent's earlier snapshot row and record the offspring's
        # ind_number there.
        _fill_offspring(gen_snapshots, parents_at_generate, population)

        i = 0
        # Read the current AL tolerance
        # from the strategy (returns the static value when no schedule
        # is configured).  Per-generation update - offspring need to
        # see the new tol before evaluate() reads x.al_tol.
        current_al_tol = (
            strategy.current_al_tol()
            if hasattr(strategy, "current_al_tol") else al_tol
        )
        for ind in population:
            ind.normalised = normalised
            ind.ind_number = i
            ind.bounds     = bounds
            # Offspring are freshly constructed by strategy.generate() - they
            # do NOT inherit sim_type or al_tol from the parent.  Without
            # these, evaluate() falls into its legacy 3-objective branch
            # for an Individual2D and the length-2-vs-length-3 fitness
            # assignment later trips DEAP's assertion.
            ind.sim_type = sim_type
            ind.al_tol   = current_al_tol
            i += 1

        # CHT-and-resample loop (Chocat 2015 Algorithm 3 step 3-2): for
        # CovarianceCHT and CHT_AL, infeasible offspring drive a covariance
        # shrinkage of each parent's Cholesky factor and are then
        # resampled from the tighter distribution.  Cheap because the
        # feasibility check does NOT call SPARK / PITOT3 - it only
        # evaluates the constraint vector via problem.feasibility.
        # Mutates population in place and tags every Individual with
        # ._g and ._feasible so the post-eval loop and update()'s
        # post-resample CHT can both consume them.
        #
        # In the AL modes the CHT only operates on box+physical g (the
        # 18-element vector); delta_vs1 is handled separately by the AL
        # in selection, not via CHT shrinkage.
        if is_cht_active(sim_type) or is_resample_active(sim_type):
            # The same _check is shared by every pre-evaluation handler,
            # so they reject on an identical feasibility set.
            def _check(ind):
                g = evaluate_constraints(ind, bounds)
                return is_feasible(g), g

            if cht_method(sim_type) == 'chocat':
                n_iter = strategy.resample_infeasibles(
                    population, feasibility_check=_check, max_iterations=5,
                )
                print(f"resample iterations this gen = {n_iter}")
                toolbox.logbook.bookshelf['resample_iterations'][gen] = n_iter
            elif cht_method(sim_type) == 'arnold':
                # Arnold (ArnoldCHT / ArnoldCHT_AL): one-shot per parent.
                # Infeasibles are consumed via Eq. 6 + Eq. 7 BEFORE
                # evaluation and marked _feasible=False so evaluate()
                # short-circuits them. The slot then contributes no
                # candidate that generation.
                strategy.apply_arnold_infeasibility(
                    population, feasibility_check=_check,
                )
                toolbox.logbook.bookshelf['resample_iterations'][gen] = 0
            else:
                # Resampling / Resampling_AL: pure rejection.  Infeasible
                # slots are redrawn from the UNCHANGED (σ, A) until feasible
                # or the cap is hit; no covariance surgery, no repair.  A
                # still-infeasible slot after the cap is dropped by selection.
                resample_cap = (
                    features.get("resample_max_iterations")
                    if isinstance(features, dict) else None
                ) or 100
                n_redraws = strategy.resample_infeasibles_rejection(
                    population, feasibility_check=_check,
                    max_iterations=resample_cap,
                )
                print(f"rejection resample draws this gen = {n_redraws}")
                toolbox.logbook.bookshelf['resample_iterations'][gen] = n_redraws

        # Retry logic for transient evaluation failures
        try:
            fitnesses = toolbox.map(toolbox.evaluate, population)
        except Exception:
            time.sleep(1)
            try:
                fitnesses = toolbox.map(toolbox.evaluate, population)
            except Exception:
                time.sleep(1)
                try:
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                except Exception:
                    time.sleep(1)
                    fitnesses = toolbox.map(toolbox.evaluate, population)

        fixed = False
        for i, (ind, result) in enumerate(zip(population, fitnesses)):
            fit, g, g_al = result
            ind._g = g
            ind._g_al = g_al
            _store_raw_delta_vs1(ind, g_al)   # Task 2: cache delta_vs1 for refresh

            if fit is None:
                # Skipped by feasibility short-circuit.  Leave fitness
                # unset so DEAP's selection treats this individual as
                # invalid.  The CHT consumes ind._g to update covariance.
                # In CHT_AL mode, evaluate() already routed PITOT3 / SPARK
                # failures to fit=None, g_al=None - those individuals are
                # excluded from AL coefficient adaptation by construction.
                ind._feasible = False
                ind._pitot3_sentinel = False
                ind._spark_sentinel = False
                continue

            ind._feasible = True
            # Detect PITOT3 / SPARK sentinels.  In CHT_AL mode fit_2d may
            # be real even when delta_vs1 is the PITOT3 sentinel (SPARK
            # succeeded, PITOT3 failed) - detecting this case requires
            # checking g_al + al_tol, not fit alone.
            if is_al_active(sim_type):
                _detect_sentinels(ind, fit, g_al)
            else:
                ind._pitot3_sentinel = False
                ind._spark_sentinel = False
            # Failure-sentinel detection in legacy 3-objective mode:
            # fit[0] is normalised delta_vs1 and == 1.0 means PITOT3 hit
            # its 3585 m/s sentinel.  In CHT_AL mode that path is already
            # caught inside evaluate() (returns fit=None), so skip the
            # check rather than indexing a 2-tuple at slot [0] which would
            # be hold_time, not delta_vs1.
            normalised_shock_speed = (
                fit[0] if not is_al_active(sim_type) else None
            )

            if normalised_shock_speed == 1.0:
                # PITOT3 / SPARK reported the failure sentinel even though
                # the candidate passed our feasibility check.  Recover by
                # substituting the strategy parent that this offspring
                # was generated from.
                #
                # The original recovery used `parents[ind.ind_number]`
                # (the previous generation's offspring batch), which
                # worked when every offspring was guaranteed feasible by
                # repair.  In CovarianceCHT mode infeasibles are kept
                # unevaluated, so an arbitrary previous-gen offspring may
                # have no fitness - DEAP then returns () for fitness.values
                # and the length-3 assignment below crashes.
                #
                # strategy.parents are guaranteed feasible by selection,
                # so they always have a length-3 fitness tuple.  Index
                # via ind._ps[1] (the donor parent recorded in generate()).
                p_idx = ind._ps[1] if hasattr(ind, "_ps") else None
                replacement = (
                    strategy.parents[p_idx]
                    if (p_idx is not None
                        and 0 <= p_idx < len(strategy.parents)
                        and strategy.parents[p_idx].fitness.valid)
                    else None
                )
                if replacement is None:
                    # Genuine corner case: we couldn't find a feasible
                    # replacement.  Fall back to marking this slot as
                    # infeasible so the CHT consumes its violation info
                    # and selection ignores it.
                    ind._feasible = False
                    continue

                new_fitness = replacement.fitness.values
                population[i] = replacement
                population[i].fitness.values = new_fitness
                population[i].ind_number = i
                population[i]._g = getattr(replacement, "_g", g)
                population[i]._feasible = True
                fixed = True
                fitness_history.append(new_fitness)
            else:
                ind.fitness.values = fit
                fitness_history.append(fit)

        # Per-generation feasibility count.  This is the diagnostic that
        # tells stagnation ("HV constant because zero offspring made it
        # through") apart from "rare improvements".  Recorded into the
        # logbook so the convergence_data.txt / summary writers can
        # surface it later.
        n_feasible = sum(
            1 for ind in population if getattr(ind, "_feasible", False)
        )
        print(f"feasible offspring this gen = {n_feasible} / {len(population)}")
        toolbox.logbook.bookshelf['feasible_offspring_count'][gen] = n_feasible

        # Snapshot the just-evaluated population for this generation.
        sigmas_per_slot     = list(sigmas_at_generate)
        parent_idx_per_slot = []
        for ind in population:
            if hasattr(ind, '_ps') and ind._ps[0] == "o":
                parent_idx_per_slot.append(ind._ps[1])
            else:
                parent_idx_per_slot.append(None)
        gen_snapshots[bookshelf_gen] = _make_snapshot(
            bookshelf_gen, population,
            sigmas_per_slot, parent_idx_per_slot, bounds,
        )

        # Task 2: refresh AL constraints against the CURRENT scheduled
        # al_tol before selection.  Offspring were just evaluated at
        # current_al_tol (no-op here), but surviving parents were evaluated
        # in earlier, wider-tol generations - recompute their g_al from the
        # fixed raw delta_vs1 so the AL penalty (inside update()'s
        # selection) and the proxy mean see a consistent, current
        # constraint.  Kills the stale-ε / frozen-parent-residual artifact.
        if is_al_active(sim_type):
            strategy.refresh_al_constraints(population)

        toolbox.update(population)

        # Persist post-update strategy state (σ, psucc, lineage per slot)
        # for every sim_type.  Cheap (~one CSV row per gen) and lets us
        # see the σ trajectory directly - needed to test the
        # death-spiral hypothesis from the CHT analysis.
        _append_strategy_per_gen_row(
            folders["strategy_diagnostics"],
            gen=bookshelf_gen,
            strategy=strategy,
        )

        # Augmented Lagrangian coefficient update (CHT_AL only)
        # Order matters: this runs AFTER toolbox.update() so the proxy
        # we feed it is the post-selection parent set - i.e. the search
        # distribution that will seed the next generate() call.  This is
        # the closest analogue to the paper's "m^(t+1)" in MOO without
        # paying for an extra centroid evaluation.
        #
        # We also re-call init_al each gen until pycma's set_coefficients
        # decides it is fully initialised (sign_average balanced, see the
        # _initialized array) - pycma short-circuits idempotently once
        # the initial-conditions are met, so the cost is negligible.
        if is_al_active(sim_type):
            F_proxy, g_al_proxy, proxy_stats = _cheap_al_proxy(strategy)
            if F_proxy is not None:
                # Refine bootstrap on additional g_al samples whilst not
                # yet fully initialised.  No-op once is_initialized=True.
                # Exclude sentinels from the bootstrap iqr scale,
                # matching the proxy filter so the initial μ_AL isn't
                # calibrated against sentinel-inflated g.
                if not strategy.al.is_initialized:
                    F_pop_now, G_AL_now = [], []
                    for p in strategy.parents:
                        if not p.fitness.valid:
                            continue
                        if getattr(p, "_g_al", None) is None:
                            continue
                        if _is_sentinel(p):
                            continue   # PITOT3 or SPARK sentinel
                        F_pop_now.append(sum(p.fitness.values))
                        G_AL_now.append(p._g_al)
                    if F_pop_now:
                        strategy.init_al(F_pop_now, G_AL_now)
                strategy.update_al(F_proxy, g_al_proxy, proxy_stats=proxy_stats)
                print(f"AL: lam={strategy.al.lam}, mu={strategy.al.mu}, "
                      f"g_al_proxy={g_al_proxy}")

        # Drain CHT diagnostics for this generation.  Must happen AFTER
        # update(), because update()'s post-eval CHT pass also appends to
        # the buffer.  drain_and_persist clears the buffer in place, so
        # next generation starts clean.  Cheap when sim_type isn't
        # CovarianceCHT or CHT_AL (buffer is always empty).
        if cht_method(sim_type) == 'chocat':
            _cht_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["cht_diagnostics"],
                n_resample_iterations=toolbox.logbook.bookshelf['resample_iterations'][gen],
                n_infeasible_post_resample=(LAMBDA - n_feasible),
                n_lambda=LAMBDA,
            )

        # Drain Arnold CHT diagnostics for this generation.  Mirrors the
        # Chocat block but writes arnold_per_*.csv.  n_infeasible = the
        # offspring that produced no selection candidate this gen (Arnold
        # drops infeasibles rather than resampling them).  Figures are NOT
        # drawn here - only the CSVs are appended every gen; the four
        # figures are generated once at end-of-run (see plot_all below).
        elif cht_method(sim_type) == 'arnold':
            _arnold_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["arnold_diagnostics"],
                n_infeasible=(LAMBDA - n_feasible),
                n_lambda=LAMBDA,
            )

        # Drain resample diagnostics for this generation.  Mirrors the Arnold
        # block but writes resample_per_gen.csv from the per-infeasible-draw
        # buffer (initial offspring + every rejected redraw).  The heatmap is
        # drawn once at end-of-run (see _resample_plot_all below).
        if is_resample_active(sim_type):
            _resample_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["resample_diagnostics"],
                n_lambda=LAMBDA,
            )

        # Drain AL diagnostics (one row per generation) - only writes
        # anything when is_al_active(sim_type); for other sim_types the
        # buffer is empty and this is a no-op write of zero rows.
        if is_al_active(sim_type):
            _al_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["al_diagnostics"],
            )

        # Mark every snapshot row whose individual is still in
        # strategy.parents.  This catches both freshly-chosen offspring and
        # surviving older parents.
        _mark_chosen(gen_snapshots, strategy.parents)

        # Post-processing history (AL family only)
        # Record, for THIS generation's elitist parent set: the scheduled ε
        # (al_tol), each real parent's g_AL, and Deb's Δ of the parent
        # front.  Parents carry the CURRENT-tol g_al because
        # refresh_al_constraints() ran before update() this generation.
        # Sentinels (heavy-evaluator failures) are excluded so the cloud
        # and the spread metric reflect only trustworthy points - matching
        # the archive / proxy filters elsewhere.
        if is_al_active(sim_type):
            gal_vals = [
                float(np.asarray(p._g_al)[0])
                for p in strategy.parents
                if p.fitness.valid
                and getattr(p, "_g_al", None) is not None
                and not _is_sentinel(p)
            ]
            parent_pairs = [
                tuple(p.fitness.values)
                for p in strategy.parents
                if p.fitness.valid and not _is_sentinel(p)
            ]
            epsilon_history.append((bookshelf_gen, current_al_tol))
            gal_history.append((bookshelf_gen, gal_vals))
            diversity_history.append(
                (bookshelf_gen, _diversity_for_set(parent_pairs)["delta"])
            )

        # HV is computed on the elitist parent set (size = mu, constant across
        # generations) rather than raw offspring. This removes the cardinality
        # noise that produced the discrete-plateau jumps in the convergence trace.
        parent_fitnesses = np.array([ind.fitness.values for ind in strategy.parents])
        hypervolume = pop_hypervolumes.compute(parent_fitnesses * -1)
        # Standard-MOO HV (nadir ref): pass fit_2d directly, no negation.
        # See pop_hypervolumes_nadir construction above for interpretation.
        hypervolume_nadir = pop_hypervolumes_nadir.compute(parent_fitnesses.copy())
        print(f'hypervolume (utopia ref, lower=better) = {hypervolume}')
        print(f'hypervolume (nadir ref, higher=better) = {hypervolume_nadir}')
        toolbox.logbook.bookshelf['hypervolume'][gen]       = hypervolume
        toolbox.logbook.bookshelf['hypervolume_nadir'][gen] = hypervolume_nadir

        # Diag-4: HV over non-sentinel parents only.  Sentinels sit at
        # the (1, 1) nadir and contribute ~= 0 to HV, but their inclusion
        # in the parent array distorts cross-run comparison when sentinel
        # rates differ.  Reporting a second HV stripped of sentinels lets
        # us compare runs on equal footing.  Excludes both PITOT3-only
        # and SPARK sentinels - for the cross-run HV comparison we want
        # only individuals whose entire heavy-eval succeeded.
        if is_al_active(sim_type):
            non_sent_fits = np.array([
                ind.fitness.values for ind in strategy.parents
                if ind.fitness.valid and not _is_sentinel(ind)
            ])
            if len(non_sent_fits) > 0:
                hv_ns = pop_hypervolumes.compute(non_sent_fits * -1)
            else:
                # Every parent is a sentinel - degenerate HV; report 1.0
                # to mean "no real progress".  Matches the convention of
                # the main HV trace for the empty-front case.
                hv_ns = 1.0
            toolbox.logbook.bookshelf['hypervolume_nonsentinel'][gen] = hv_ns
            print(f'hypervolume (non-sentinel) = {hv_ns}')

        # Periodic outputs every SAVE_INTERVAL generations.
        if bookshelf_gen % SAVE_INTERVAL == 0:
            _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders,
                          strategy=strategy, bounds=bounds)
            # Refresh the CHT diagnostic figure from the CSVs the drain
            # block has been appending to every generation.  The plot is
            # stateless (read-from-disk), so this is a pure side-effect
            # that doesn't need to share state with the main loop.
            # Both CovarianceCHT and CHT_AL drain CHT records (the box+
            # physical constraint handling is identical between them),
            # so both should regenerate the figure.
            if sim_type in ('CovarianceCHT', 'CHT_AL'):
                _cht_plot(folders["cht_diagnostics"], current_gen=bookshelf_gen)
            # Force a full GC pass: matplotlib's render buffers and the
            # transient numpy arrays in the CHT covariance update can
            # accumulate as uncollected garbage between gc cycles, and
            # over hundreds of generations that drift adds up to hundreds
            # of MB.  Doing this just after each save burst is the
            # natural pause point in the loop.
            gc.collect()
            # Surface RSS for this Python process so memory growth is
            # visible in real time, not only after an OOM.  ru_maxrss is
            # in KiB on Linux.
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"parent RSS (peak) = {rss_mb:.1f} MB")

    # External archive dump
    # Write the non-dominated archive accumulated across all generations
    # to summary/archive.csv.  Distinct from the final-parent set: the
    # archive captures every individual that was ever Pareto-optimal in
    # raw objective space, even if subsequently displaced from the parent
    # set by HV-contribution selection drift.
    summary_dir = folders["summary"]
    if hasattr(strategy, "flush_archive_to_csv"):
        strategy.flush_archive_to_csv(summary_dir)

    # D2: Diversity metrics
    # Compute Deb's Δ (and ext_0, ext_1, spacing) on both the final
    # parent set and the external archive.  Dumped to a human-readable
    # txt file alongside convergence_data.txt.  Only meaningful for
    # 2-objective sim_types (CHT_AL); the writer no-ops for 3-obj runs.
    if is_al_active(sim_type):
        _write_diversity_metrics(
            out_path=summary_dir / "diversity_metrics.txt",
            final_parents=strategy.parents,
            archive=getattr(strategy, "external_archive", []),
        )

    # Post-processing figures (.eps) - AL family only
    # Render the three end-of-run figures from the archive and the
    # per-generation history collected in the loop above.  Each returns
    # None (and writes nothing) if its input is empty, which is fine for
    # a run that never populated the archive / bootstrapped the AL.
    if is_al_active(sim_type):
        postproc_dir = folders["postprocessing"]
        p_arc = plot_archive_pareto(
            getattr(strategy, "external_archive", []),
            postproc_dir / "archive_pareto.eps",
        )
        p_eps = plot_epsilon_and_gal(
            epsilon_history, gal_history,
            postproc_dir / "epsilon_gal_per_gen.eps",
        )
        p_div = plot_diversity(
            diversity_history,
            postproc_dir / "diversity_per_gen.eps",
        )
        written = [str(p) for p in (p_arc, p_eps, p_div) if p is not None]
        print(f"Post-processing figures: wrote {len(written)} .eps file(s) "
              f"to {postproc_dir}")

    # D3: Arnold diagnostic figures (once, at end of run)
    # The arnold_per_*.csv files have been appended every generation by the
    # drain block.  Here we read them back and emit the four standalone
    # figures exactly once - the violation heatmap, infeasibility rate,
    # mean ‖v_j‖ per active constraint, and κ(C) by lineage.  Each is its
    # own PNG in folders["arnold_diagnostics"].
    if cht_method(sim_type) == 'arnold':
        figs = _arnold_plot_all(folders["arnold_diagnostics"])
        print(f"Arnold diagnostics: wrote {len(figs)} figure(s) to "
              f"{folders['arnold_diagnostics']}")

    # Resample diagnostic figures (once, at end of run)
    # resample_per_gen.csv has been appended every generation by the drain
    # block.  Emit the per-constraint violation heatmap (all draws) and the
    # infeasible-draws-per-generation trace from it.
    if is_resample_active(sim_type):
        figs = _resample_plot_all(folders["resample_diagnostics"])
        print(f"Resample diagnostics: wrote {len(figs)} figure(s) to "
              f"{folders['resample_diagnostics']}")

    # Convergence data
    convergence_dir = folders["convergence"]

    with open(convergence_dir / "convergence_data.txt", "w") as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"Step Size = {step_size}\n")
        file.write(f'Pop Size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Hypervolume (nadir ref, higher = better) per generation:\n")
        for gen in range(NGEN):
            file.write(f"Generation {gen + 1}: {toolbox.logbook.bookshelf['hypervolume_nadir'][gen]}\n")

        file.write(f"\nHypervolume (utopia ref, lower = closer to ideal; legacy) per generation:\n")
        for gen in range(NGEN):
            file.write(f"Generation {gen + 1}: {toolbox.logbook.bookshelf['hypervolume'][gen]}\n")

        file.write(f"\nFeasible offspring per generation (out of {LAMBDA}):\n")
        for gen in range(NGEN):
            file.write(
                f"Generation {gen + 1}: "
                f"{toolbox.logbook.bookshelf['feasible_offspring_count'][gen]}\n"
            )

        # Diag-4: non-sentinel HV trace (only meaningful for CHT_AL).
        if is_al_active(sim_type):
            file.write("\nHypervolume (non-sentinel parents) per generation:\n")
            for gen in range(NGEN):
                file.write(
                    f"Generation {gen + 1}: "
                    f"{toolbox.logbook.bookshelf['hypervolume_nonsentinel'][gen]}\n"
                )

        if sim_type == 'CovarianceCHT':
            file.write("\nCHT resample iterations per generation:\n")
            for gen in range(NGEN):
                file.write(
                    f"Generation {gen + 1}: "
                    f"{toolbox.logbook.bookshelf['resample_iterations'][gen]}\n"
                )

    # Hypervolume convergence plot
    x_range = NGEN
    tick_interval = x_range / 5

    plt.figure(dpi=200)
    plt.title("Convergence (nadir-ref hypervolume, higher is better)")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (ref = nadir point)")
    plt.ylim((0, 1.1))

    gen_axis = list(range(1, NGEN + 1))
    avg_hv_list = [toolbox.logbook.bookshelf['hypervolume_nadir'][g - 1] for g in gen_axis]
    plt.plot(gen_axis, avg_hv_list)
    plt.savefig(convergence_dir / f"convergence_{sim_type}.png")
    plt.close()

    # Fixer count plot (non-Penalty runs only)
    fixer_count = []
    running_total = 0
    for entry in toolbox.logbook.bookshelf["fixer_count"].keys():
        if entry != '0':
            running_total += toolbox.logbook.bookshelf["fixer_count"][entry]
            fixer_count.append(running_total)

    e1 = time.time()
    toolbox.logbook.bookshelf["time taken"] = e1 - s1

    # Skip the "cumulative fixes" plot for any sim_type that doesn't run
    # the repair while-loop in generate().  Penalty was already excluded
    # by name; CovarianceCHT also bypasses repair (the CHT shrinkage
    # replaces it).  The empty-data check covers both cases and any
    # future no-repair sim_type without needing a name list.
    if fixer_count:
        generation = list(range(1, len(fixer_count) + 1))
        plt.figure(dpi=200)
        plt.title("Cumulative Number of Individuals Fixed")
        plt.xlabel("Generation")
        plt.ylabel("Number of Individuals Fixed")
        plt.plot(generation, fixer_count)
        plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
        plt.savefig(summary_dir / f"cma_es_mo_fpd_{sim_type}RunningTotal.png")
        plt.close()

    # Output summary text file
    # ideal_point / nadir_point come from _run_constants() at the top of
    # main(); they match the dimensionality of fitness_history's tuples
    # (3-D for legacy, 2-D for CHT_AL) so unnormalise_fitness works
    # without further branching.
    initial_dimensionalised_fitness = [
        unnormalise_fitness(ind, ideal_point, nadir_point)
        for ind in fitness_history[:MU]
    ]
    final_dimensionalised_fitness = [
        unnormalise_fitness(ind, ideal_point, nadir_point)
        for ind in fitness_history[-MU:]
    ]
    # Header used in the per-population objective tables below.  In
    # CHT_AL mode delta_vs1 has been moved out of fitness_history (it
    # is a constraint, not an objective) - so the header omits it.
    objectives_header = (
        "Driver Hold Time (ms) | Piston Impact Speed (m/s)"
        if is_al_active(sim_type)
        else "Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)"
    )
    # ms-conversion column index (hold_time): index 1 in 3-D, index 0 in 2-D.
    holdtime_col_idx = 0 if is_al_active(sim_type) else 1

    sig_figs = 6

    def _fmt_var(variable, index):
        """Format a single design variable for the output table."""
        col_widths = [
            'Percent Helium ', 'Driver Pressure (MPa) ', ' p4 (MPa) ',
            ' Throat Diameter (mm) ', ' Reservoir Pressure (MPa) ', ' Buffer Length (mm) '
        ]
        scale = [1, 1e-6, 1e-6, 1e3, 1e-6, 1e3]
        variable = round(variable * scale[index], sig_figs - str(variable * scale[index]).find('.'))
        white_space = int(np.round((len(col_widths[index]) - len(f'{variable}')) / 2))
        sep = '|' if index in [1, 3, 4] else ''
        return sep + ' ' * white_space + f'{variable}' + ' ' * white_space

    with open(summary_dir / "output.txt", "w") as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"step size = {experiment_type[2]}\n")
        file.write(f'pop size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        # Provenance stamp: the exact code + working-tree state and the full
        # anti-degeneration feature set that produced this run.  Two runs that
        # differ only by features are now
        # distinguishable straight from output.txt, no diagnostics archaeology.
        file.write(f"git commit = {_git_stamp()}\n")
        file.write(f"features = {_features_oneline(features)}\n")
        file.write(f"time taken = {toolbox.logbook.bookshelf['time taken']}\n")
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Number of individuals that produced no hold time = "
                   f"{toolbox.logbook.bookshelf['No. individuals that produced no hold time']}\n")
        file.write(f"Fixer Count = {running_total}\n")
        file.write('*' * 100 + '\n')
        file.write("INITIAL POPULATION:\n")
        file.write(
            "Percent Helium |Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer Length (mm)\n"
        )
        for ind in initial_population:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write(objectives_header + "\n")
        for ind in initial_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == holdtime_col_idx:
                    obj *= 1e3   # seconds -> milliseconds
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

        file.write('\n' + '*' * 100 + '\n')
        file.write("FINAL POPULATION:\n")
        file.write(
            "Percent Helium | Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer length (mm)\n"
        )
        for ind in strategy.parents:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write(objectives_header + "\n")
        for ind in final_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == holdtime_col_idx:
                    obj *= 1e3   # seconds -> milliseconds
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

    # Final save catches any in-flight snapshots (e.g. when NGEN is not a
    # multiple of SAVE_INTERVAL) and re-writes earlier CSVs with any newly
    # available chosen / offspring data.
    _save_outputs(toolbox.logbook.bookshelf['generation'],
                  gen_snapshots, fitness_history, MU, folders,
                  strategy=strategy, bounds=bounds)

    print('\n\nEND OF SIM')
    print('*' * 60)
    print('\n\n')

    return strategy.parents


# Entry point

if __name__ == "__main__":
    _config_path = pathlib.Path(__file__).parent.parent / "config" / "experiments.yaml"
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f)

    # 10-tuple: (sim_type, pop_size, step_size, p4_treatment, al_tol,
    #            cht_gamma, features_dict, arnold_beta, arnold_cc, n_gen).
    # ``al_tol``      : (AL family) constraint tolerance ε (m/s).
    # ``cht_gamma``   : (Chocat family) shrinkage strength; None -> default.
    # ``features``    : dict of optional anti-degeneration toggles.  See the
    #                   YAML header comment.  Empty dict = baseline.
    # ``arnold_beta`` / ``arnold_cc`` : (Arnold family) Eq. 7 / Eq. 6
    #                   coefficients; None -> paper defaults.
    # ``n_gen``       : number of generations; omitted -> historic default 350.
    experiment_types = [
        (
            exp["sim_type"],
            exp["pop_size"],
            exp["step_size"],
            exp["p4_treatment"],
            exp.get("al_tol", 100.0),
            exp.get("cht_gamma", None),
            exp.get("features", {}) or {},
            exp.get("arnold_beta", None),
            exp.get("arnold_cc", None),
            exp.get("n_gen", 350),
        )
        for exp in _config["experiments"]
    ]

    # Optional seed population: a .npz from init_population_l1d.py.  Its
    # ``x_norm`` array (normalised [1,2]^6 designs) replaces the random
    # initial population.  Threaded through the worker subprocess via the
    # same flag so dispatcher and worker stay in lockstep.
    def _load_seed_npz(path):
        with np.load(path) as data:
            if "x_norm" not in data:
                raise KeyError(
                    f"{path} has no 'x_norm' array - expected an "
                    f"init_population_l1d.py output (.npz)."
                )
            return np.asarray(data["x_norm"], dtype=float)

    _seed_npz = None
    if "--seed-npz" in sys.argv:
        _seed_npz = sys.argv[sys.argv.index("--seed-npz") + 1]

    if "--experiment-index" in sys.argv:
        # Worker mode
        # This branch runs when the dispatcher below launched us as a child
        # subprocess.  We execute exactly one experiment and then exit,
        # letting the OS reclaim every byte of RAM the run accumulated.
        idx = int(sys.argv[sys.argv.index("--experiment-index") + 1])
        _seed_pop = _load_seed_npz(_seed_npz) if _seed_npz else None
        solutions = main(experiment_types[idx], seed_population=_seed_pop)

    else:
        # Dispatcher mode
        # Run each experiment in a fresh Python interpreter so that memory
        # (PITOT3 Lua VMs, matplotlib caches, gdtk gas-model objects, the
        # multiprocessing worker pool) is fully reclaimed between experiments.
        # Without this, successive runs in the same process accumulate RSS
        # until the kernel OOM-kills the parent (~3x slowdown by run 4).
        #
        # sys.executable   - same interpreter that is running this script,
        #                    so virtual-environment / conda paths are preserved.
        # Path(__file__).resolve() - absolute path to main.py, works
        #                    regardless of the working directory the user
        #                    invoked us from.
        _script = pathlib.Path(__file__).resolve()

        for i, experiment_type in enumerate(experiment_types):
            print(f"\n{'=' * 60}")
            print(f"Experiment {i + 1} / {len(experiment_types)}: {experiment_type}")
            print(f"{'=' * 60}\n")
            _cmd = [sys.executable, str(_script), "--experiment-index", str(i)]
            if _seed_npz:
                # Forward the seed so every worker starts from the same
                # init_population_l1d.py population.
                _cmd += ["--seed-npz", _seed_npz]
            result = subprocess.run(
                _cmd,
                check=False,           # don't raise - report and continue
            )
            if result.returncode != 0:
                print(
                    f"\nWARNING: experiment {i + 1} exited with code "
                    f"{result.returncode}.  Continuing with the next one.\n"
                )
