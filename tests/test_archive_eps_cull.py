"""
Unit tests for the ε-refresh cull on the external non-dominated archive.

The archive (feature D1) admits on dominance alone.  Under a moving
``al_tol`` schedule (feature B3) that lets it accumulate members which
satisfied only the LOOSE early tolerance: in run al_cht_0111, 38 of the
100 archived members violated the final ε = 100 m/s, and they made up the
entire ``t_hold > 1.87 ms`` edge of the front.

``StrategyMultiObjective._update_archive`` therefore re-tests the whole
archive against the CURRENT ε every generation and gates new admissions
the same way.  These tests exercise that in isolation - no L1d, no heavy
evaluator.
"""
import os
import sys

import numpy as np


import main  # noqa: F401 - registers creator.Individual2D / FitnessMulti2D
from deap import creator
from algorithm.cmaes import StrategyMultiObjective
from problem.config import BOUNDS
from problem.feasibility import evaluate_constraints, is_feasible
from problem.transforms import variable_transformation

SCHEDULE = [3585.0, 100.0, 100, 250]


def _feasible_norm_pop(n, rng):
    """Return n feasible normalised [1,2]^6 individuals (as plain lists)."""
    from init_population_l1d import lhs_unit_to_physical
    out = []
    while len(out) < n:
        phys = lhs_unit_to_physical(rng.random(6))
        x_norm = variable_transformation([phys], BOUNDS)[0]
        if is_feasible(evaluate_constraints(x_norm, BOUNDS)):
            out.append([float(v) for v in x_norm])
    return out


def _make_ind(x, raw_delta, tol, fitness):
    """One archive-eligible individual carrying a raw residual."""
    ind = creator.Individual2D(x)
    ind.bounds = BOUNDS
    ind.sim_type = 'ArnoldCHT_AL'
    ind.al_tol = tol
    ind.fitness.values = fitness
    ind._feasible = True
    ind._pitot3_sentinel = False
    ind._spark_sentinel = False
    if raw_delta is None:
        ind._g_al = None
    else:
        ind._raw_delta_vs1 = raw_delta
        ind._g_al = np.array([raw_delta - tol])
    return ind


def _make_strategy(parents, cull=True):
    n_constraints = len(evaluate_constraints(parents[0], BOUNDS))
    return StrategyMultiObjective(
        parents, sigma=0.1, mu=len(parents), lambda_=len(parents),
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
        features={'al_tol_schedule': SCHEDULE},
        archive_eps_cull=cull,
    )


def _raws(strategy):
    return sorted(m["raw_delta_vs1"] for m in strategy.external_archive)


def test_loose_era_member_is_culled_when_epsilon_tightens():
    """A member admitted at a loose ε must leave once ε drops below it."""
    rng = np.random.default_rng(0)
    xs = _feasible_norm_pop(2, rng)
    parents = [_make_ind(xs[0], 2412.0, 3585.0, (0.4, 0.6)),
               _make_ind(xs[1], 60.0, 3585.0, (0.6, 0.4))]
    strat = _make_strategy(parents)

    # Generation 1: ε is still 3585, so both are admissible.
    strat._generation = 1
    strat._update_archive(parents)
    assert _raws(strat) == [60.0, 2412.0]
    assert strat.archive_eps_culled_total == 0

    # Generation 250: ε has tightened to 100.  The 2412 m/s member is no
    # longer compliant with the tolerance the run is enforcing.
    strat._generation = 250
    assert strat.current_al_tol() == 100.0
    strat._update_archive([])

    assert _raws(strat) == [60.0]
    assert strat.archive_eps_culled_gen == 1
    assert strat.archive_eps_culled_total == 1


def test_cull_runs_with_no_new_candidates():
    """The schedule tightens regardless of what this generation produced."""
    rng = np.random.default_rng(1)
    xs = _feasible_norm_pop(1, rng)
    parents = [_make_ind(xs[0], 500.0, 3585.0, (0.5, 0.5))]
    strat = _make_strategy(parents)
    strat._generation = 1
    strat._update_archive(parents)
    assert len(strat.external_archive) == 1

    # No candidates at all - the early-return must not skip the cull.
    strat._generation = 250
    strat._update_archive([])
    assert strat.external_archive == []
    assert strat.archive_eps_culled_total == 1


def test_new_admissions_are_gated_at_current_epsilon():
    """An individual violating the current ε never enters the archive."""
    rng = np.random.default_rng(2)
    xs = _feasible_norm_pop(2, rng)
    parents = [_make_ind(xs[0], 80.0, 100.0, (0.4, 0.6))]
    strat = _make_strategy(parents)
    strat._generation = 250

    violator = _make_ind(xs[1], 250.0, 100.0, (0.1, 0.9))  # non-dominated
    strat._update_archive(parents + [violator])

    assert _raws(strat) == [80.0]


def test_members_without_a_raw_measurement_are_kept():
    """Non-AL runs (g_al is None) must be unaffected - we cull only what
    we can measure."""
    rng = np.random.default_rng(3)
    xs = _feasible_norm_pop(2, rng)
    parents = [_make_ind(xs[0], None, 3585.0, (0.4, 0.6)),
               _make_ind(xs[1], None, 3585.0, (0.6, 0.4))]
    strat = _make_strategy(parents)
    strat._generation = 1
    strat._update_archive(parents)
    assert len(strat.external_archive) == 2

    strat._generation = 250
    strat._update_archive([])
    assert len(strat.external_archive) == 2
    assert strat.archive_eps_culled_total == 0


def test_switch_off_restores_legacy_accumulation():
    """archive_eps_cull=False reproduces the pre-fix behaviour."""
    rng = np.random.default_rng(4)
    xs = _feasible_norm_pop(2, rng)
    parents = [_make_ind(xs[0], 2412.0, 3585.0, (0.4, 0.6)),
               _make_ind(xs[1], 60.0, 3585.0, (0.6, 0.4))]
    strat = _make_strategy(parents, cull=False)
    strat._generation = 1
    strat._update_archive(parents)
    strat._generation = 250
    strat._update_archive([])

    assert _raws(strat) == [60.0, 2412.0]     # the violator survives
    assert strat.archive_eps_culled_total == 0


def test_raw_residual_falls_back_to_g_al_reconstruction():
    """Individuals with no cached _raw_delta_vs1 reconstruct it from
    g_al + birth al_tol (main.py's convention)."""
    rng = np.random.default_rng(5)
    xs = _feasible_norm_pop(1, rng)
    ind = _make_ind(xs[0], 2412.0, 3585.0, (0.5, 0.5))
    del ind._raw_delta_vs1                     # force the fallback path
    strat = _make_strategy([ind])
    strat._generation = 1
    strat._update_archive([ind])

    assert _raws(strat) == [2412.0]


def test_flushed_csv_carries_the_raw_residual(tmp_path):
    """archive.csv gains a raw_delta_vs1 column so post-hoc tools can
    audit compliance without reconstructing the schedule."""
    import csv
    rng = np.random.default_rng(6)
    xs = _feasible_norm_pop(1, rng)
    parents = [_make_ind(xs[0], 95.0, 100.0, (0.5, 0.5))]
    strat = _make_strategy(parents)
    strat._generation = 250
    strat._update_archive(parents)
    strat.flush_archive_to_csv(tmp_path)

    with (tmp_path / "archive.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert np.isclose(float(rows[0]["raw_delta_vs1"]), 95.0)
