"""
Unit tests for Task 2: per-generation refresh of the AL constraint g_al
against the moving al_tol schedule.

The raw ``delta_vs1`` measurement is fixed at evaluation, but the AL
constraint ``g_al = delta_vs1 - al_tol(gen)`` must track the schedule for
surviving (elitist) parents - otherwise their g_al stays frozen at their
birth-generation al_tol (the stale-ε / frozen-parent-residual artifact).
These tests exercise StrategyMultiObjective.refresh_al_constraints in
isolation, with no L1d / heavy evaluator.
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


def _make_parents(x_norms, raw_deltas, birth_tol):
    parents = []
    for i, x in enumerate(x_norms):
        ind = creator.Individual2D(x)
        ind.ind_number = i
        ind.bounds = BOUNDS
        ind.sim_type = 'ArnoldCHT_AL'
        ind.al_tol = birth_tol
        ind.fitness.values = (0.5 + 0.01 * i, 0.5 - 0.005 * i)
        ind._feasible = True
        ind._g = evaluate_constraints(x, BOUNDS)
        ind._raw_delta_vs1 = raw_deltas[i]
        ind._g_al = np.array([raw_deltas[i] - birth_tol])
        parents.append(ind)
    return parents


def _make_strategy(parents, schedule):
    n_constraints = len(evaluate_constraints(parents[0], BOUNDS))
    return StrategyMultiObjective(
        parents, sigma=0.1, mu=len(parents), lambda_=len(parents),
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
        features={'al_tol_schedule': schedule},
    )


def test_refresh_unstales_parent_g_al():
    """A parent born at al_tol=3585 must be re-evaluated against the
    current (tighter) al_tol once the schedule has progressed."""
    rng = np.random.default_rng(0)
    raw = [2412.0, 3000.0, 1800.0, 2600.0]
    parents = _make_parents(_feasible_norm_pop(4, rng), raw, birth_tol=3585.0)
    strat = _make_strategy(parents, schedule=[3585.0, 100.0, 100, 250])

    # Jump to gen 250 where al_tol has tightened to its floor of 100.
    strat._generation = 250
    assert strat.current_al_tol() == 100.0

    strat.refresh_al_constraints(offspring=[])

    for p, r in zip(strat.parents, raw):
        assert np.isclose(p._g_al[0], r - 100.0)
        assert np.isclose(p.al_tol, 100.0)
        # At the tight tol every design with raw >> 100 is now AL-infeasible
        # - exactly the truth the refresh is meant to surface.
        assert p._g_al[0] > 0


def test_refresh_intermediate_generation_interpolates():
    rng = np.random.default_rng(1)
    raw = [2412.0, 3000.0]
    parents = _make_parents(_feasible_norm_pop(2, rng), raw, birth_tol=3585.0)
    strat = _make_strategy(parents, schedule=[3585.0, 100.0, 100, 250])

    strat._generation = 175           # midpoint of [100, 250]
    cur = strat.current_al_tol()
    assert np.isclose(cur, 1842.5)    # 3585 + (100-3585)*0.5

    strat.refresh_al_constraints(offspring=[])
    for p, r in zip(strat.parents, raw):
        assert np.isclose(p._g_al[0], r - 1842.5)


def test_refresh_offspring_noop_at_birth_tol():
    """An offspring evaluated this gen at current al_tol is unchanged."""
    rng = np.random.default_rng(2)
    parents = _make_parents(_feasible_norm_pop(2, rng), [2412.0, 3000.0], 3585.0)
    strat = _make_strategy(parents, schedule=[3585.0, 100.0, 100, 250])
    strat._generation = 175
    cur = strat.current_al_tol()

    off = creator.Individual2D(parents[0][:])
    off.al_tol = cur
    off._raw_delta_vs1 = 2200.0
    off._g_al = np.array([2200.0 - cur])
    before = float(off._g_al[0])

    strat.refresh_al_constraints(offspring=[off])
    assert np.isclose(off._g_al[0], before)            # no-op
    assert np.isclose(off._g_al[0], 2200.0 - cur)


def test_refresh_skips_individuals_without_measurement():
    """Box/phys-infeasible individuals (never L1d-evaluated) carry no raw
    measurement and must be left untouched."""
    rng = np.random.default_rng(3)
    parents = _make_parents(_feasible_norm_pop(2, rng), [2412.0, 3000.0], 3585.0)
    strat = _make_strategy(parents, schedule=[3585.0, 100.0, 100, 250])
    strat._generation = 250

    ghost = creator.Individual2D(parents[0][:])
    ghost._raw_delta_vs1 = None
    ghost._g_al = None

    strat.refresh_al_constraints(offspring=[ghost])
    assert ghost._g_al is None


def test_refresh_noop_without_schedule_reproduces_birth_value():
    """With no schedule, current_al_tol == static al_tol, so the recompute
    reproduces the birth g_al exactly (a safe no-op)."""
    rng = np.random.default_rng(4)
    raw = [2412.0, 3000.0]
    parents = _make_parents(_feasible_norm_pop(2, rng), raw, birth_tol=3585.0)
    strat = _make_strategy(parents, schedule=None)     # static al_tol = 3585
    strat._generation = 300
    strat.refresh_al_constraints(offspring=[])
    for p, r in zip(strat.parents, raw):
        assert np.isclose(p._g_al[0], r - 3585.0)
