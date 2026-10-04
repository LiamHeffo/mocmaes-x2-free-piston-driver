"""
Smoke test for the ported ArnoldCHT_AL strategy (no L1d / SPARK).

Exercises the full per-generation lifecycle that main.py drives -
generate() -> apply_arnold_infeasibility() -> (synthetic) evaluation ->
update() - with a cheap synthetic fitness, so the Arnold covariance
update (Eq. 6/7), the AL layer, and the infeasible-drop selection are
all validated quickly without the heavy evaluator.
"""
import os
import sys

import numpy as np


import main  # noqa: F401  - registers creator.Individual2D / FitnessMulti2D
from deap import creator
from algorithm.cmaes import (
    StrategyMultiObjective, is_al_active, cht_method, ARNOLD_SIM_TYPES,
)
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


def _make_parents(x_norms, sim_type):
    parents = []
    for i, x in enumerate(x_norms):
        ind = creator.Individual2D(x)
        ind.ind_number = i
        ind.bounds = BOUNDS
        ind.sim_type = sim_type
        ind.al_tol = 3585.0
        # synthetic 2-objective fitness (both minimised, in [0,1])
        ind.fitness.values = (0.5 + 0.01 * i, 0.5 - 0.005 * i)
        ind._feasible = True
        ind._g = evaluate_constraints(x, BOUNDS)
        ind._g_al = np.array([0.0])
        parents.append(ind)
    return parents


def test_arnold_constructs_with_v_accumulators():
    rng = np.random.default_rng(0)
    pop = _make_parents(_feasible_norm_pop(4, rng), 'ArnoldCHT_AL')
    n_constraints = len(evaluate_constraints(pop[0], BOUNDS))
    strat = StrategyMultiObjective(
        pop, sigma=0.1, mu=4, lambda_=4,
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
    )
    assert is_al_active('ArnoldCHT_AL')
    assert cht_method('ArnoldCHT_AL') == 'arnold'
    assert 'ArnoldCHT_AL' in ARNOLD_SIM_TYPES
    # one v_j accumulator per constraint per parent
    assert len(strat.v) == 4
    assert all(len(vj) == n_constraints for vj in strat.v)
    assert strat.al is not None  # AL layered on


def test_arnold_requires_n_constraints():
    rng = np.random.default_rng(1)
    pop = _make_parents(_feasible_norm_pop(4, rng), 'ArnoldCHT_AL')
    try:
        StrategyMultiObjective(
            pop, sigma=0.1, mu=4, lambda_=4,
            sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
            al_tol=3585.0,  # n_constraints omitted on purpose
        )
    except ValueError as e:
        assert 'n_constraints' in str(e)
    else:
        raise AssertionError("expected ValueError when n_constraints is missing")


def test_arnold_consume_updates_A_and_keeps_it_psd():
    rng = np.random.default_rng(2)
    pop = _make_parents(_feasible_norm_pop(4, rng), 'ArnoldCHT_AL')
    n_constraints = len(evaluate_constraints(pop[0], BOUNDS))
    strat = StrategyMultiObjective(
        pop, sigma=0.5, mu=4, lambda_=4,            # big σ -> some infeasibles
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
    )
    A_before = [A.copy() for A in strat.A]

    offspring = strat.generate(creator.Individual2D)
    # generate() must tag the step the Arnold update consumes
    assert all(hasattr(o, '_Az') for o in offspring)
    assert all(hasattr(o, '_ps') for o in offspring)

    def _check(ind):
        g = evaluate_constraints(ind, BOUNDS)
        return is_feasible(g), g

    strat.apply_arnold_infeasibility(offspring, feasibility_check=_check)

    # every offspring now carries _feasible / _g
    assert all(hasattr(o, '_feasible') for o in offspring)
    # every per-parent A is still a valid Cholesky factor of a PSD matrix
    for A in strat.A:
        C = A @ A.T
        np.linalg.cholesky(0.5 * (C + C.T))   # raises if not PSD
    # at least one A changed OR everything happened to be feasible
    changed = any(not np.allclose(a0, a1)
                  for a0, a1 in zip(A_before, strat.A))
    n_infeasible = sum(1 for o in offspring if not o._feasible)
    assert changed or n_infeasible == 0


def test_arnold_full_generation_runs():
    """generate -> consume -> synthetic eval -> update completes for ArnoldCHT_AL."""
    rng = np.random.default_rng(3)
    pop = _make_parents(_feasible_norm_pop(6, rng), 'ArnoldCHT_AL')
    n_constraints = len(evaluate_constraints(pop[0], BOUNDS))
    strat = StrategyMultiObjective(
        pop, sigma=0.15, mu=6, lambda_=6,
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
    )

    def _check(ind):
        g = evaluate_constraints(ind, BOUNDS)
        return is_feasible(g), g

    for _gen in range(3):
        offspring = strat.generate(creator.Individual2D)
        strat.apply_arnold_infeasibility(offspring, feasibility_check=_check)
        for k, o in enumerate(offspring):
            if o._feasible:
                # synthetic 2-objective fitness + AL constraint slot
                o.fitness.values = (0.4 + 0.02 * k + 0.01 * _gen,
                                    0.4 - 0.01 * k)
                o._g_al = np.array([rng.uniform(-50, 50)])
        strat.update(offspring)
        # parents remain a feasible, mu-sized set with valid fitness
        assert len(strat.parents) == 6
        assert all(p.fitness.valid for p in strat.parents)
