# tests/test_cht_integration.py
#
# Integration tests for the full CovarianceCHT path through
# StrategyMultiObjective.update(): _select must skip infeasibles before
# the Pareto sort, and the per-parent loop must call _chtCovarianceUpdate
# when sim_type == 'CovarianceCHT' and the offspring pool contains
# infeasibles.

import sys, os

import numpy as np
from deap import base, creator

from algorithm.cmaes import StrategyMultiObjective


# Register DEAP types once for these tests.  Use distinct class names so
# this file can run alongside main.py without colliding.
if not hasattr(creator, "_TestFM"):
    creator.create("_TestFM", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("_TestInd", list, fitness=creator._TestFM,
                   ind_number=int, sim_type=str, bounds=list)


def _make_ind(coords, fitness=None, g=None, feasible=None, ps=None):
    ind = creator._TestInd(list(coords))
    if fitness is not None:
        ind.fitness.values = fitness
    if g is not None:
        ind._g = np.asarray(g, dtype=float)
    if feasible is not None:
        ind._feasible = feasible
    if ps is not None:
        ind._ps = ps
    return ind


def test_select_filters_infeasibles_into_not_chosen():
    """_select must put infeasibles in not_chosen and never feed them to
    DEAP's sortLogNondominated (which crashes on invalid fitness)."""
    parents = [
        _make_ind([1.5, 1.5], fitness=(0.3, 0.3, 0.3),
                  g=np.array([-0.1]), feasible=True, ps=("p", 0)),
        _make_ind([1.6, 1.4], fitness=(0.4, 0.2, 0.4),
                  g=np.array([-0.1]), feasible=True, ps=("p", 1)),
    ]
    strat = StrategyMultiObjective(population=parents, sigma=0.5, mu=2, lambda_=2)

    # Three offspring: two feasible, one infeasible (no fitness, _feasible=False)
    offspring = [
        _make_ind([1.55, 1.45], fitness=(0.35, 0.25, 0.30),
                  g=np.array([-0.05]), feasible=True, ps=("o", 0)),
        _make_ind([1.65, 1.35], fitness=(0.45, 0.15, 0.40),
                  g=np.array([-0.05]), feasible=True, ps=("o", 1)),
        _make_ind([0.5, 1.50], g=np.array([0.5]),
                  feasible=False, ps=("o", 0)),    # box-violator
    ]

    chosen, not_chosen = strat._select(offspring + parents)
    assert len(chosen) == 2
    assert all(ind.fitness.valid for ind in chosen), \
        "_select returned an infeasible in chosen"
    assert any(not getattr(ind, "_feasible", True) for ind in not_chosen), \
        "infeasible not surfaced via not_chosen"


def test_update_runs_with_mixed_feasibility_and_modifies_A():
    """update() must run end-to-end with infeasibles in the population and,
    when sim_type == 'CovarianceCHT', actually change A[i] via the CHT."""
    n = 6
    parents = [
        _make_ind([1.5] * n, fitness=(0.3, 0.3, 0.3),
                  g=np.full(18, -0.1), feasible=True, ps=("p", 0)),
        _make_ind([1.5] * n, fitness=(0.4, 0.2, 0.4),
                  g=np.full(18, -0.1), feasible=True, ps=("p", 1)),
    ]
    strat = StrategyMultiObjective(
        population=parents, sigma=0.5, mu=2, lambda_=2,
        sim_type='CovarianceCHT',
    )
    A0 = strat.A[0].copy()

    # Two offspring: one feasible (will be selected), one infeasible
    # (drives the CHT shrinkage along its violation axis).
    feas = _make_ind([1.55] + [1.5] * (n - 1),
                     fitness=(0.25, 0.25, 0.25),
                     g=np.full(18, -0.1), feasible=True, ps=("o", 0))
    g_inf = np.full(18, -0.1); g_inf[5] = 0.3      # violates physical constraint 5
    inf  = _make_ind([1.5] * n,
                     g=g_inf, feasible=False, ps=("o", 1))
    inf[0] = 1.7      # axis-0 displacement so projection is on eigenvector 0

    population = [feas, inf]

    # Should not raise.
    strat.update(population)

    # The chosen offspring's parent (parent 0) had its A modified by the
    # CHT (covariance shrinkage along axis 0 because of the infeasible
    # sibling), then by the rank-mu_succ + rank-one updates.
    assert not np.allclose(strat.A[0], A0), \
        "parent 0 A unchanged after CovarianceCHT update"


def test_update_with_no_infeasibles_still_works():
    """All-feasible population: CHT call is a no-op (empty pool); update
    must behave exactly like the non-CHT path."""
    n = 6
    parents = [
        _make_ind([1.5] * n, fitness=(0.3, 0.3, 0.3),
                  g=np.full(18, -0.1), feasible=True, ps=("p", 0)),
        _make_ind([1.5] * n, fitness=(0.4, 0.2, 0.4),
                  g=np.full(18, -0.1), feasible=True, ps=("p", 1)),
    ]
    strat = StrategyMultiObjective(
        population=parents, sigma=0.5, mu=2, lambda_=2,
        sim_type='CovarianceCHT',
    )

    offspring = [
        _make_ind([1.55] + [1.5]*(n-1), fitness=(0.25, 0.25, 0.25),
                  g=np.full(18, -0.1), feasible=True, ps=("o", 0)),
        _make_ind([1.45] + [1.5]*(n-1), fitness=(0.35, 0.35, 0.35),
                  g=np.full(18, -0.1), feasible=True, ps=("o", 1)),
    ]
    # Should not raise and should not crash because of an empty pool.
    strat.update(offspring)
    assert len(strat.parents) == 2
