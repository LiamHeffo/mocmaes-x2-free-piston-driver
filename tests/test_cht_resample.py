# tests/test_cht_resample.py
#
# Unit tests for StrategyMultiObjective.resample_infeasibles -
# the CHT-and-resample loop from Chocat 2015 Algorithm 3 step 3-2.

import sys, os

import numpy as np
from deap import base, creator

from algorithm.cmaes import StrategyMultiObjective


# Reuse the same DEAP types declared by test_cht_integration if it ran first,
# otherwise create them.  Distinct names keep this isolated from main.py.
if not hasattr(creator, "_TestFM"):
    creator.create("_TestFM", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("_TestInd", list, fitness=creator._TestFM,
                   ind_number=int, sim_type=str, bounds=list)


def _make_ind(coords, ps=("o", 0)):
    ind = creator._TestInd(list(coords))
    ind._ps = ps
    return ind


def _strategy_with_two_parents(n=4, sigma=1.0):
    parents = [_make_ind([1.5] * n, ps=("p", 0)),
               _make_ind([1.5] * n, ps=("p", 1))]
    return StrategyMultiObjective(
        population=parents, sigma=sigma, mu=2, lambda_=2,
    )


def test_no_infeasibles_returns_zero_iterations():
    """If every offspring is already feasible, the loop must exit on the
    first feasibility check and report 0 CHT-and-resample iterations."""
    strat = _strategy_with_two_parents()
    population = [_make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 0)),
                  _make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 1))]
    A0 = [a.copy() for a in strat.A]

    n_iter = strat.resample_infeasibles(
        population,
        feasibility_check=lambda ind: (True, np.full(2, -0.1)),
    )
    assert n_iter == 0
    # No CHT updates fired, so A is unchanged.
    for a, a0 in zip(strat.A, A0):
        assert np.array_equal(a, a0)
    # _feasible / _g still get populated on the first pass.
    assert all(ind._feasible for ind in population)
    assert all(ind._g is not None for ind in population)


def test_persistent_infeasibility_hits_iteration_cap():
    """If the feasibility check always returns False, the loop must run
    exactly max_iterations times, mutate self.A every iteration, and
    leave every individual marked infeasible."""
    strat = _strategy_with_two_parents()
    population = [_make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 0)),
                  _make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 1))]
    A0 = [a.copy() for a in strat.A]

    n_iter = strat.resample_infeasibles(
        population,
        feasibility_check=lambda ind: (False, np.array([1.0, 1.0])),
        max_iterations=3,
    )
    assert n_iter == 3
    # CHT modified every parent's A at least once.
    for a, a0 in zip(strat.A, A0):
        assert not np.array_equal(a, a0)
    assert all(not ind._feasible for ind in population)


def test_loop_converges_when_feasibility_becomes_true():
    """A check that flips to True after one iteration must produce
    1 iteration of CHT-and-resample, then exit."""
    strat = _strategy_with_two_parents()
    population = [_make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 0)),
                  _make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 1))]

    state = {"calls": 0}
    def flipping_check(ind):
        state["calls"] += 1
        # First two calls (initial pass) say infeasible.  After that, feasible.
        return (state["calls"] > 2, np.array([0.5, -0.1]))

    n_iter = strat.resample_infeasibles(
        population, feasibility_check=flipping_check, max_iterations=5,
    )
    # Initial pass tagged both infeasible (2 calls).  Iteration 1: CHT, then
    # resample 2 slots, each re-checked (2 more calls) -> both now feasible.
    # Loop iteration count returned is 1.
    assert n_iter == 1
    assert all(ind._feasible for ind in population)


def test_resample_replaces_coordinates_in_place():
    """Resampled offspring must keep their Individual identity (so DEAP
    fitness slots, ind_number, _ps survive) and have their coordinates
    overwritten by a fresh sample from the updated distribution."""
    strat = _strategy_with_two_parents(sigma=0.5)
    ind0 = _make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 0))
    ind1 = _make_ind([1.5, 1.5, 1.5, 1.5], ps=("o", 1))
    population = [ind0, ind1]
    original_id_0, original_id_1 = id(ind0), id(ind1)
    original_ps_0 = ind0._ps

    strat.resample_infeasibles(
        population,
        feasibility_check=lambda ind: (False, np.array([0.5])),
        max_iterations=2,
    )

    # Same Python object, _ps preserved.
    assert id(population[0]) == original_id_0
    assert id(population[1]) == original_id_1
    assert population[0]._ps == original_ps_0
    # Coordinates differ from the starting point (resample produced a new x).
    assert not np.allclose(np.asarray(population[0]), np.array([1.5] * 4))
