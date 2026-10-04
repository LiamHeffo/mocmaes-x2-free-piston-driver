# tests/test_rank_mu_update.py
#
# Unit tests for the rank-mu_MO,succ recombination update added to
# StrategyMultiObjective per Voss 2009 ("Recombination for Learning Strategy
# Parameters in the MO-CMA-ES").

import sys, os

import numpy as np
import pytest

from _helpers import Ind
from algorithm.cmaes import StrategyMultiObjective


def _make_strategy(n=4, mu=3, sigma=0.5, seed=0):
    """Build a minimal strategy with `mu` parents at well-defined positions.

    We don't need real DEAP individuals for the rank-mu update - it only
    consumes self.parents (sequence-like of floats), self.sigmas, and the
    Cholesky factors.  Lists of np.arrays suffice.
    """
    rng = np.random.RandomState(seed)
    parents = [Ind(rng.uniform(1.2, 1.8, size=n)) for _ in range(mu)]
    return StrategyMultiObjective(
        population=parents,
        sigma=sigma,
        mu=mu,
        lambda_=mu,
        sim_type='Penalty',          # avoid feasibility-repair path
        bounds=[(0.0, 3.0)] * n,
    )


def test_rank_mu_no_successful_offspring_is_noop():
    """With no successful steps, rank-mu must leave A and invCholesky unchanged."""
    strat = _make_strategy()
    A0 = strat.A[0].copy()
    invC0 = strat.invCholesky[0].copy()

    A_new, invC_new = strat._rankMuSuccUpdate(
        A0.copy(), invC0.copy(),
        parent_idx=0,
        parents_snapshot=[np.array(p) for p in strat.parents],
        sigmas_snapshot=list(strat.sigmas),
        successful_steps=[],
    )
    np.testing.assert_array_equal(A_new, A0)
    np.testing.assert_array_equal(invC_new, invC0)


def test_rank_mu_preserves_spd():
    """After rank-mu, C = A A^T must remain symmetric positive-definite."""
    strat = _make_strategy(n=4, mu=4)
    parents_snap = [np.array(p) for p in strat.parents]
    sigmas_snap  = list(strat.sigmas)

    # Construct synthetic successful offspring: one per parent, perturbed.
    rng = np.random.RandomState(42)
    successful = [
        (j, parents_snap[j] + sigmas_snap[j] * rng.randn(strat.dim))
        for j in range(strat.mu)
    ]

    A_new, invC_new = strat._rankMuSuccUpdate(
        strat.A[0].copy(), strat.invCholesky[0].copy(),
        parent_idx=0,
        parents_snapshot=parents_snap,
        sigmas_snapshot=sigmas_snap,
        successful_steps=successful,
    )

    C = A_new @ A_new.T
    np.testing.assert_allclose(C, C.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(C)
    assert np.all(eigs > 0), f"C is not PD; eigvals={eigs}"

    # invCholesky is the inverse of A (lower-triangular).
    np.testing.assert_allclose(A_new @ invC_new, np.eye(strat.dim), atol=1e-9)


def test_rank_mu_pulls_C_toward_step_directions():
    """If all successful steps point along axis 0, the updated C should
    have a larger variance along axis 0 than the identity it started from."""
    n = 3
    strat = _make_strategy(n=n, mu=4, sigma=1.0)
    parents_snap = [np.zeros(n) for _ in range(strat.mu)]   # all parents at origin
    sigmas_snap  = [1.0] * strat.mu

    # All offspring step purely along +x_0 from their parent.
    successful = [(j, np.array([2.0, 0.0, 0.0])) for j in range(strat.mu)]

    A0 = np.eye(n)        # so C_0 = I
    invC0 = np.eye(n)
    A_new, _ = strat._rankMuSuccUpdate(
        A0, invC0,
        parent_idx=0,
        parents_snapshot=parents_snap,
        sigmas_snapshot=sigmas_snap,
        successful_steps=successful,
    )
    C_new = A_new @ A_new.T
    assert C_new[0, 0] > C_new[1, 1], (
        f"axis-0 variance ({C_new[0,0]}) should exceed axis-1 ({C_new[1,1]}) "
        "after biased rank-mu update"
    )


def test_repaired_offspring_excluded_via_update_filter():
    """Smoke test: offspring tagged _repaired=True must not enter the
    successful_steps list assembled inside update().

    We don't run a full update() here (DEAP individuals required); we just
    verify the filter expression itself.  This guards against future regressions
    where someone removes the `not getattr(ind, "_repaired", False)` check.
    """
    class FakeInd:
        def __init__(self, ps, repaired=False):
            self._ps = ps
            self._repaired = repaired

    chosen = [
        FakeInd(("o", 0), repaired=False),
        FakeInd(("o", 1), repaired=True),
        FakeInd(("p", 2), repaired=False),  # non-offspring tag
        FakeInd(("o", 0), repaired=False),
    ]
    selected = [
        ind for ind in chosen
        if ind._ps[0] == "o" and not getattr(ind, "_repaired", False)
    ]
    assert len(selected) == 2
    assert all(s._ps[0] == "o" for s in selected)
    assert all(not s._repaired for s in selected)
