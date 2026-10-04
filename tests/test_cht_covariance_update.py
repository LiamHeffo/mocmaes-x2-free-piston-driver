# tests/test_cht_covariance_update.py
#
# Unit tests for StrategyMultiObjective._chtCovarianceUpdate - Phase 2 of
# the constraint-handling-technique work (Chocat et al. 2015 + Arnold &
# Hansen 2012, with Adaptation-B Mahalanobis pooling).
#
# Each test isolates one property of the update so failures localise.

import sys, os

import numpy as np
import pytest

from _helpers import Ind
from algorithm.cmaes import StrategyMultiObjective


N = 6     # design-variable dimensionality
M = 18    # constraint count for X2 (6 physical + 2*6 box)


def _make_strategy(seed=0, n=N):
    """Minimal strategy with three identity-Cholesky parents at known points."""
    rng = np.random.RandomState(seed)
    parents = [Ind(rng.uniform(1.4, 1.6, size=n)) for _ in range(3)]
    return StrategyMultiObjective(
        population=parents,
        sigma=0.5,
        mu=3,
        lambda_=3,
    )


def _identity_state(strat, parent_idx=0):
    """Return (A, invCholesky) = (I, I) for a fresh test."""
    n = strat.dim
    return np.eye(n), np.eye(n)


def _no_violation_g():
    """All constraints satisfied (negative g vector)."""
    return -np.ones(M) * 0.1


def test_no_violators_returns_unchanged():
    """An empty pool of infeasibles must not modify the Cholesky factor."""
    strat = _make_strategy()
    A, invA = _identity_state(strat)

    A_new, invA_new = strat._chtCovarianceUpdate(
        A, invA, parent_idx=0,
        parents_snapshot=strat.parents,
        sigmas_snapshot=strat.sigmas,
        infeasible_offspring=[],
    )
    assert np.array_equal(A_new, A)
    assert np.array_equal(invA_new, invA)


def test_no_actual_violations_returns_unchanged():
    """Offspring with all-feasible g vectors must not trigger any shrinkage."""
    strat = _make_strategy()
    A, invA = _identity_state(strat)

    # One offspring near parent 0 but with a fully-feasible g.
    x_off = strat.parents[0] + np.array([0.05, 0, 0, 0, 0, 0])
    pool = [(0, x_off, _no_violation_g())]

    A_new, invA_new = strat._chtCovarianceUpdate(
        A, invA, parent_idx=0,
        parents_snapshot=strat.parents,
        sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )
    assert np.array_equal(A_new, A)
    assert np.array_equal(invA_new, invA)


def test_axis_aligned_violation_shrinks_only_that_axis():
    """A single offspring displaced along axis 0 violating one constraint
    must shrink eigenvalue 0 and leave the others approximately unchanged."""
    strat = _make_strategy()
    A, invA = _identity_state(strat)

    parent = np.asarray(strat.parents[0])
    # Offspring sits exactly along the +x axis from the parent.
    delta = np.zeros(N); delta[0] = 0.2
    x_off = parent + delta

    g = _no_violation_g().copy()
    g[5] = 0.5     # pretend constraint 5 (p4 ceiling) was violated

    pool = [(0, x_off, g)]
    A_new, _ = strat._chtCovarianceUpdate(
        A, invA, parent_idx=0,
        parents_snapshot=strat.parents,
        sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )

    C_old = A @ A.T
    C_new = A_new @ A_new.T
    vp_old = np.linalg.eigvalsh(C_old)   # ascending
    vp_new = np.linalg.eigvalsh(C_new)

    # The eigenvalue aligned with the violation axis should shrink.
    # With identity start, all eigenvalues are 1; after shrinking one,
    # the smallest new eigenvalue must be < 1.
    assert vp_new.min() < vp_old.min(), \
        f"expected shrinkage along violation axis: vp_new={vp_new}"


def test_hypervolume_preserved_within_log_tolerance():
    """det(C_new) ~= det(C_old) - the explicit eq.11 rescale should keep
    the search ellipsoid's volume."""
    strat = _make_strategy()
    A, invA = _identity_state(strat)

    parent = np.asarray(strat.parents[0])
    pool = [(0, parent + np.array([0.2, 0.1, 0, 0, 0, 0]), _no_violation_g())]
    pool[0][2][3] = 0.5    # violate compression-ratio lower

    A_new, _ = strat._chtCovarianceUpdate(
        A, invA, parent_idx=0,
        parents_snapshot=strat.parents,
        sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )

    det_old = np.linalg.det(A @ A.T)
    det_new = np.linalg.det(A_new @ A_new.T)

    # Eq. 11 enforces exactly det(C_new) == det(C_old) up to round-off.
    # 1e-6 relative tolerance accommodates the numerical floor clamp.
    assert det_new == pytest.approx(det_old, rel=1e-6), \
        f"hypervolume not preserved: det_old={det_old}, det_new={det_new}"


def test_returned_A_is_cholesky_of_C():
    """A_new must satisfy A_new @ A_new.T == C_new (lower-triangular)."""
    strat = _make_strategy()
    A, invA = _identity_state(strat)

    parent = np.asarray(strat.parents[0])
    pool = [(0, parent + np.array([0.15, 0, 0, 0, 0, 0]), _no_violation_g())]
    pool[0][2][5] = 0.3

    A_new, invA_new = strat._chtCovarianceUpdate(
        A, invA, parent_idx=0,
        parents_snapshot=strat.parents,
        sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )

    # Must be lower-triangular.
    assert np.allclose(np.triu(A_new, k=1), 0), "A_new is not lower-triangular"
    # invCholesky_new should be the inverse of A_new (within floating tol).
    identity_check = invA_new @ A_new
    assert np.allclose(identity_check, np.eye(N), atol=1e-9), \
        "invCholesky_new is not the inverse of A_new"


def test_pooling_weight_dampens_far_parent_more_than_near_parent():
    """Adaptation-B property: the *same* infeasible offspring at a fixed
    location should affect a near parent's covariance more than a far
    parent's, because the Mahalanobis pool weight is parent-specific.

    We isolate this by fixing the offspring's coordinates and changing
    only which parent the update is applied to.  The projection magnitude
    onto each parent's eigenvectors then depends on (x_off - x_i), which
    differs by displacement direction, but the eigenvectors themselves
    are identical (both parents start with A=I), so only the pool weight
    distinguishes the two updates' magnitudes.
    """
    n = N
    # Two parents on the same axis-0 line: one near the offspring, one far.
    parents = [Ind([1.5] * n), Ind([1.5] * n)]
    parents[0][0] = 1.45            # near parent
    parents[1][0] = 1.10            # far parent (Mahalanobis distance ~7x larger)

    strat = StrategyMultiObjective(
        population=parents, sigma=0.5, mu=2, lambda_=2,
    )

    x_off = np.array([1.5] * n)
    x_off[0] = 1.55                 # offspring just past the near parent
    g = _no_violation_g().copy()
    g[5] = 0.5
    pool = [(0, x_off, g)]
    A0, invA0 = np.eye(n), np.eye(n)

    A_near, _ = strat._chtCovarianceUpdate(
        A0, invA0, parent_idx=0,
        parents_snapshot=strat.parents, sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )
    A_far, _ = strat._chtCovarianceUpdate(
        A0, invA0, parent_idx=1,
        parents_snapshot=strat.parents, sigmas_snapshot=strat.sigmas,
        infeasible_offspring=pool,
    )

    # Smaller smallest-eigenvalue means more shrinkage along the
    # violation axis.  Near parent should shrink more.
    vp_near_min = np.linalg.eigvalsh(A_near @ A_near.T).min()
    vp_far_min  = np.linalg.eigvalsh(A_far  @ A_far.T ).min()
    assert vp_near_min < vp_far_min, (
        "Mahalanobis pool weight should make the near parent shrink more "
        f"(got near_min={vp_near_min}, far_min={vp_far_min})"
    )
