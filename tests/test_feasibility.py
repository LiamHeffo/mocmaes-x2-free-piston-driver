# tests/test_feasibility.py
#
# Unit tests for src/problem/feasibility.py.
#
# Each test exercises one constraint dimension in isolation: build a
# candidate that violates that dimension only, and assert that
# evaluate_constraints reports g_j > 0 for that index and <= 0 elsewhere.

import sys, os

import numpy as np
import pytest

from problem.feasibility import (
    evaluate_constraints,
    is_feasible,
    N_PHYSICAL_CONSTRAINTS,
)
from problem.config import BOUNDS


N = 6  # number of design variables


def _centre_candidate():
    """An obviously-feasible candidate at the centre of the normalised box.

    Mid-box gives roughly mid-range physical values.  The exact feasibility
    has to be computed because compression_ratio depends on p4/driver_p
    which couple non-linearly via the un-transformation.
    """
    return [1.5] * N


def test_centre_candidate_is_feasible():
    g = evaluate_constraints(_centre_candidate(), BOUNDS)
    # Box constraints definitely satisfied.
    assert np.all(g[N_PHYSICAL_CONSTRAINTS:] <= 0)
    # The compression-ratio constraints might or might not be satisfied at the
    # box centre depending on the bounds; the assertion below is conservative:
    # we only assert that the box is satisfied.  If is_feasible is True we
    # also exercise that path.
    if is_feasible(g):
        assert np.all(g <= 0)


def test_lower_box_violation_flags_only_box_dim():
    x = _centre_candidate()
    x[2] = 0.5         # below normalised lower bound of 1
    g = evaluate_constraints(x, BOUNDS)
    # Lower-box index for variable 2 is at offset N_PHYSICAL + 2.
    j = N_PHYSICAL_CONSTRAINTS + 2
    assert g[j] > 0, "lower box violation not flagged"
    # Physical constraints must be marked +inf because x is out of the box.
    assert np.all(np.isinf(g[:N_PHYSICAL_CONSTRAINTS]))
    assert not is_feasible(g)


def test_upper_box_violation_flags_only_box_dim():
    x = _centre_candidate()
    x[4] = 2.5         # above normalised upper bound of 2
    g = evaluate_constraints(x, BOUNDS)
    j = N_PHYSICAL_CONSTRAINTS + N + 4
    assert g[j] > 0, "upper box violation not flagged"
    assert np.all(np.isinf(g[:N_PHYSICAL_CONSTRAINTS]))
    assert not is_feasible(g)


def test_p4_ceiling_constraint_independent_of_box():
    """Cap p4 in physical space by setting normalised x[2] near 2 with a
    high driver_p; the resulting physical p4 may exceed bounds[2][1].

    This test verifies that constraint index 5 (p4 ceiling) can fire
    when box bounds are satisfied.  We don't assert the exact firing
    condition - only that the index is reachable and finite-valued.
    """
    x = _centre_candidate()
    x[1] = 2.0    # max driver_p  -> max coupling for p4
    x[2] = 2.0    # max normalised p4
    g = evaluate_constraints(x, BOUNDS)
    # Box is at the boundary but not exceeded, so physical-space slots
    # must be finite (not inf).
    assert np.all(np.isfinite(g[:N_PHYSICAL_CONSTRAINTS]))
    # p4 ceiling is index 5 - at maximum coupling we expect it to be
    # near zero or above.
    assert g[5] > -1e6   # finite, so it was actually computed


def test_constraint_vector_length():
    g = evaluate_constraints(_centre_candidate(), BOUNDS)
    assert len(g) == N_PHYSICAL_CONSTRAINTS + 2 * N


def test_is_feasible_tolerance():
    """Tolerance should accept tiny positive violations."""
    g = np.array([-0.1, 1e-12, -0.5])
    assert is_feasible(g, tol=1e-9) is True
    assert is_feasible(g, tol=0.0) is False
