# tests/test_variable_transforms.py

import sys, os

import pytest
import numpy as np
from problem.transforms import variable_transformation, variable_untransformation
from problem.config import BOUNDS, driver_p_lower, driver_p_upper


# Helpers

def transform_single(physical_point):
    """Convenience: transform one physical point, return the normalised list."""
    return variable_transformation([physical_point], BOUNDS)[0]


# Round-trip tests
#
# The most important property of this pair of functions is that they are exact
# inverses of each other.  A bug in either formula (especially for the coupled
# variables p4 and reservoir_p) would break this property.
#
# We test multiple physically distinct points so that a compensation error
# (wrong formula that happens to round-trip one specific value) is not missed.

def test_roundtrip_mid_range_point():
    """A typical operating point round-trips exactly.

    driver_p = 300 000 Pa.  p4 must be in [14.62 * 300000, 1190.63 * 300000]
    = [4 386 000, 357 189 000].  reservoir_p must be in [300 000, 8 000 000].
    """
    physical = [85, 300_000, 10_000_000, 0.07, 5_000_000, 0.10]
    normalised = transform_single(physical)
    recovered = variable_untransformation(normalised, BOUNDS)
    assert recovered == pytest.approx(physical, rel=1e-9)


def test_roundtrip_different_driver_pressure():
    """A second point with a higher driver_p verifies the coupled bounds scale correctly.

    driver_p = 1 000 000 Pa.  p4 in [14 620 000, 1 190 630 000].
    reservoir_p in [1 000 000, 8 000 000].
    """
    physical = [92, 1_000_000, 50_000_000, 0.06, 6_000_000, 0.12]
    normalised = transform_single(physical)
    recovered = variable_untransformation(normalised, BOUNDS)
    assert recovered == pytest.approx(physical, rel=1e-9)


def test_roundtrip_near_lower_bounds():
    """A point close to the lower physical bounds round-trips correctly."""
    driver_p = 5_000
    physical = [71, driver_p, 14.62 * driver_p * 1.01, 0.051, driver_p * 1.1, 0.06]
    normalised = transform_single(physical)
    recovered = variable_untransformation(normalised, BOUNDS)
    assert recovered == pytest.approx(physical, rel=1e-9)


def test_roundtrip_near_upper_bounds():
    """A point close to the upper physical bounds round-trips correctly."""
    driver_p = 2_000_000
    physical = [99, driver_p, 1190.63 * driver_p * 0.99, 0.084, 7_900_000, 0.14]
    normalised = transform_single(physical)
    recovered = variable_untransformation(normalised, BOUNDS)
    assert recovered == pytest.approx(physical, rel=1e-9)


# Boundary tests
#
# The normalisation formula is designed so that:
#   physical lower bound  ->  normalised 1.0
#   physical upper bound  ->  normalised 2.0
#
# For p4 and reservoir_p the bounds depend on driver_p, so we must use the
# coupled bounds explicitly, not the constants from BOUNDS[2] / BOUNDS[4].

def test_lower_physical_bounds_map_to_one():
    """Every variable at its lower physical bound normalises to exactly 1.0.

    For coupled variables:
      p4 lower bound          = 14.62 * driver_p
      reservoir_p lower bound = driver_p  (same as driver fill pressure)
    """
    driver_p = driver_p_lower                 # 1 000 Pa (lower bound for driver_p)
    p4_at_lower = 14.62 * driver_p           # lower bound of p4 range for this driver_p
    reservoir_at_lower = driver_p             # lower bound of reservoir range

    physical = [
        BOUNDS[0][0],       # he = 70  (lower bound)
        driver_p,           # driver_p at lower bound
        p4_at_lower,        # p4 at its coupled lower bound
        BOUNDS[3][0],       # D_throat = 0.05  (lower bound)
        reservoir_at_lower, # reservoir_p at its coupled lower bound
        BOUNDS[5][0],       # buffer_length = 0.05  (lower bound)
    ]
    normalised = transform_single(physical)
    assert normalised == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], rel=1e-9)


def test_upper_physical_bounds_map_to_two():
    """Every variable at its upper physical bound normalises to exactly 2.0.

    For coupled variables:
      p4 upper bound          = 1190.63 * driver_p
      reservoir_p upper bound = 8e6  (absolute cap, independent of driver_p)
    """
    driver_p = driver_p_upper                 # (40/14.62)*1e6 Pa  (upper bound for driver_p)
    p4_at_upper = 1190.63 * driver_p         # upper bound of p4 range for this driver_p
    reservoir_at_upper = BOUNDS[4][1]         # 8e6 Pa  (absolute upper bound)

    physical = [
        BOUNDS[0][1],       # he = 100  (upper bound)
        driver_p,           # driver_p at upper bound
        p4_at_upper,        # p4 at its coupled upper bound
        BOUNDS[3][1],       # D_throat = 0.085  (upper bound)
        reservoir_at_upper, # reservoir_p at upper bound
        BOUNDS[5][1],       # buffer_length = 0.15  (upper bound)
    ]
    normalised = transform_single(physical)
    assert normalised == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], rel=1e-9)


# Midpoint test (uncoupled variables only)
#
# For a linearly-normalised variable, the midpoint of the physical range must
# map to exactly 1.5 (the midpoint of [1, 2]).  This verifies that the scaling
# factor and offset are both correct, not merely one of them.
#
# We test the three cleanly uncoupled variables: percent_He, D_throat,
# buffer_length.  driver_p is also uncoupled but is fixed at an arbitrary value
# here so the coupled variables (p4, reservoir_p) remain physically valid.

def test_midpoint_of_uncoupled_variables_maps_to_1_5():
    """The physical midpoint of each uncoupled variable normalises to 1.5."""
    driver_p = 300_000    # arbitrary reference value for the coupled variables

    he_mid            = (BOUNDS[0][0] + BOUNDS[0][1]) / 2   # 85.0
    D_throat_mid      = (BOUNDS[3][0] + BOUNDS[3][1]) / 2   # 0.0675
    buffer_length_mid = (BOUNDS[5][0] + BOUNDS[5][1]) / 2   # 0.10

    # p4 and reservoir_p are set somewhere valid (not necessarily at their midpoints)
    p4_valid        = 14.62 * driver_p * 1.5   # within the valid coupled range
    reservoir_valid = driver_p * 5             # within [driver_p, 8e6]

    physical = [he_mid, driver_p, p4_valid, D_throat_mid, reservoir_valid, buffer_length_mid]
    normalised = transform_single(physical)

    assert normalised[0] == pytest.approx(1.5, rel=1e-9)   # percent_He
    assert normalised[3] == pytest.approx(1.5, rel=1e-9)   # D_throat
    assert normalised[5] == pytest.approx(1.5, rel=1e-9)   # buffer_length


# Near-bound round-trip test
#
# A physical point slightly outside the lower physical bounds should round-trip
# exactly: transform(physical) -> normalised (slightly below 1) ->
# untransform -> recover original physical value.
#
# This verifies strict 1:1 mapping.  It will FAIL if variable_untransformation
# uses np.abs(x[i] - 1), because the abs() reflects normalised values below 1
# back upward rather than recovering the original sub-bound physical value.
# To make this test pass, replace np.abs(x[i] - 1) with (x[i] - 1) in
# variable_untransformation for all six variables.
#
# Note: physical values slightly ABOVE the upper bound are not tested here
# because a normalised value above 2 still has (x - 1) > 0, so abs() is
# inert in that direction and the round-trip already works.

def test_roundtrip_slightly_below_lower_physical_bound():
    """A physical point slightly below the lower bounds round-trips exactly.
    The uncoupled variables (percent_He, D_throat, buffer_length) are set just
    below their lower bounds.  driver_p, p4, and reservoir_p are kept at valid
    in-bounds values so that the coupling logic is not involved.
    """
    driver_p = 300_000   # valid, in-bounds driver_p

    physical = [
        69.5,                     # percent_He: slightly below lower bound of 70
        driver_p,                 # driver_p: valid
        14.62 * driver_p * 1.5,  # p4: valid (within coupled range)
        0.049,                    # D_throat: slightly below lower bound of 0.05
        driver_p * 3,             # reservoir_p: valid
        0.048,                    # buffer_length: slightly below lower bound of 0.05
    ]
    normalised = transform_single(physical)
    recovered = variable_untransformation(normalised, BOUNDS)
    assert recovered == pytest.approx(physical, rel=1e-9)
