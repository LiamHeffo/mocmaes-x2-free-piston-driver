# tests/test_hypervolume.py

import sys, os

from algorithm.hypervolume import HyperVolume
import numpy as np
import pytest


def test_hypervolume_single_point_2d():
    """Baseline: one point in 2D, reference at origin.

    Front point [-0.5, -0.5] dominates a square from (-0.5,-0.5) to (0,0).
    HV = 0.5 x 0.5 = 0.25.
    """
    reference = np.array([0.0, 0.0])
    front = np.array([[-0.5, -0.5]])
    hv = HyperVolume(reference)
    result = hv.compute(front)
    assert result == pytest.approx(0.25, rel=1e-9)


def test_hypervolume_two_points_2d_l_shape():
    """Two non-dominated points in 2D form an L-shaped dominated region.

    This tests that the algorithm correctly accumulates area across strips
    without double-counting the overlap between the two dominated boxes.

    Front: A=[-0.8, -0.2], B=[-0.2, -0.8] (neither dominates the other).

    Computed as two vertical strips:
      Strip 1 (x: -0.8 -> -0.2, width 0.6): only A active, height 0.2 -> 0.12
      Strip 2 (x: -0.2 -> 0,   width 0.2): B now active, height 0.8 -> 0.16
      Total HV = 0.28
    """
    reference = np.array([0.0, 0.0])
    front = np.array([[-0.8, -0.2],
                      [-0.2, -0.8]])
    hv = HyperVolume(reference)
    result = hv.compute(front)
    assert result == pytest.approx(0.28, rel=1e-9)


def test_hypervolume_nonzero_reference_applies_shift():
    """Non-zero reference activates the coordinate-shift branch in compute().

    compute() only shifts the front when any(referencePoint) is True.
    This test targets that branch: with reference [1, 1] and front [[0.5, 0.5]],
    the shift maps the front point to [-0.5, -0.5] (same as the baseline test),
    so the expected HV is still 0.25.

    A bug in the shift logic would produce a different value here while leaving
    the zero-reference tests unaffected.
    """
    reference = np.array([1.0, 1.0])
    front = np.array([[0.5, 0.5]])
    hv = HyperVolume(reference)
    result = hv.compute(front)
    assert result == pytest.approx(0.25, rel=1e-9)


def test_hypervolume_single_point_3d():
    """Single point in 3D exercises the full recursive dimension-sweep.

    The 2D tests only exercise hvRecursive(dimIndex=1).  A 3D front triggers
    hvRecursive(dimIndex=2), which recurses back into dimIndex=1 - the
    genuinely recursive part of the algorithm.

    Front point [-0.5, -0.5, -0.5] dominates a cube of side 0.5.
    HV = 0.5 x 0.5 x 0.5 = 0.125.
    """
    reference = np.array([0.0, 0.0, 0.0])
    front = np.array([[-0.5, -0.5, -0.5]])
    hv = HyperVolume(reference)
    result = hv.compute(front)
    assert result == pytest.approx(0.125, rel=1e-9)
