# tests/test_transforms.py

import sys, os

import pytest
from problem.transforms import normalise_fitness, unnormalise_fitness

# (x, y, z) = residual shock speed, hold time, impact speed
IDEAL = (0, 0.01, 0)
NADIR = (3500, 0.1, 350)

""" Defining three test cases. One for ensuring that the ideal values map to 0,
 one for ensuring that the nadir values map to 1, and one for ensuring that the
middle values map to 0.5. """

def test_normalise_fitness_ideal_point_maps_to_zero():
    result = normalise_fitness(IDEAL, IDEAL, NADIR)
    assert result == pytest.approx((0.0, 0.0, 0.0), rel=1e-9)

def test_normalise_fitness_nadir_point_maps_to_one():
    result = normalise_fitness(NADIR, IDEAL, NADIR)
    assert result == pytest.approx((1.0, 1.0, 1.0), rel=1e-9)

def test_normalise_fitness_middle_point_maps_to_half():
    raw = ((IDEAL[0] + NADIR[0]) / 2,
           (IDEAL[1] + NADIR[1]) / 2,
           (IDEAL[2] + NADIR[2]) / 2)
    result = normalise_fitness(raw, IDEAL, NADIR)
    assert result == pytest.approx((0.5, 0.5, 0.5), rel=1e-9)

def test_normalise_unnormalise_fitness_consistency():
    raw = ((IDEAL[0] + NADIR[0]) / 2,
           (IDEAL[1] + NADIR[1]) / 2,
           (IDEAL[2] + NADIR[2]) / 2)
    result = normalise_fitness(raw, IDEAL, NADIR)
    recovered = unnormalise_fitness(result, IDEAL, NADIR)
    assert recovered == pytest.approx(raw, rel=1e-9)
