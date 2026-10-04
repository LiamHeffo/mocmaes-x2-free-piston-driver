# tests/test_utils.py

import sys, os

import numpy as np
from utils import valid

def test_valid_returns_true_for_in_bounds_point():
    ind = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # middle of [1,2]^6
    assert valid(ind)

def test_valid_returns_false_when_one_variable_above_upper_bound():
    ind = np.array([2.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # first variable > 2
    assert not valid(ind)

def test_valid_returns_false_when_one_variable_below_lower_bound():
    ind = np.array([0.9, 1.5, 1.5, 1.5, 1.5, 1.5])  # first variable < 1
    assert not valid(ind)

def test_valid_returns_false_when_one_variable_on_lower_bound():
    ind = np.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.5])  # first variable = 1
    assert valid(ind)

def test_valid_returns_false_when_one_variable_on_upper_bound():
    ind = np.array([2.0, 1.5, 1.5, 1.5, 1.5, 1.5])  # first variable = 2
    assert valid(ind)
