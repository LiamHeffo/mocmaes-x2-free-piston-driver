"""Shared test helpers."""
import numpy as np


class Ind(np.ndarray):
    """An ndarray that accepts attributes.

    StrategyMultiObjective tags each parent with a ``_lineage_id``, which a
    bare ndarray rejects. In production it is handed DEAP Individual
    objects, which allow attributes; this is the numeric equivalent for
    tests that only care about the covariance arithmetic.
    """

    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)
