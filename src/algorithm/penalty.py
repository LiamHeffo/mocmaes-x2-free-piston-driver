"""
Constraint-handling via closest-feasible penalty.

ClosestValidPenalty computes a penalised fitness for individuals that fall
outside the feasible domain by evaluating the nearest feasible point and
adding a weighted distance penalty.

STATUS: OBSOLETE - do not use.
The 'Penalty' sim_type is now handled inline inside algorithm/cmaes.py
(StrategyMultiObjective.generate).  ClosestValidPenalty.wrapper() is
incomplete (the 'func' reference is never bound and 'self' is inaccessible
from a plain-function call) and was never exercised in production runs.
This class is retained as a reference for the intended design but should not
be called.
"""

from itertools import repeat
from collections.abc import Sequence


class ClosestValidPenalty(object):
    r"""Return penalised fitness for invalid individuals.

    For a valid individual the original fitness is returned unchanged.
    For an invalid individual the fitness of the closest valid point is
    returned, reduced by a weighted distance penalty:

    .. math::

       f^\mathrm{penalty}_i(\mathbf{x})
           = f_i(\operatorname{valid}(\mathbf{x}))
             - \alpha\, w_i\, d_i(\operatorname{valid}(\mathbf{x}),\, \mathbf{x})

    Parameters
    ----------
    feasibility : callable
        Returns True if an individual is in the feasible domain.
    feasible : callable
        Returns the closest feasible individual to an infeasible one.
    alpha : float
        Multiplicative factor on the distance penalty.
    distance : callable, optional
        Returns a scalar or per-objective distance.  Defaults to zero penalty.
    """

    def __init__(self, feasibility, feasible, alpha, distance=None):
        self.fbty_fct = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance

    def wrapper(individual, *args, **kwargs):
        # NOTE: 'self' and 'func' are not bound here - this method requires
        # refactoring before the Penalty sim_type can be used.
        if self.fbty_fct(individual):
            return func(individual, *args, **kwargs)

        f_ind = self.fbl_fct(individual)
        print("individual", f_ind)
        f_fbl = func(f_ind, *args, **kwargs)
        print("feasible", f_fbl)

        weights = tuple(1.0 if w >= 0 else -1.0 for w in individual.fitness.weights)

        if len(weights) != len(f_fbl):
            raise IndexError("Fitness weights and computed fitness are of different size.")

        dists = tuple(0 for w in individual.fitness.weights)
        if self.dist_fct is not None:
            dists = self.dist_fct(f_ind, individual)
            if not isinstance(dists, Sequence):
                dists = repeat(dists)

        print("penalty ", tuple(-w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
        print("returned", tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
        return tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists))
