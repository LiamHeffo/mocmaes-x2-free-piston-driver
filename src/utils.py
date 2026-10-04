"""Small helpers shared across the optimisation pipeline."""

from problem.config import MIN_BOUND, MAX_BOUND


def valid(individual):
    """True if the individual lies inside the normalised search domain."""
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True
