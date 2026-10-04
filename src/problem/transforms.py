"""
Variable transformation and fitness normalisation utilities.

The optimisation operates in a normalised space where every design variable
lives in [1, 2].  These functions map between that normalised space and the
physical (dimensional) space.

Physical variables and their bounds:
    0  percent_He          [70, 100]
    1  driver_p            [1000, (40/14.62)*1e6]  Pa
    2  p4                  [14.62*driver_p, 1190.63*driver_p]  (pressure-ratio coupled)
    3  D_throat            [0.05, 0.085]  m
    4  reservoir_p         [driver_p, 8e6]  (driver_p-coupled lower bound)
    5  buffer_length       [0.05, 0.15]  m
"""

import numpy as np


def variable_transformation(pop, bounds):
    """Map a population from physical space to normalised [1, 2]^6 space."""
    new_population = []
    for x in pop:
        x_new_0 = (x[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) + 1  # percent_he
        x_new_1 = (x[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) + 1  # driver_p
        x_new_2 = (x[2] - 14.62 * x[1]) / (1190.63 * x[1] - 14.62 * x[1]) + 1  # p4
        x_new_3 = (x[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0]) + 1  # D_throat
        x_new_4 = (x[4] - x[1]) / (bounds[4][1] - x[1]) + 1  # reservoir_p
        x_new_5 = (x[5] - bounds[5][0]) / (bounds[5][1] - bounds[5][0]) + 1  # buffer_length

        x_new = [x_new_0, x_new_1, x_new_2, x_new_3, x_new_4, x_new_5]
        new_population.append(x_new)

    return new_population


def variable_untransformation(x, bounds):
    """Map a single individual from normalised [1, 2]^6 space back to physical space."""
    x_new_0 = (x[0] - 1) * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    x_new_1 = (x[1] - 1) * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
    x_new_2 = (x[2] - 1) * (1190.63 * x_new_1 - 14.62 * x_new_1) + 14.62 * x_new_1
    x_new_3 = (x[3] - 1) * (bounds[3][1] - bounds[3][0]) + bounds[3][0]
    x_new_4 = (x[4] - 1) * (bounds[4][1] - x_new_1) + x_new_1
    x_new_5 = (x[5] - 1) * (bounds[5][1] - bounds[5][0]) + bounds[5][0]

    return [x_new_0, x_new_1, x_new_2, x_new_3, x_new_4, x_new_5]


def normalise_fitness(fitness, ideal_point, nadir_point):
    """Linearly scale raw objective values into [0, 1] using ideal/nadir reference points."""
    return tuple(
        (np.array(fitness) - np.array(ideal_point))
        / (np.array(nadir_point) - np.array(ideal_point))
    )


def unnormalise_fitness(fitness, ideal_point, nadir_point):
    """Inverse of normalise_fitness - recover physical objective values."""
    return tuple(
        np.array(fitness) * (np.array(nadir_point) - np.array(ideal_point))
        + np.array(ideal_point)
    )
