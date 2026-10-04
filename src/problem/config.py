"""
X2 free-piston driver problem definition.

Centralises:
  - Problem dimensionality and normalised-space bounds (used by validity checks)
  - Physical design-variable bounds
  - Objective reference points (ideal and nadir) used for normalisation
  - PITOT3 configuration dictionary builders
"""

import numpy as np

# Problem size

N = 6  # number of design variables

# The algorithm works in a normalised space where every variable lives in [1, 2].
# These arrays are used by the feasibility checks in utils.py.
MIN_BOUND = np.ones(N)
MAX_BOUND = np.ones(N) + 1

# Objective reference points  (physical units)

# Ideal point: best conceivable values for each objective
#   (delta_vs1=0, hold_time=0.005 s, impact_speed=0 m/s)
APPROX_IDEAL = (0, 0.005, 0)

# Nadir point: worst expected values for each objective
#   (delta_vs1=3585 m/s, hold_time=0 s, impact_speed=350 m/s)
# delta_vs1's nadir matches the vs1-failure sentinel (|vs1 - 3585| with
# vs1 -> 0 = 3585) so a sentinel individual normalises to exactly 1.0,
# which the legacy/Penalty-mode detection (normalised delta_vs1 == 1.0)
# relies on.
APPROX_NADIR = (3585, 0, 350)

# 2-objective reference points used by the AL constraint-handling path
# (CHT_AL sim_type), where delta_vs1 is no longer a Pareto objective but a
# constraint adapted via Augmented Lagrangian.  These slice off the
# delta_vs1 entry; their indices match the (hold_time, impact_speed)
# ordering that evaluate.py emits for AL mode.
APPROX_IDEAL_2D = (APPROX_IDEAL[1], APPROX_IDEAL[2])
APPROX_NADIR_2D = (APPROX_NADIR[1], APPROX_NADIR[2])

# Physical design-variable bounds

he_lower,           he_upper           = 70,    100
driver_p_lower,     driver_p_upper     = 1000,  (40 / 14.62) * 1e6
p4_lower,           p4_upper           = 150e3, 40e6
D_throat_lower,     D_throat_upper     = 0.05,  0.085
reservoir_lower,    reservoir_upper    = 1000,  8e6
buffer_length_lower, buffer_length_upper = 0.05, 0.15

BOUNDS = [
    (he_lower,            he_upper),
    (driver_p_lower,      driver_p_upper),
    (p4_lower,            p4_upper),
    (D_throat_lower,      D_throat_upper),
    (reservoir_lower,     reservoir_upper),
    (buffer_length_lower, buffer_length_upper),
]

# PITOT3 configuration builders

def base_config_dict():
    """Return the base PITOT3 facility configuration dictionary."""
    return {
        'mode':               'fully_theoretical',
        'output_filename':    'optimisation_test_setup',
        'facility':           'x2_nrst_85_mm_shock_tube',
        'driver_condition':   'custom_from_dict',
        'test_gas_gas_model': 'CEAGas',
        'test_gas_name':      'he-with-ions',
        'p1':                 150e3,
    }


def base_driver_dict(x):
    """
    Build the PITOT3 driver condition dictionary from a physical-space
    design vector x = [percent_He, driver_p, p4, D_throat, reservoir_p, buffer_length].
    """
    driver_dict = {
        'percent_He':    x[0],
        'driver_p':      x[1],
        'p4':            x[2],
        'D_throat':      x[3],
        'reservoir_p':   x[4],
        'buffer_length': x[5],
    }

    driver_dict['driver_fill_composition'] = {
        'He': float(driver_dict['percent_He'] / 100),
        'Ar': float(1 - driver_dict['percent_He'] / 100),
    }
    driver_dict['driver_condition_name']  = 'x2lwp-2.0mm-0'
    driver_dict['driver_condition_type']  = 'empirical'
    driver_dict['driver_gas_model']       = 'thermally-perfect-preset'
    driver_dict['driver_fill_gas_name']   = 'he-ar'
    driver_dict['driver_speciesList']     = list(driver_dict['driver_fill_composition'].keys())
    driver_dict['driver_inputUnits']      = 'moles'
    driver_dict['driver_withIons']        = False
    driver_dict['M_throat']               = 1.0

    return driver_dict
