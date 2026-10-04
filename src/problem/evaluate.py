"""
X2 free-piston driver evaluation - L1d4 implementation.

This is the l1d_cht_al branch's replacement for the SPARK + PITOT3
evaluation path on the parent CHT_AL branch.  All three measured
quantities now come from a single L1d4 simulation per individual:

    hold_time        : duration that the driver pressure at the primary
                       diaphragm sits within ±10% of p_burst, post-burst.
    impact_speed     : piston velocity at the moment on_buffer flips.
    delta_vs1        : |vs1 - 3585| from time-of-flight between two
                       shock-tube transducers.

The function signatures, sentinel values, and return shapes match the
SPARK + PITOT3 path so that ``main.py`` (sentinel detection, AL plumbing,
logbook counters) is untouched.

Sentinel encoding (preserved across the SPARK -> L1d port):
    objectives  : (hold_time, impact_speed) = (0, 350) on failure
    constraint  : delta_vs1 = 3585 on failure (the _PITOT3_FAILURE_SENTINEL
                  constant that main._detect_sentinels greps against; equals
                  VS1_TARGET, i.e. |vs1 - 3585| with vs1 -> 0).
"""
import sys

import numpy as np

from problem.config import (
    APPROX_IDEAL, APPROX_NADIR,
    APPROX_IDEAL_2D, APPROX_NADIR_2D,
    BOUNDS, base_config_dict,
)
from problem.feasibility import evaluate_constraints, is_feasible
from problem.transforms import variable_untransformation, normalise_fitness
from problem.l1d_job import run_l1d, TRANSDUCER_XS, SENTINEL_VS_DELTA
from utils import valid
from algorithm.penalty import ClosestValidPenalty


# Sentinel constant - name retained for main._detect_sentinels (which still
# imports it as _PITOT3_FAILURE_SENTINEL).  Semantically it is the L1d
# vs1-failure sentinel: the delta_vs1 = |vs1 - VS1_TARGET| recorded when no
# shock is detected (vs1 -> 0), i.e. VS1_TARGET (= 3585) itself.
#
# Single-sourced from l1d_job.SENTINEL_VS_DELTA - the value run_l1d already
# returns on failure - so the run's sentinel and the value the rest of the
# pipeline matches against can never drift apart (they previously did:
# run_l1d said 3585 while this constant said 3500, so every failure was
# silently recorded as 3500).

_PITOT3_FAILURE_SENTINEL = SENTINEL_VS_DELTA


# Logbook injection - unchanged

_logbook = None


def set_logbook(logbook):
    """Register the run logbook so that evaluate() can update counters."""
    global _logbook
    _logbook = logbook


def _log_failure():
    """Increment the SPARK-failure counters when a heavy evaluation fails.

    Counter names retained from the SPARK era - they still measure
    "individual produced no usable objective values" on the L1d branch,
    so the semantics survive the port.
    """
    if _logbook is not None:
        _logbook.bookshelf["No. individuals that failed objective tests"] += 1
        _logbook.bookshelf["No. individuals that produced no hold time"]  += 1


# Heuristic pre-checks
# These guarded the SPARK/PITOT3 pipeline against obviously-broken designs.
# They are tool-independent (they enforce physical sanity on the design
# vector itself), so they are preserved verbatim.

def _heuristic_passes(driver_dict):
    """Return True if the design clears the cheap physical pre-checks."""
    if driver_dict['driver_p'] > driver_dict['reservoir_p']:
        return False
    if 0.0849 < driver_dict['D_throat'] < 0:    # legacy clause; kept for parity
        return False
    pressure_ratio = driver_dict["p4"] / driver_dict["driver_p"]
    compression_ratio = pressure_ratio ** (1 / 1.667)
    if not (5 < compression_ratio < 70):
        return False
    if driver_dict['p4'] > BOUNDS[2][1]:
        return False
    return True


# Single heavy evaluation

def _evaluate_l1d(x_normalised, ind_number, bounds):
    """Run L1d once for ``x`` and return ``(t_hold, impact, delta_vs)``.

    Returns the SPARK-equivalent sentinel triple on any failure or
    pre-check rejection, so the caller does not need to distinguish.
    """
    x_phys = variable_untransformation(x_normalised, bounds)
    driver_dict = {
        'percent_He':    x_phys[0],
        'driver_p':      x_phys[1],
        'p4':            x_phys[2],
        'D_throat':      x_phys[3],
        'reservoir_p':   x_phys[4],
        'buffer_length': x_phys[5],
    }

    if not _heuristic_passes(driver_dict):
        _log_failure()
        return 0.0, 350.0, _PITOT3_FAILURE_SENTINEL

    # Test-gas fill pressure is read from the existing PITOT3 config dict
    # so the value stays single-source: editing config.base_config_dict()
    # updates both the SPARK/PITOT3 branch and the L1d branch.
    test_gas_p1 = float(base_config_dict()['p1'])

    t_hold, impact_speed, delta_vs, ok = run_l1d(
        x_phys=x_phys,
        ind_number=ind_number,
        test_gas_p1=test_gas_p1,
        transducer_xs=TRANSDUCER_XS,
    )
    if not ok:
        _log_failure()
        return 0.0, 350.0, _PITOT3_FAILURE_SENTINEL
    return t_hold, impact_speed, delta_vs


# Combined evaluation - public entry point

def evaluate(x):
    """Evaluate constraints and (if feasible) fitness for one individual.

    Returns
    -------
    (fit, g, g_al) : 3-tuple
        fit  : tuple of normalised objective values, or None.
                 - For legacy sim_types: 3-tuple (delta_vs, hold_time, impact).
                 - For 'CHT_AL': 2-tuple (hold_time, impact).
                 None means infeasible (box+phys violated, or in CHT_AL mode
                 the L1d simulation failed).
        g    : np.ndarray, the box+physical constraint vector (length 18).
        g_al : np.ndarray of length 1, or None.
                 - For 'CHT_AL': np.array([delta_vs - al_tol]).
                 - For all other sim_types: None.
    """
    g = evaluate_constraints(x, x.bounds)
    ind_number = getattr(x, "ind_number", 0)

    if x.sim_type == "Penalty":
        # Penalty mode: always produce a fitness; closest-valid penalty
        # for box-violators.  The L1d evaluation happens only inside the
        # valid() branch - invalid candidates never reach the simulator.
        if valid(x):
            t_hold, impact, delta_vs = _evaluate_l1d(x, ind_number, x.bounds)
            fit = normalise_fitness(
                (delta_vs, t_hold, impact), APPROX_IDEAL, APPROX_NADIR,
            )
        else:
            fit = ClosestValidPenalty.wrapper(x)
        return fit, g, None

    if x.sim_type in ("CHT_AL", "ArnoldCHT_AL", "Resampling_AL"):
        # AL family: delta_vs becomes the AL constraint; objectives are 2-D
        # (hold_time, impact).  The pre-eval handler differs by sim_type
        # (Chocat cov-shrink / Arnold cov-shrink / pure rejection) but the
        # evaluation contract here is identical.  NOTE: this tuple must track
        # AL_ENABLED_SIM_TYPES in algorithm/cmaes.py - a missing member here
        # silently routes an AL run into the legacy 3-objective branch below.
        if not is_feasible(g):
            return None, g, None
        t_hold, impact, delta_vs = _evaluate_l1d(x, ind_number, x.bounds)
        fit_2d = normalise_fitness(
            (t_hold, impact), APPROX_IDEAL_2D, APPROX_NADIR_2D,
        )
        al_tol = getattr(x, "al_tol", 100.0)
        g_al = np.array([delta_vs - al_tol], dtype=float)
        return fit_2d, g, g_al

    # Legacy 3-objective sim_types (CovarianceCHT etc.).
    if not is_feasible(g):
        return None, g, None
    t_hold, impact, delta_vs = _evaluate_l1d(x, ind_number, x.bounds)
    fit = normalise_fitness(
        (delta_vs, t_hold, impact), APPROX_IDEAL, APPROX_NADIR,
    )
    return fit, g, None
