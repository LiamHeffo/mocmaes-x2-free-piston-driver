# tests/test_l1d_watchdog.py
#
# Unit tests for the side-channel watchdog predicates in problem.l1d_job.
# These exercise the pure file-reader helpers (no subprocess needed) by
# writing synthetic data files in the same text format that L1d4 produces
# live during a --run-simulation step:
#
#   piston-0000.data    : "# tindx x vel is_restrain brakes_on on_buffer"
#   diaphragm-0000.data : "# tindx state"
#   times.data          : "# tindx time"

import sys, os

from problem.l1d_job import (
    _count_piston_inflections,
    _piston_velocity_reversed,
    _diaphragm_burst_time,
    _current_sim_time,
)


# _count_piston_inflections

def _write_piston(tmp_path, velocities):
    """Write a piston-0000.data file with the given velocity column."""
    f = tmp_path / "piston-0000.data"
    with f.open("w") as fh:
        fh.write("# tindx  x  vel  is_restrain  brakes_on  on_buffer\n")
        for i, v in enumerate(velocities):
            fh.write(f"{i} 0.0 {v:.6e} 0 0 0\n")
    return str(f)


def test_inflections_zero_when_file_missing(tmp_path):
    assert _count_piston_inflections(str(tmp_path / "nope.data")) == 0


def test_inflections_zero_for_monotonic_velocity(tmp_path):
    # Pure positive ramp -- no sign flips, no inflections.
    f = _write_piston(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert _count_piston_inflections(f) == 0


def test_inflections_one_for_single_reversal(tmp_path):
    # Goes positive, then negative -- one inflection (one sign flip).
    f = _write_piston(tmp_path, [1.0, 2.0, 3.0, -1.0, -2.0])
    assert _count_piston_inflections(f) == 1


def test_inflections_three_for_three_strokes(tmp_path):
    # +,+,-,-,+,+,-,-  -> three sign flips.
    f = _write_piston(tmp_path,
                      [1.0, 2.0, -1.0, -2.0, 1.0, 2.0, -1.0, -2.0])
    assert _count_piston_inflections(f) == 3


def test_inflections_ignores_initial_zero(tmp_path):
    # First sample is always v=0 in L1d output (piston at rest).  That
    # must NOT be counted as its own sign.
    f = _write_piston(tmp_path, [0.0, 1.0, 2.0, 3.0])
    assert _count_piston_inflections(f) == 0


def test_inflections_ignores_zeros_at_rest(tmp_path):
    # Post-buffer-impact rest: trailing v=0 samples must not be counted
    # as a sign change either.
    f = _write_piston(tmp_path, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert _count_piston_inflections(f) == 0


# _piston_velocity_reversed (the new pre-rupture turnaround gate)

def test_reversed_false_when_file_missing(tmp_path):
    assert _piston_velocity_reversed(str(tmp_path / "nope.data")) is False


def test_reversed_false_during_forward_stroke(tmp_path):
    # Piston still accelerating forward (v>0 throughout) -- not yet turned
    # around, so the gate must NOT fire.
    f = _write_piston(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert _piston_velocity_reversed(f) is False


def test_reversed_true_at_first_turnaround(tmp_path):
    # v goes +,+,+ then negative -- the turnaround (peak compression) has
    # happened; one reversal is enough to trip the gate.
    f = _write_piston(tmp_path, [1.0, 2.0, 3.0, -1.0])
    assert _piston_velocity_reversed(f) is True


def test_reversed_ignores_initial_and_rest_zeros(tmp_path):
    # The launch v=0 and any at-rest zeros must not count as a reversal:
    # a pure forward stroke bracketed by zeros has NOT turned around.
    f = _write_piston(tmp_path, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    assert _piston_velocity_reversed(f) is False


# _diaphragm_burst_time and _current_sim_time

def _write_diaphragm(tmp_path, states):
    f = tmp_path / "diaphragm-0000.data"
    with f.open("w") as fh:
        fh.write("# tindx state\n")
        for i, s in enumerate(states):
            fh.write(f"{i} {s}\n")
    return str(f)


def _write_times(tmp_path, times):
    f = tmp_path / "times.data"
    with f.open("w") as fh:
        fh.write("# tindx time\n")
        for i, t in enumerate(times):
            fh.write(f"{i} {t:.6e}\n")
    return str(f)


def test_burst_time_none_before_rupture(tmp_path):
    dia = _write_diaphragm(tmp_path, [0, 0, 0, 0, 0])
    times = _write_times(tmp_path, [0.0, 1e-4, 2e-4, 3e-4, 4e-4])
    assert _diaphragm_burst_time(dia, times) is None


def test_burst_time_picks_first_state_2_row(tmp_path):
    # State flip at tindx=3 -> burst time is times[3].
    dia = _write_diaphragm(tmp_path, [0, 0, 0, 2, 2])
    times = _write_times(tmp_path, [0.0, 1e-4, 2e-4, 3e-4, 4e-4])
    assert _diaphragm_burst_time(dia, times) == 3e-4


def test_burst_time_none_when_tindx_not_yet_in_times(tmp_path):
    # Transient mid-write: diaphragm row exists but times hasn't caught
    # up yet.  Predicate should return None (will re-check next poll).
    dia = _write_diaphragm(tmp_path, [0, 0, 0, 2])
    times = _write_times(tmp_path, [0.0, 1e-4, 2e-4])  # missing tindx=3
    assert _diaphragm_burst_time(dia, times) is None


def test_current_sim_time_returns_last_row(tmp_path):
    times = _write_times(tmp_path, [0.0, 1e-4, 2e-4, 3e-4, 4e-4])
    assert _current_sim_time(times) == 4e-4


def test_current_sim_time_none_when_missing(tmp_path):
    assert _current_sim_time(str(tmp_path / "nope.data")) is None
