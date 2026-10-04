"""Unit tests for the t_hold hold-window measurement.

These use synthetic traces with smoothing disabled (``smooth_window_s=None``)
so they exercise the window logic directly rather than the filter.  The real
L1d archive does not cover the new code paths -- no archived design spends any
time above the band, and none has an out-of-band excursion short enough for
the hysteresis to bridge -- so without synthetic cases those branches would be
untested.
"""
import sys, os

import numpy as np
import pytest

from problem.t_hold import compute_t_hold

DT = 1.0e-5          # 10 us sampling
P_BURST = 10.0e6
T_BURST = 2.0e-3
KW = dict(smooth_window_s=None)      # test the logic, not the filter


def _trace(segments):
    """Build (t, p) from ``[(duration_s, pressure_Pa), ...]`` starting at t=0."""
    ps = []
    for dur, val in segments:
        ps.append(np.full(int(round(dur / DT)), float(val)))
    p = np.concatenate(ps)
    return np.arange(p.size) * DT, p


def test_simple_hold_is_measured_from_burst():
    t, p = _trace([(T_BURST, 0.5 * P_BURST),      # pre-burst ramp region
                   (5.0e-3, P_BURST),             # 5 ms in band
                   (5.0e-3, 0.5 * P_BURST)])      # decay out
    t_hold, t_start, t_exit = compute_t_hold(t, p, P_BURST, T_BURST, **KW)
    assert t_hold == pytest.approx(5.0e-3, abs=2 * DT)
    assert t_start == pytest.approx(T_BURST, abs=2 * DT)
    assert t_exit == pytest.approx(T_BURST + 5.0e-3, abs=2 * DT)


def test_longest_window_wins_not_the_first():
    """A brief hold followed by a longer one must report the LONGER one.

    This is the regression guard for the old first-exit behaviour, which
    returned the short leading window and discarded the real hold.
    """
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (0.5e-3, P_BURST),             # short hold
                   (0.5e-3, 0.5 * P_BURST),       # genuine drop-out
                   (4.0e-3, P_BURST),             # the real hold
                   (2.0e-3, 0.5 * P_BURST)])
    # entry_window must admit the later window's start for it to qualify.
    t_hold, t_start, _ = compute_t_hold(
        t, p, P_BURST, T_BURST, entry_window_s=2.0e-3, **KW)
    assert t_hold == pytest.approx(4.0e-3, abs=2 * DT)
    assert t_start == pytest.approx(T_BURST + 1.0e-3, abs=2 * DT)


def test_entry_window_rejects_a_late_hold():
    """A long hold that starts after entry_window_s does not count."""
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (3.0e-3, 0.5 * P_BURST),       # still below band
                   (4.0e-3, P_BURST)])
    t_hold, t_start, t_exit = compute_t_hold(
        t, p, P_BURST, T_BURST, entry_window_s=1.0e-3, **KW)
    assert t_hold == 0.0
    assert t_start == T_BURST and t_exit == T_BURST


def test_hysteresis_bridges_a_momentary_spike():
    """A brief excursion above the ceiling must not end the window."""
    segs = [(T_BURST, 0.5 * P_BURST),
            (2.0e-3, P_BURST),
            (0.05e-3, 1.3 * P_BURST),             # 50 us spike above band
            (2.0e-3, P_BURST),
            (2.0e-3, 0.5 * P_BURST)]
    t, p = _trace(segs)
    # Without grace the spike splits the hold into two ~2 ms halves.
    short, _, _ = compute_t_hold(t, p, P_BURST, T_BURST, hysteresis_s=0.0, **KW)
    assert short == pytest.approx(2.0e-3, abs=2 * DT)
    # With grace wider than the spike the window spans it.
    bridged, _, _ = compute_t_hold(t, p, P_BURST, T_BURST,
                                   hysteresis_s=0.1e-3, **KW)
    assert bridged == pytest.approx(4.05e-3, abs=2 * DT)


def test_default_hysteresis_bridges_a_ceiling_clip():
    """Regression: a marginal clip of the ceiling must not split the hold.

    Reproduces a design that holds inside the band for
    1.553 ms but grazes the +10 % ceiling for 0.1101 ms on the way past
    its peak.  Under the previous 0.10 ms default that excursion missed
    being bridged by 10 us, so compute_t_hold returned the longer of the
    two fragments (0.745 ms) instead of the true dwell.  The excursion
    here is 0.11 ms -- between the old default and the new one -- so this
    test fails if DEFAULT_HYSTERESIS_S is ever moved back below it.
    """
    segs = [(T_BURST, 0.5 * P_BURST),
            (0.70e-3, P_BURST),
            (0.11e-3, 1.101 * P_BURST),           # 0.11 ms, 0.1 % over
            (0.75e-3, P_BURST),
            (2.0e-3, 0.5 * P_BURST)]
    t, p = _trace(segs)
    t_hold, t_start, _ = compute_t_hold(t, p, P_BURST, T_BURST, **KW)
    assert t_hold == pytest.approx(1.56e-3, abs=3 * DT)
    # And the window must start at rupture, not after the clip -- the
    # late start is how the artifact showed up in the overlay figure.
    assert t_start == pytest.approx(T_BURST, abs=2 * DT)


def test_hysteresis_does_not_bridge_a_genuine_dropout():
    """An excursion longer than the grace still ends the window."""
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (2.0e-3, P_BURST),
                   (0.5e-3, 0.5 * P_BURST),       # 0.5 ms genuine drop-out
                   (2.0e-3, P_BURST),
                   (2.0e-3, 0.5 * P_BURST)])
    t_hold, _, _ = compute_t_hold(t, p, P_BURST, T_BURST,
                                  hysteresis_s=0.1e-3, **KW)
    assert t_hold == pytest.approx(2.0e-3, abs=2 * DT)


def test_hysteresis_never_extends_past_the_end_of_the_hold():
    """Only excursions bracketed by in-band on both sides may be bridged."""
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (3.0e-3, P_BURST),
                   (0.05e-3, 0.5 * P_BURST)])     # trailing, never returns
    t_hold, _, t_exit = compute_t_hold(t, p, P_BURST, T_BURST,
                                       hysteresis_s=1.0e-3, **KW)
    assert t_hold == pytest.approx(3.0e-3, abs=2 * DT)
    assert t_exit <= t[-1]


def test_upper_band_grace_admits_a_sustained_high_hold():
    """A design holding just above the symmetric ceiling scores zero by
    default, and its full dwell once band_frac_hi is widened."""
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (5.0e-3, 1.11 * P_BURST),      # 11% above target
                   (2.0e-3, 0.5 * P_BURST)])
    symmetric, _, _ = compute_t_hold(t, p, P_BURST, T_BURST, **KW)
    assert symmetric == 0.0

    graced, t_start, _ = compute_t_hold(
        t, p, P_BURST, T_BURST, band_frac_hi=0.15, **KW)
    assert graced == pytest.approx(5.0e-3, abs=2 * DT)
    assert t_start == pytest.approx(T_BURST, abs=2 * DT)


def test_upper_band_grace_leaves_the_floor_alone():
    """band_frac_hi must not widen the lower edge."""
    t, p = _trace([(T_BURST, 0.5 * P_BURST),
                   (5.0e-3, 0.86 * P_BURST),      # 14% BELOW target
                   (2.0e-3, 0.5 * P_BURST)])
    t_hold, _, _ = compute_t_hold(
        t, p, P_BURST, T_BURST, band_frac_hi=0.15, **KW)
    assert t_hold == 0.0


def test_no_in_band_samples_returns_zero():
    t, p = _trace([(T_BURST, 0.5 * P_BURST), (5.0e-3, 0.5 * P_BURST)])
    assert compute_t_hold(t, p, P_BURST, T_BURST, **KW)[0] == 0.0


def test_burst_beyond_end_of_trace_returns_zero():
    t, p = _trace([(2.0e-3, P_BURST)])
    t_hold, t_start, t_exit = compute_t_hold(t, p, P_BURST, 99.0, **KW)
    assert (t_hold, t_start, t_exit) == (0.0, 99.0, 99.0)
