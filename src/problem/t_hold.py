"""
Canonical t_hold computation for the X2 driver optimisation.

This module is the single source of truth for the two functions that
characterise the driver-pressure hold window:

  smooth_pressure   -- Savitzky-Golay low-pass filter.
  compute_t_hold    -- longest-contiguous in-band dwell, with hysteresis.

The parser (problem.l1d_job.parse_l1d_outputs) imports from here, so
the hold-time definition lives in one place: re-tuning any default
below changes it everywhere the hold time is measured.

Defaults reflect the values that were workshopped and validated against
a sweep of historical condition_N runs:

    band_frac        = 0.10     (±10 % band, X2 spec)
    entry_window_s   = 1.0e-3   (1 ms settling tolerance)
    hysteresis_s     = 1.5e-4   (0.15 ms; see DEFAULT_HYSTERESIS_S)
    smooth_window_s  = 5.0e-4   (~50 samples at dt_plot = 10 µs)
    smooth_polyorder = 3        (preserves peaks + gradients)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# Canonical defaults -- the parameters compute_t_hold uses when callers
# don't override.  Re-tune here, both prod and workshop pick it up.
DEFAULT_BAND_FRAC        = 0.10
DEFAULT_ENTRY_WINDOW_S   = 1.0e-3
DEFAULT_SMOOTH_WINDOW_S  = 5.0e-4
DEFAULT_SMOOTH_POLYORDER = 3

# Out-of-band excursions shorter than this do not break the hold window.
#
# Some designs sit steadily inside the band but clip its edge briefly on
# the way past the pressure peak. Without hysteresis such a design is
# scored on whichever fragment is longer, well under its true dwell, and
# the reported hold time becomes sensitive to a fractional change in the
# band width. 0.15 ms is wide enough to bridge those excursions and
# narrow enough to leave genuine dropouts separate; raising it further
# starts merging windows that really are distinct. It is well above the
# sample spacing either way.
DEFAULT_HYSTERESIS_S     = 1.5e-4


# The filter width is derived from the local sample spacing within this
# many filter-widths of t_ref.  Wide enough to hold thousands of samples
# (so the median is stable), narrow enough to exclude the coarse late-time
# tail that made the width depend on run duration.
_LOCAL_DT_SPAN = 10.0


def smooth_pressure(
    t: np.ndarray,
    p: np.ndarray,
    window_s: float | None,
    polyorder: int = DEFAULT_SMOOTH_POLYORDER,
    t_ref: float | None = None,
) -> np.ndarray:
    """
    Low-pass smooth a pressure trace with a Savitzky-Golay filter.

    Parameters
    ----------
    t         : time array (s) -- used only to derive the sample dt
    p         : raw pressure array (Pa)
    window_s  : smoothing window in SECONDS (not samples).  If None, the
                input is returned unchanged.  Internally converted to an
                odd integer number of samples using dt = median(diff(t)).
    polyorder : Savitzky-Golay polynomial order.  3 preserves peaks and
                gradients well; 2 behaves more like a moving average.

    t_ref     : if given, the sample spacing is estimated LOCALLY around
                this time instead of over the whole trace.  Pass t_burst.

    Trace dt depends on the L1d dt_plot setting, which changes between
    runs. Specifying the filter scale physically (time) makes the
    smoothing portable.

    The sample spacing is estimated near ``t_ref`` rather than over the
    whole trace. L1d's output cadence varies by more than two orders of
    magnitude along a single run, so a whole-trace median makes the filter
    width, and with it the measured hold time, depend on how long the
    simulation was allowed to continue rather than on the design. Taking
    the estimate locally keeps the window spanning the intended span of
    real time in the region being measured.
    """
    if window_s is None or len(p) < polyorder + 3:
        return p
    if t_ref is None:
        dt = float(np.median(np.diff(t)))
    else:
        span = _LOCAL_DT_SPAN * window_s
        near = (t >= t_ref - span) & (t <= t_ref + span)
        # Fall back to the global estimate if the local window is too thin
        # to give a meaningful median (very short or coarsely sampled trace).
        dt = float(np.median(np.diff(t[near]))) if np.count_nonzero(near) > 8 \
            else float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return p
    n = int(round(window_s / dt))
    if n < polyorder + 2:
        return p   # window too narrow to be meaningful
    if n % 2 == 0:
        n += 1     # savgol_filter requires an odd window length
    if n > len(p):
        n = len(p) if len(p) % 2 == 1 else len(p) - 1
    return savgol_filter(p, window_length=n, polyorder=polyorder)


def _contiguous_runs(mask):
    """Return ``[(i, j), ...]`` inclusive index pairs of each maximal True run."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends   = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def _bridge_short_excursions(in_band, t, hysteresis_s):
    """Fill interior out-of-band runs shorter than ``hysteresis_s``.

    Only runs bracketed by in-band samples on BOTH sides are bridged: a
    trace that has not yet entered the band, or that has left it for good,
    must not be joined to anything.  This is what gives a momentary spike
    across a band edge grace, without extending the window past its real end.
    """
    if hysteresis_s is None or hysteresis_s <= 0.0:
        return in_band
    out = np.array(in_band, dtype=bool, copy=True)
    n = out.size
    for i, j in _contiguous_runs(~in_band):
        if i == 0 or j == n - 1:
            continue
        if (t[j] - t[i]) < hysteresis_s:
            out[i:j + 1] = True
    return out


def compute_t_hold(
    t: np.ndarray,
    p: np.ndarray,
    p_burst: float,
    t_burst: float,
    band_frac: float = DEFAULT_BAND_FRAC,
    entry_window_s: float = DEFAULT_ENTRY_WINDOW_S,
    smooth_window_s: float | None = DEFAULT_SMOOTH_WINDOW_S,
    smooth_polyorder: int = DEFAULT_SMOOTH_POLYORDER,
    band_frac_hi: float | None = None,
    hysteresis_s: float | None = DEFAULT_HYSTERESIS_S,
) -> tuple[float, float, float]:
    """
    Duration for which the driver pressure holds inside the band after rupture.

    Returns
    -------
    (t_hold, t_start, t_exit)
        The LONGEST contiguous in-band interval that begins within
        ``entry_window_s`` of ``t_burst``.  ``(0.0, t_burst, t_burst)`` if
        no qualifying interval exists.

    Algorithm
    ---------
    Phase 0  Savitzky-Golay smoothing of p (if smooth_window_s is set).
    Phase 1  in-band mask over the post-rupture trace.
    Phase 2  bridge out-of-band excursions shorter than ``hysteresis_s``.
    Phase 3  take the LONGEST surviving contiguous run whose start lies
             within ``entry_window_s`` of t_burst.

    The window is the longest contiguous span inside the band, not the
    time to the first band exit. A trace oscillating near an edge can
    cross it briefly without the driver condition changing, so a
    first-exit rule measures the ripple rather than the design and gives
    very different readings for traces that are physically the same.
    Taking the longest contiguous span is stable against that.

    The band
    --------
    ``band_frac`` sets the lower edge and, by default, the upper edge too.
    Pass ``band_frac_hi`` to widen ONLY the ceiling -- a design that settles
    slightly above p_burst and holds there is a usable driver condition, but
    a symmetric band scores it zero.  ``band_frac_hi=None`` (the default)
    reproduces the symmetric +-band_frac spec exactly.
    """
    # t_ref=t_burst: size the filter from the sampling rate where the
    # measurement happens, not from the whole-trace median.
    p = smooth_pressure(t, p, smooth_window_s, smooth_polyorder, t_ref=t_burst)

    frac_hi = band_frac if band_frac_hi is None else band_frac_hi
    p_lo = p_burst * (1.0 - band_frac)
    p_hi = p_burst * (1.0 + frac_hi)

    k0 = int(np.searchsorted(t, t_burst, side="left"))
    if k0 >= len(t):
        return 0.0, t_burst, t_burst          # burst beyond end of trace

    ts, ps = t[k0:], p[k0:]
    in_band = (ps >= p_lo) & (ps <= p_hi)
    if not in_band.any():
        return 0.0, t_burst, t_burst

    in_band = _bridge_short_excursions(in_band, ts, hysteresis_s)

    best = None
    for i, j in _contiguous_runs(in_band):
        if (ts[i] - t_burst) > entry_window_s:
            continue                          # hold began too late to count
        duration = float(ts[j] - ts[i])
        if best is None or duration > best[0]:
            best = (duration, float(ts[i]), float(ts[j]))

    if best is None:
        return 0.0, t_burst, t_burst
    return best
