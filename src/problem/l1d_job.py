"""
L1d4 job-script templating, subprocess driver, and output parsing for the
X2 free-piston driver optimisation.

Three responsibilities:

1. ``write_job_script(...)`` - materialise an L1d4 input script for one
   individual.  The script is syntactically a valid Python script that
   ``l1d4-prep`` ingests via ``exec()`` with the L1d helper objects
   (``add_break_point``, ``Piston``, ``GasSlug`` ...) bound into the
   globals.

2. ``parse_l1d_outputs(...)`` - read the simulation outputs and return
   ``(t_hold, impact_speed, vs1, ok)``.  Failure to extract any of the
   three returns the existing SPARK/PITOT3 sentinel values so the
   downstream CHT_AL machinery sees the same encoding it does today.

3. ``run_l1d(...)`` - the top-level wrapper called by ``evaluate.py``.
   Owns the per-individual working directory, the three L1d subprocess
   invocations, the timeout, and the cleanup.

Per-individual filesystem layout::

    L1d_Outputs/DEAP_<i>/
    |-- DEAP_<i>.py                  <- the templated job script
    |-- ideal_air.lua                <- symlinked gas models
    |-- he-ar-gas-model.lua
    |-- piston-0000-history.data     <- produced by --piston-history
    `-- DEAP_<i>/                    <- L1d's job-name subdirectory
        |-- diaphragm-0000.data
        |-- history-loc-0000.data    <- shock-tube transducer 1 (vs1 ToF)
        |-- history-loc-0001.data    <- shock-tube transducer 2 (vs1 ToF)
        |-- history-loc-0002.data    <- driver-side probe (t_hold)
        |-- piston-0000.data
        `-- slug-*.data
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from problem.l1d_geometry import (
    build_break_points,
    PISTON_FRONT_X,
    LAUNCHER_LARGE_D,
    BUFFER_PLATE_X_NOMINAL,
)
from problem.t_hold import compute_t_hold

# Piston half-length: distance from the centre (what L1d tracks) to either face.
# L1d's x_buffer triggers when piston CENTRE crosses x_buffer, so to stop the
# FRONT FACE at a target axial coordinate x_target we must pass
# (x_target - PISTON_HALF_LENGTH) as x_buffer.  The target here is the upstream
# tip of the buffer studs (x_stud_tip = BUFFER_PLATE_X_NOMINAL - buffer_length):
# this models the piston coming to a hard stop the instant its front face
# first contacts the buffer studs, with no crush distance.
PISTON_HALF_LENGTH = (PISTON_FRONT_X - LAUNCHER_LARGE_D[0]) / 2   # = 0.1105 m


# Module-level configuration constants (edit these to retune the bring-up)

# Failure sentinels - must match the values the rest of the pipeline
# (evaluate.py, main.py _detect_sentinels) expects.
SENTINEL_OBJECTIVE = (0.0, 350.0)
SENTINEL_VS_DELTA  = 3585.0
SENTINEL_TUPLE     = (*SENTINEL_OBJECTIVE, SENTINEL_VS_DELTA, False)
VS1_TARGET         = 3585.0

# Gas-model file names (these match the user's existing prep-gas outputs
# in the project root) and are symlinked into each per-individual job dir.
# Both the driver gas and the shock-tube test gas use he-ar-gas-model.lua
# (the test gas is pure He via a massf override); the reservoir uses
# ideal_air.lua.  A missing file raises a clear error rather than silently
# failing inside l1d4-prep.
GAS_MODEL_FILES = ("ideal_air.lua", "he-ar-gas-model.lua")

# Per-evaluation budget.  L1d's t_finish is set to 28 ms simulated time;
# wall-clock varies with mesh.  The 60-min wall budget mirrors the SPARK
# path; widen if mesh_scale_factor is raised.
EVAL_TIMEOUT_S       = 60 * 60
PREP_TIMEOUT_S       = 2 * 60
POSTPROCESS_TIMEOUT_S = 60

# Watchdog (early-termination) parameters
# The --run-simulation step writes piston-0000.data, diaphragm-0000.data
# and times.data continuously while the sim advances.  A side-channel
# watchdog re-reads those files every WATCHDOG_POLL_S seconds and SIGTERMs
# the child early on any of:
#   (a) FAIL-FAST (reversal): the piston's velocity has changed sign at
#       least once (it has turned around) without the diaphragm reaching
#       state==2.  Driver pressure is MAXIMISED at the turnaround -- the
#       instant v first crosses zero -- because the driver gas is then
#       most compressed; the piston rebounds afterwards and no later
#       stroke recovers that pressure (energy only dissipates).  So if
#       p4 was not beaten at the first turnaround it never will be.
#       Gated by FAIL_ON_PISTON_REVERSAL.
#   (b) FAIL-FAST (legacy): the piston has inflected (sign-flipped
#       velocity) MAX_INFLECTIONS_NO_RUPTURE times without rupture.  This
#       is the more conservative predecessor of (a); with (a) enabled it
#       is subsumed (1 reversal fires before 3 inflections) and acts only
#       as a fallback when FAIL_ON_PISTON_REVERSAL is False.
#   (c) SUCCESS-AND-DONE: the diaphragm has burst AND simulated time has
#       advanced GRACE_SIM_TIME_S past t_burst.  t_exit and both shock
#       arrivals all fall inside that window in practice.
# All three predicates short-circuit L1d's natural t_finish run, saving
# wall clock on clearly-doomed and clearly-finished evaluations alike.
# The outer EVAL_TIMEOUT_S budget remains the hard safety ceiling.
WATCHDOG_POLL_S            = 60
FAIL_ON_PISTON_REVERSAL    = True   # gate (a): kill on first v sign change pre-rupture
MAX_INFLECTIONS_NO_RUPTURE = 3      # gate (b): legacy fallback, see note above
GRACE_SIM_TIME_S           = 5.0e-3
WATCHDOG_KILL_GRACE_S      = 5

# Positional tolerance when testing whether the piston has reached the
# buffer plane.  The job script writes x_buffer with "%.6f", so the
# threshold L1d receives is quantised to 1 um; comparing a full-precision
# piston position against it can miss a graze by less than that.
X_BUFFER_TOL = 1.0e-6

# Mesh & wall-resolution controls.
MESH_SCALE_FACTOR    = 1     # per-slug ncells multiplier
TUBE_N               = 4000   # tube-wall mesh resolution

# Time-stepping constants.
T_FINISH             = 90.0e-3
T_SWITCH             = 20.0e-3

# Provisional X2-default transducer x-positions (relative to PD at x=0).
# TRANSDUCER_XS = (4.231, 4.746)
TRANSDUCER_XS = (1.5, 2.5)

# Keep failed (sentinel) job directories on disk for post-hoc debugging.
KEEP_FAILED_JOBS     = False

# Keep successful (non-sentinel) job directories on disk.  Lets a
# high-volume sweep retain only the FEW designs that ruptured while
# discarding the many no-rupture jobs per-eval - the no-rupture runs go
# the full t_finish and write the largest files, so they are exactly the
# ones that must NOT accumulate (see the disk-blowup note for KEEP_ALL_JOBS).
KEEP_SUCCESSFUL_JOBS = False

# Keep every job directory (successful or failed) on disk.  Useful during
# bring-up when you want to inspect the L1d outputs after each call.
# Must be False before kicking off parallel CMA-ES runs OR the population
# initialiser - otherwise L1d_Outputs/ grows unbounded (λ workers x many
# evaluations) and fills the disk mid-run (OSError 28).
KEEP_ALL_JOBS        = True

# Stream the stdout/stderr of every L1d subprocess directly to the parent
# terminal.  Useful during bring-up so you can watch the run progress.
# Flip to False before kicking off parallel CMA-ES runs - otherwise the
# λ workers will interleave their output and the terminal becomes
# unreadable, and the captured-stderr error path below won't have the
# diagnostic text to print on subprocess failure.
STREAM_L1D_OUTPUT    = True


# Job-script template
# The L1d4-prep tool exec()'s this file with the L1d helper API bound
# into globals.  Everything inside {{double-braces}} is a literal brace
# that needs to survive .format(); everything inside {single-braces} is a
# placeholder that write_job_script() fills in.

JOB_SCRIPT_TEMPLATE = '''\
# L1d4 job script - auto-generated for individual {ind_number}.
# Do not edit by hand; regenerated per evaluation.
import math

config.title = "X2 driver opt: individual {ind_number}"

# --- Gas models ----------------------------------------------------------
gm_ideal_air = add_gas_model("ideal_air.lua")
gm_he_ar     = add_gas_model("he-ar-gas-model.lua")

# Per-individual He/Ar mass fractions derived from percent_He.
massf_he_ar = config.gmodels[gm_he_ar].molef2massf(
    {{"He": {percent_He:.6f} / 100.0, "Ar": 1.0 - {percent_He:.6f} / 100.0}}
)

# --- Tube geometry (volume-conserving break-points from Hodson 2025) -----
{break_point_calls}
tube.n = {tube_n}

# --- Gas slugs and gas-path objects -------------------------------------
T_amb = {T_amb:.4f}

left_wall = VelocityEnd(x0={reservoir_start_x:.4f}, vel=0.0)

res_gas = GasSlug(
    gmodel_id=gm_ideal_air,
    p={reservoir_p:.6e}, T=T_amb, vel=0.0,
    ncells={n_res}, viscous_effects=1, hcells=1,
    label="reservoir",
)

piston = Piston(
    mass=10.524, diam={D_compression:.4f},
    xL0={piston_xL0:.4f}, xR0={piston_xR0:.4f}, vel0=0.0,
    front_seal_f=0.2,
    front_seal_area=0.020 * {D_compression:.4f} * math.pi,
    x_buffer={x_buffer:.6f}, on_buffer=0,
    label="lightweight piston",
)

driver_gas = GasSlug(
    gmodel_id=gm_he_ar,
    p={driver_p:.6e}, T=T_amb, vel=0.0, massf=massf_he_ar,
    ncells={n_drv}, cluster_strength=1.1, to_end_R=True,
    viscous_effects=1, hcells=1,
    label="driver gas",
)

primary_diaphragm = Diaphragm(x0={pd_x:.4f}, p_burst={p4:.6e}, state=0)

test_gas = GasSlug(
    gmodel_id=gm_he_ar,
    p={test_gas_p1:.6e}, T=T_amb, vel=0.0,
    massf={{"He": 1.0, "Ar": 0.0}},
    ncells={n_test}, cluster_strength=1.1, to_end_L=True,
    viscous_effects=0, hcells=1,
    label="test gas",
)

right_free = FreeEnd(x0={shock_tube_end_x:.4f})

assemble_gas_path(left_wall, res_gas, piston, driver_gas,
                  primary_diaphragm, test_gas, right_free)

# --- Loss regions -------------------------------------------------------
add_loss_region({launcher_loss_x0:.4f}, {launcher_loss_x1:.4f}, 3.1)
add_loss_region({diaphragm_loss_x0:.6f}, {pd_x:.4f}, 0.7)

# --- History locations (indices wired in parse_l1d_outputs) --------------
add_history_loc({transducer_x_1:.4f})        # idx 0 - vs1 ToF station 1
add_history_loc({transducer_x_2:.4f})        # idx 1 - vs1 ToF station 2
add_history_loc({pd_x:.4f} - 0.156)           # idx 2 - driver, 0.0005 m upstream of PD (t_hold probe)


# --- Time stepping ------------------------------------------------------
config.dt_init   = 1.0e-10
config.max_time  = {t_finish:.4e}
config.max_step  = 25_000_000
add_cfl_value(0.0, 0.25)
add_dt_plot(0.0,         2.0e-4, 2.0e-4)
add_dt_plot({t_switch:.4e}, 1.0e-5, 1.0e-6)
'''


# Writing the job script

def write_job_script(out_path, params, ind_number):
    """Materialise the L1d job script for one individual.

    Returns the ``derived`` dict from ``build_break_points`` so the
    parser knows where the primary-diaphragm and transducer stations sit.
    """
    bps, derived = build_break_points(
        buffer_length=params["buffer_length"],
        D_throat=params["D_throat"],
    )
    bp_lines = "\n".join(
        f"add_break_point({x:.6f}, {d:.6f})" for x, d in bps
    )

    # Launcher loss region spans the constant-D=0.1561 launcher tube.
    # End of T3 ramp (small-D side) -> start of launcher exit ramp.
    launcher_loss_x0 = next(x for x, d in bps if abs(d - 0.1561) < 1e-4)
    launcher_loss_x1 = next(x for x, d in reversed(bps) if abs(d - 0.1561) < 1e-4)

    # Piston front face stops when it first contacts the buffer-stud tips,
    # which project upstream from the buffer plate by buffer_length.
    x_stud_tip = BUFFER_PLATE_X_NOMINAL - params["buffer_length"]

    # The buffer plane, in the piston-CENTRE coordinate L1d tracks.  Rounded
    # to the precision the template writes ({:.6f}) so that the value stored
    # in `derived` is bit-identical to the one L1d is actually given - the
    # parser compares piston positions against this threshold, and a
    # sub-micron mismatch between "the number we reason with" and "the number
    # L1d used" is enough to misidentify the contact sample.
    x_buffer = round(x_stud_tip - PISTON_HALF_LENGTH, 6)

    script = JOB_SCRIPT_TEMPLATE.format(
        ind_number=ind_number,
        break_point_calls=bp_lines,
        tube_n=TUBE_N,
        T_amb=298.15,
        reservoir_start_x=-8.7188,
        D_compression=0.2568,
        piston_xL0=derived["piston_xL0"],
        piston_xR0=derived["piston_xR0"],
        x_buffer=x_buffer,
        pd_x=derived["pd_x"],
        shock_tube_end_x=3.0,
        launcher_loss_x0=launcher_loss_x0,
        launcher_loss_x1=launcher_loss_x1,
        diaphragm_loss_x0=derived["x_outer_buffer"],
        transducer_x_1=params["transducer_xs"][0],
        transducer_x_2=params["transducer_xs"][1],
        t_finish=T_FINISH,
        t_switch=T_SWITCH,
        n_res=60 * params["mesh_scale"],
        n_drv=90 * params["mesh_scale"],
        n_test=60 * params["mesh_scale"],
        percent_He=params["percent_He"],
        driver_p=params["driver_p"],
        reservoir_p=params["reservoir_p"],
        p4=params["p4"],
        test_gas_p1=params["test_gas_p1"],
    )

    Path(out_path).write_text(script)
    # Hand the buffer plane back to the caller alongside the geometry, so
    # parse_l1d_outputs can test against the same threshold rather than
    # re-deriving it from the design vector.
    derived["x_buffer"] = x_buffer
    return derived


# Parsing L1d outputs

def _load_history_loc(job_dir, idx):
    """Load one history-loc-NNNN.data file, returning (t, p) columns.

    Verified column layout (L1d 4.0): ``1:t  2:vel  3:L_bar  4:rho  5:p
    6:T  ...``.  We only use ``t`` (index 0) and ``p`` (index 4).
    """
    path = os.path.join(job_dir, f"history-loc-{idx:04d}.data")
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0], arr[:, 4]


def _load_times_map(job_dir):
    """Load ``times.data`` as a ``tindx -> time`` dict.

    L1d writes ``diaphragm-NNNN.data`` and ``piston-NNNN.data`` without
    embedded timestamps; the times live in a separate ``times.data`` file
    and are joined by tindx.  We need this join to recover t_burst from
    the diaphragm state-flip.
    """
    arr = np.loadtxt(os.path.join(job_dir, "times.data"), comments="#")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return {int(row[0]): float(row[1]) for row in arr}


def parse_l1d_outputs(job_dir, p_burst, transducer_xs, x_buffer, p_band=0.10):
    """Extract ``(t_hold, impact_speed, vs1, ok)`` from L1d outputs.

    Parameters
    ----------
    job_dir : str
        Path to L1d's <job_name>/ subdirectory (NOT the per-individual
        L1d_Outputs/DEAP_<i>/ parent).  This is where the diaphragm,
        slug, piston, and history-loc files live.
    p_burst : float
        Rupture pressure (Pa).  Defines the hold-time band centre.
    transducer_xs : (float, float)
        x-positions of the two shock-tube transducer stations, used as
        the path length for the vs1 time-of-flight calculation.
    x_buffer : float
        The buffer plane in the piston-centre coordinate, as handed to
        L1d.  Take it from ``write_job_script``'s returned ``derived``
        dict; do not re-derive it from the design vector.
    p_band : float
        Fractional half-width of the hold-time band (default +/-10%).

    Returns
    -------
    (t_hold, impact_speed, vs_delta, ok) : tuple
        On success ``ok=True`` and the three numbers are real measurements.
        On any failure the SPARK/PITOT3 sentinel tuple is returned, so
        downstream code (main._detect_sentinels) sees the same encoding.
    """
    parent = os.path.dirname(job_dir)
    piston_file = os.path.join(parent, "piston-0000-history.data")

    # 1. Impact speed: the velocity at the sample immediately before the
    #    piston first reaches the buffer plane.  Piston-history is sampled
    #    at dt_history and the collision is resolved by L1d's internal
    #    sub-stepping between two output samples, so the last pre-contact
    #    sample is the closest available proxy for the impact instant.
    #
    #    Contact is detected by position, not by L1d's on_buffer flag.
    #    That flag latches when the piston comes to rest against the
    #    buffer, which is a later event than first contact: a piston that
    #    strikes and micro-rebounds loses its velocity at the crossing and
    #    only re-contacts, at a fraction of a m/s, some time afterwards.
    #    Keying off the flag reports the settling speed rather than the
    #    collision speed, which for a soft-landing design understates the
    #    impact by orders of magnitude.
    #
    #    Both signals are still consulted and the EARLIER one wins, so the
    #    flag remains a backstop for a piston that grazes the plane
    #    without the position test firing.  X_BUFFER_TOL absorbs the
    #    rounding in the job script's "%.6f" write of x_buffer.
    try:
        piston = np.loadtxt(piston_file, comments="#")
    except OSError:
        return SENTINEL_TUPLE
    if piston.ndim == 1:
        piston = piston[np.newaxis, :]

    reached = np.where(piston[:, 2] >= x_buffer - X_BUFFER_TOL)[0]
    latched = np.where(piston[:, 6] == 1)[0]
    first_evidence = [int(rows[0]) for rows in (reached, latched) if rows.size]
    if not first_evidence:
        return SENTINEL_TUPLE            # piston never reached the buffer
    contact_idx = min(first_evidence)
    if contact_idx == 0:
        # Piston was already on the buffer at t=0 - pathological initial
        # condition, cannot recover an impact speed.
        return SENTINEL_TUPLE
    impact_speed = abs(float(piston[contact_idx - 1, 3]))

    # 2. t_burst from diaphragm state-flip.  L1d writes diaphragm state
    #    as just (tindx, state); the time is in a separate times.data
    #    file and must be joined by tindx.
    #
    #    With L1d's default (instantaneous) diaphragm model the state
    #    sequence is 0 -> 2, never visiting 1.  State 1 only appears in
    #    finite-rupture-time models.  Match on state==2 (fully ruptured).
    diaphragm_file = os.path.join(job_dir, "diaphragm-0000.data")
    try:
        diaphragm = np.loadtxt(diaphragm_file, comments="#")
        times_map = _load_times_map(job_dir)
    except OSError:
        return SENTINEL_TUPLE
    if diaphragm.ndim == 1:
        diaphragm = diaphragm[np.newaxis, :]
    burst_rows = np.where(diaphragm[:, 1] == 2)[0]
    if burst_rows.size == 0:
        return SENTINEL_TUPLE
    burst_tindx = int(diaphragm[burst_rows[0], 0])
    if burst_tindx not in times_map:
        return SENTINEL_TUPLE
    t_burst = times_map[burst_tindx]

    # 3. Hold time: driver-side pressure trace (history-loc-0002).
    #    The algorithm itself lives in problem.t_hold.compute_t_hold and
    #    is shared with the workshop replotter so both stay in sync.
    #       Phase 0  Savitzky-Golay low-pass smoothing
    #       Phase 1  in-band mask over the post-rupture trace
    #       Phase 2  bridge out-of-band excursions shorter than the
    #                hysteresis (default 0.1 ms)
    #       Phase 3  longest surviving contiguous run that starts within
    #                entry_window_s of rupture
    #    Only band_frac is forwarded here; the hysteresis and the optional
    #    asymmetric ceiling (band_frac_hi) take problem.t_hold's module
    #    defaults, which is where to retune them.
    #    t_hold = 0.0 is a legitimate result (design failed to establish
    #    a hold) -- we return it directly rather than mapping to the
    #    sentinel, so CMA-ES sees a continuous fitness landscape.
    try:
        t_drv, p_drv = _load_history_loc(job_dir, idx=2)
    except OSError:
        return SENTINEL_TUPLE
    if not (t_drv >= t_burst).any():
        # No post-burst data at all -- sim integrity failure, not a
        # design failure.  Keep the sentinel here.
        return SENTINEL_TUPLE
    t_hold, _t_start, _t_exit = compute_t_hold(
        t_drv, p_drv, p_burst, t_burst, band_frac=p_band,
    )

    # 4. vs1: shock arrival times at the two shock-tube transducer
    #    stations (history-loc-0000, 0001).  Use a "first-crossing of
    #    2x quiescent fill pressure" detector - the quiescent region is
    #    the first 50 samples (well before burst).
    arrivals = []
    for hist_idx in (0, 1):
        try:
            t_s, p_s = _load_history_loc(job_dir, idx=hist_idx)
        except OSError:
            return SENTINEL_TUPLE
        p_quiescent = max(float(p_s[:50].mean()), 1.0)
        crossings = np.where(p_s > 2.0 * p_quiescent)[0]
        if crossings.size == 0:
            return SENTINEL_TUPLE
        arrivals.append(float(t_s[crossings[0]]))

    dt_arrival = abs(arrivals[1] - arrivals[0])
    if dt_arrival <= 0.0:
        return SENTINEL_TUPLE
    vs1 = abs(transducer_xs[1] - transducer_xs[0]) / dt_arrival

    return t_hold, impact_speed, abs(vs1 - VS1_TARGET), True


# Per-individual subprocess driver

def _stage_gas_models(job_root, source_dir):
    """Symlink the .lua gas-model files into the per-individual job
    directory so l1d4-prep finds them in cwd.

    Raises FileNotFoundError with a clear message if any are missing -
    a typo in the file name will otherwise surface as a confusing
    l1d4-prep traceback.
    """
    for fname in GAS_MODEL_FILES:
        src = Path(source_dir) / fname
        dst = Path(job_root) / fname
        if not src.is_file():
            raise FileNotFoundError(
                f"Gas-model file '{fname}' not found at {src}. "
                f"Run prep-gas to generate it, or update GAS_MODEL_FILES."
            )
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


# Live-data predicates + side-channel watchdog
# The two predicates below are pure file readers - they take only paths to
# files that L1d is updating live, and return small scalars/flags.  Keeping
# them side-effect-free makes them straightforward to unit-test against
# synthetic data without needing a running L1d subprocess.

def _count_piston_inflections(piston_file):
    """Return the number of velocity sign-flips in the live piston file.

    The file layout is the slug-level live dump:
        tindx  x  vel  is_restrain  brakes_on  on_buffer
    (velocity in column index 2 - NOT column 3, which is the layout of
    piston-0000-history.data produced later by --piston-history).

    Sign flips are counted on the non-zero-velocity subsequence: the
    initial v=0 sample and the v~=0 post-buffer-impact rest samples
    would otherwise be counted as their own "sign change" and inflate
    the inflection count.

    Returns 0 if the file is absent or has fewer than two non-zero
    samples - i.e. "not enough information yet, don't trigger".
    """
    try:
        arr = np.loadtxt(piston_file, comments="#")
    except OSError:
        return 0
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] < 2:
        return 0
    s = np.sign(arr[:, 2])
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(s) != 0))


def _piston_velocity_reversed(piston_file):
    """True once the piston velocity has changed sign at least once.

    The piston launches forward (v>0) and the driver gas reaches PEAK
    pressure at the turnaround -- the instant v first crosses zero to
    negative -- since the gas is then most compressed.  So a single sign
    change is the signature that peak compression has come and gone; if
    the diaphragm has not burst by then it never will (later strokes only
    dissipate energy).

    Implemented in terms of _count_piston_inflections so the v=0 launch /
    at-rest filtering (which keeps the initial and post-buffer zeros from
    being miscounted as a sign change) lives in exactly one place.  The
    first inflection IS the first reversal, so ``>= 1`` is the test.

    Returns False when the file is absent or has too few non-zero samples
    -- "not enough information yet, don't trigger".
    """
    return _count_piston_inflections(piston_file) >= 1


def _diaphragm_burst_time(diaphragm_file, times_file):
    """Sim time at which the diaphragm first reached state==2, or None.

    Returns None if either file is unreadable, if no state==2 row has
    been written yet, or if the matching tindx isn't in times.data yet
    (a transient mid-write inconsistency - we just wait for the next
    poll cycle).
    """
    try:
        dia = np.loadtxt(diaphragm_file, comments="#")
        times = np.loadtxt(times_file, comments="#")
    except OSError:
        return None
    if dia.ndim == 1:
        dia = dia[np.newaxis, :]
    if times.ndim == 1:
        times = times[np.newaxis, :]
    if dia.size == 0 or times.size == 0:
        return None
    burst_rows = np.where(dia[:, 1] == 2)[0]
    if burst_rows.size == 0:
        return None
    burst_tindx = int(dia[burst_rows[0], 0])
    match = np.where(times[:, 0] == burst_tindx)[0]
    if match.size == 0:
        return None
    return float(times[match[0], 1])


def _current_sim_time(times_file):
    """Largest sim time written so far, or None if the file is empty."""
    try:
        arr = np.loadtxt(times_file, comments="#")
    except OSError:
        return None
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.size == 0:
        return None
    return float(arr[-1, 1])


def _run_l1d_with_watchdog(cmd, *, job_inner, timeout, capture):
    """Drop-in replacement for ``subprocess.run`` on the L1d run-simulation
    step, with a side-channel watchdog that may SIGTERM the child early.

    The child is launched with ``Popen``; the parent ``wait``s with a
    timeout equal to WATCHDOG_POLL_S so we get immediate wake-up the
    moment the child exits naturally, and otherwise fall through to
    predicate evaluation every WATCHDOG_POLL_S seconds.

    Outcomes
    --------
    'completed'       child exited on its own with rc 0.
    'aborted_success' diaphragm has burst and (sim_t - t_burst) >=
                      GRACE_SIM_TIME_S; we SIGTERMed.  Output files are
                      complete enough for parse_l1d_outputs.
    'aborted_fail'    no rupture and the piston has either reversed
                      direction (FAIL_ON_PISTON_REVERSAL) or inflected
                      >= MAX_INFLECTIONS_NO_RUPTURE times; we SIGTERMed.
                      Caller should return SENTINEL without parsing.

    Errors are re-raised in the same shapes ``subprocess.run`` would
    have used, so the existing except clauses in ``run_l1d`` catch
    them unchanged:
      - non-zero exit       -> subprocess.CalledProcessError
      - wall-clock exhausted -> subprocess.TimeoutExpired
    """
    piston_file    = os.path.join(job_inner, "piston-0000.data")
    diaphragm_file = os.path.join(job_inner, "diaphragm-0000.data")
    times_file     = os.path.join(job_inner, "times.data")

    # When capturing, route stdout/stderr to spooled tempfiles rather
    # than subprocess.PIPE.  PIPE has a ~64 kB kernel buffer; if neither
    # parent thread is reading it, a chatty L1d run will eventually
    # block on its own stdout write and deadlock.  Tempfiles have no
    # such limit.  Inheriting parent fds (capture=False) has no buffer
    # issue.
    out_buf = (tempfile.SpooledTemporaryFile(max_size=4 * 1024 * 1024)
               if capture else None)
    err_buf = (tempfile.SpooledTemporaryFile(max_size=4 * 1024 * 1024)
               if capture else None)

    proc = subprocess.Popen(cmd, stdout=out_buf, stderr=err_buf)
    deadline = time.monotonic() + timeout

    def _stderr_tail():
        if err_buf is None:
            return None
        err_buf.seek(0)
        data = err_buf.read()
        return data[-2000:] if data else b""

    def _terminate(reason):
        try:
            proc.terminate()
            proc.wait(timeout=WATCHDOG_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if STREAM_L1D_OUTPUT:
            print(f"L1d watchdog: terminated child ({reason})",
                  file=sys.stderr)

    try:
        while True:
            poll_window = min(
                WATCHDOG_POLL_S,
                max(deadline - time.monotonic(), 0.0),
            )
            if poll_window <= 0.0:
                _terminate("eval timeout")
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                rc = proc.wait(timeout=poll_window)
            except subprocess.TimeoutExpired:
                pass   # child still running - evaluate predicates
            else:
                if rc != 0:
                    raise subprocess.CalledProcessError(
                        rc, cmd, stderr=_stderr_tail(),
                    )
                return 'completed'

            # predicate evaluation
            burst_t = _diaphragm_burst_time(diaphragm_file, times_file)
            sim_t   = _current_sim_time(times_file)

            if burst_t is not None and sim_t is not None:
                # Success path - wait the grace window past rupture.
                if (sim_t - burst_t) >= GRACE_SIM_TIME_S:
                    _terminate(
                        f"rupture at {burst_t*1e3:.3f} ms + "
                        f"{GRACE_SIM_TIME_S*1e3:.1f} ms grace elapsed"
                    )
                    return 'aborted_success'
            else:
                # Fail-fast - no rupture yet.  Two gates, tightest first.
                n_inf = _count_piston_inflections(piston_file)
                # (a) reversal gate: the piston has turned around at least
                #     once, so peak driver pressure has been reached and
                #     passed without rupture - it can never rupture now.
                if FAIL_ON_PISTON_REVERSAL and n_inf >= 1:
                    _terminate(
                        "piston velocity reversed (turnaround) before rupture"
                    )
                    return 'aborted_fail'
                # (b) legacy inflection gate: subsumed by (a) when enabled,
                #     retained as a fallback when the reversal gate is off.
                if n_inf >= MAX_INFLECTIONS_NO_RUPTURE:
                    _terminate(
                        f"{n_inf} piston inflections, no rupture"
                    )
                    return 'aborted_fail'
    finally:
        # Belt-and-braces: if we exit via an exception, never leak the
        # child.  poll() returns None iff the process is still running.
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if out_buf is not None:
            out_buf.close()
        if err_buf is not None:
            err_buf.close()


def run_l1d(x_phys, ind_number, *, test_gas_p1, transducer_xs=TRANSDUCER_XS,
            mesh_scale=MESH_SCALE_FACTOR, gas_model_dir=None,
            output_root=None):
    """Run one L1d evaluation end to end.

    Parameters
    ----------
    x_phys : sequence of 6 floats
        Physical-space design vector ``[percent_He, driver_p, p4,
        D_throat, reservoir_p, buffer_length]``.
    ind_number : int
        Population index, used to name the per-individual work directory.
    test_gas_p1 : float
        Shock-tube fill pressure (Pa).  Passed in by the caller so it
        stays single-source (read from problem.config in evaluate.py).
    transducer_xs : (float, float)
        Shock-tube transducer x-positions (m).
    mesh_scale : int
        Per-slug ncells multiplier.
    gas_model_dir : str or None
        Directory holding the .lua gas-model files.  Defaults to the
        project root.
    output_root : str or None
        Parent directory under which L1d_Outputs/DEAP_<i>/ is created.
        Defaults to the project src/ directory.

    Returns
    -------
    (t_hold, impact_speed, vs_delta, ok) : tuple
        On success ok=True; on any failure mode the SPARK/PITOT3 sentinel
        tuple is returned so the downstream pipeline sees the existing
        encoding.
    """
    percent_He, driver_p, p4, D_throat, reservoir_p, buffer_length = x_phys
    params = {
        "percent_He":    percent_He,
        "driver_p":      driver_p,
        "p4":            p4,
        "D_throat":      D_throat,
        "reservoir_p":   reservoir_p,
        "buffer_length": buffer_length,
        "test_gas_p1":   test_gas_p1,
        "mesh_scale":    mesh_scale,
        "transducer_xs": tuple(transducer_xs),
    }

    if gas_model_dir is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        gas_model_dir = os.path.join(repo_root, "gas_models")
    if output_root is None:
        output_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    job_name = f"DEAP_{ind_number}"
    job_root = os.path.join(output_root, "L1d_Outputs", job_name)
    job_inner = os.path.join(job_root, job_name)
    os.makedirs(job_root, exist_ok=True)

    success = False
    try:
        script_path = os.path.join(job_root, f"{job_name}.py")
        try:
            derived = write_job_script(script_path, params, ind_number)
        except ValueError as e:
            # build_break_points raises on infeasible orifice geometry.
            print(f"L1d geometry infeasible for ind {ind_number}: {e}",
                  file=sys.stderr)
            return SENTINEL_TUPLE

        _stage_gas_models(job_root, gas_model_dir)

        # When STREAM_L1D_OUTPUT is True the subprocesses inherit the
        # parent's stdout/stderr so progress is visible live in the
        # terminal; when False, both streams are captured so the error
        # path below can print the last 2000 chars of stderr on failure.
        capture = not STREAM_L1D_OUTPUT
        cwd = os.getcwd()
        os.chdir(job_root)
        try:
            subprocess.run(
                ["l1d4-prep", f"--job={job_name}"],
                check=True, timeout=PREP_TIMEOUT_S, capture_output=capture,
            )
            outcome = _run_l1d_with_watchdog(
                ["l1d4", "--run-simulation", f"--job={job_name}"],
                job_inner=job_inner,
                timeout=EVAL_TIMEOUT_S,
                capture=capture,
            )
            if outcome == 'aborted_fail':
                # No rupture occurred and the piston has stopped doing
                # useful work.  Skip the piston-history postprocess and
                # the parser - we already know the answer.  success
                # stays False, so the outer finally cleans up the dir
                # (unless KEEP_ALL_JOBS / KEEP_FAILED_JOBS override).
                return SENTINEL_TUPLE
            subprocess.run(
                ["l1d4", "--piston-history",
                 f"--job={job_name}", "--pindx=0"],
                check=True, timeout=POSTPROCESS_TIMEOUT_S,
                capture_output=capture,
            )
        finally:
            os.chdir(cwd)

        result = parse_l1d_outputs(
            job_dir=job_inner,
            p_burst=p4,
            transducer_xs=transducer_xs,
            x_buffer=derived["x_buffer"],
        )
        success = result[3]
        return result

    except subprocess.CalledProcessError as e:
        print(f"L1d subprocess failed for ind {ind_number}: "
              f"cmd={e.cmd} rc={e.returncode}", file=sys.stderr)
        # When STREAM_L1D_OUTPUT=True the subprocess inherits stderr, so
        # e.stderr is None and the user has already seen the error text
        # in the terminal.  Only re-emit captured stderr when we have it.
        if e.stderr:
            print(e.stderr.decode(errors="replace")[-2000:], file=sys.stderr)
        return SENTINEL_TUPLE
    except subprocess.TimeoutExpired as e:
        print(f"L1d timeout for ind {ind_number}: cmd={e.cmd}",
              file=sys.stderr)
        return SENTINEL_TUPLE
    except Exception as e:
        print(f"L1d unexpected failure for ind {ind_number}: {e}",
              file=sys.stderr)
        return SENTINEL_TUPLE
    finally:
        keep = (KEEP_ALL_JOBS
                or (success and KEEP_SUCCESSFUL_JOBS)
                or ((not success) and KEEP_FAILED_JOBS))
        if not keep:
            shutil.rmtree(job_root, ignore_errors=True)
