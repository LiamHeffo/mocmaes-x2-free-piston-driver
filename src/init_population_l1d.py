"""
Quasi-random initialisation of an all-non-sentinel population (L1d4 path).

Builds an initial population of ``pop_size`` X2-driver designs in which
every individual yields a real L1d4 evaluation. A sentinel is the failure
encoding the rest of the pipeline uses: the heuristic pre-check rejecting
the design, or run_l1d returning ``ok=False`` for no rupture, a timeout or
unparseable output.

Sampling (--method) defaults to ``sobol``: draw from a scrambled Sobol
sequence, evaluate, keep the non-sentinel points and draw more from the
same sequence to refill. A Sobol sequence is built to be extended, so each
refill explores new regions of the box rather than jittering around the
point it replaces. ``lhs`` is a legacy Latin-hypercube path, kept for
reference.

Evaluations run as processes rather than threads because run_l1d calls
os.chdir(), which mutates process-global state; separate processes each
get their own working directory.

Writes an ``.npz`` holding the accepted designs in physical and normalised
[1,2]^6 space, their raw objective triples, and the per-sector attempt
counts, then prints a summary. The normalised vectors are what main.py
seeds CMA-ES with.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import qmc

# Make `problem` importable whether this is run from src/ or the repo root.
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from problem import l1d_job
from problem.l1d_job import run_l1d, TRANSDUCER_XS, MESH_SCALE_FACTOR
from problem.evaluate import _heuristic_passes, _PITOT3_FAILURE_SENTINEL
from problem.config import (
    BOUNDS, base_config_dict,
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.transforms import variable_transformation

N_DIMS = 6


# LHS construction and the per-sector unit->physical mapping

def build_lhs_strata(n, rng, d=N_DIMS):
    """Return the (n, d) integer stratum matrix of a Latin Hypercube.

    Column ``d`` is an independent random permutation of ``range(n)``;
    row ``i`` is the "sector" of sample ``i`` - its stratum index on
    each axis.  Holding a row fixed while redrawing the offset ξ is what
    keeps the one-point-per-stratum LHS guarantee intact across
    resamples.
    """
    return np.column_stack([rng.permutation(n) for _ in range(d)])


def lhs_unit_to_physical(u):
    """Map a latent-cube row ``u ∈ [0,1]^6`` to a PHYSICAL design vector.

    Mirrors the conditional-bound semantics of the validated
    problem.sampling.lhs_sample: p4 and reservoir_p are stratified in the
    latent cube but mapped through bounds that depend on the just-drawn
    driver_p, so the heuristic relations (reservoir_p >= driver_p,
    p4 <= p4_upper) hold by construction.

    Column order: [percent_He, driver_p, p4, D_throat, reservoir_p,
    buffer_length].
    """
    he      = he_lower            + u[0] * (he_upper            - he_lower)
    drv_p   = driver_p_lower      + u[1] * (driver_p_upper      - driver_p_lower)
    Dthroat = D_throat_lower      + u[3] * (D_throat_upper      - D_throat_lower)
    buf_len = buffer_length_lower + u[5] * (buffer_length_upper - buffer_length_lower)

    p4_hi = min(1190.63 * drv_p, p4_upper)
    p4_lo = 14.62 * drv_p
    p4    = p4_lo + u[2] * (p4_hi - p4_lo)

    res_p = drv_p + u[4] * (reservoir_upper - drv_p)

    return [he, drv_p, p4, Dthroat, res_p, buf_len]


# Worker - runs in a child process (own cwd, own subprocess tree)

def _init_worker():
    """Pool initializer for the per-evaluation child processes.

    Two pieces of per-worker state:

    * Quiet the streaming so the parallel workers don't interleave L1d
      stdout into an unreadable mess (see STREAM_L1D_OUTPUT in
      problem/l1d_job.py).

    * Override the job-retention flags so the sweep is disk-safe.  An
      initialisation sweep evaluates HUNDREDS of designs to accept a
      handful, and at production mesh the ~95% no-rupture jobs each run
      the full t_finish and write the largest output files.  The module
      default KEEP_ALL_JOBS=True would retain every one of them and fill
      the disk mid-run (OSError 28).  We instead delete rejected jobs the
      instant they finish and keep only the ruptured ones; the sweep's
      own keep_jobs flag then decides whether even those survive at the
      end (it rmtrees the whole scratch root when keep_jobs is False)."""
    l1d_job.STREAM_L1D_OUTPUT = False
    l1d_job.KEEP_ALL_JOBS = False
    l1d_job.KEEP_FAILED_JOBS = False
    l1d_job.KEEP_SUCCESSFUL_JOBS = True


def evaluate_individual(x_phys, job_tag, test_gas_p1, transducer_xs,
                        mesh_scale, output_root):
    """Evaluate one physical design and report whether it is non-sentinel.

    ``job_tag`` is a UNIQUE-per-attempt identifier (e.g. ``init_s003_t1``):
    run_l1d names the job directory ``<output_root>/L1d_Outputs/DEAP_<tag>``,
    so reusing a population slot across resamples never writes into - or
    reads stale output files from - a previous attempt's directory.
    ``output_root`` is a dedicated scratch root that keeps these init jobs
    isolated from the optimiser's own DEAP_* directories.

    Returns ``((t_hold, impact, delta_vs), ok)``.  ``ok`` is False when
    the design is a sentinel - either the heuristic pre-check rejects it
    (mirrors problem.evaluate._evaluate_l1d) or run_l1d reports ok=False.
    """
    driver_dict = {
        'percent_He':    x_phys[0],
        'driver_p':      x_phys[1],
        'p4':            x_phys[2],
        'D_throat':      x_phys[3],
        'reservoir_p':   x_phys[4],
        'buffer_length': x_phys[5],
    }
    if not _heuristic_passes(driver_dict):
        return (0.0, 350.0, float(_PITOT3_FAILURE_SENTINEL)), False

    t_hold, impact, delta_vs, ok = run_l1d(
        x_phys=x_phys,
        ind_number=job_tag,
        test_gas_p1=test_gas_p1,
        transducer_xs=transducer_xs,
        mesh_scale=mesh_scale,
        output_root=output_root,
    )
    return (t_hold, impact, delta_vs), bool(ok)


# Driver

def initialise_population(pop_size=12, *, seed=None, max_workers=None,
                          max_attempts=50, mesh_scale=MESH_SCALE_FACTOR,
                          transducer_xs=TRANSDUCER_XS, scratch_dir=None,
                          keep_jobs=True, verbose=True):
    """Build a ``pop_size`` population in which every individual is
    non-sentinel under the L1d4 evaluation.

    Parameters
    ----------
    scratch_dir : str or None
        Dedicated root for the per-attempt L1d job directories.  Each
        evaluation gets its own ``<scratch_dir>/L1d_Outputs/DEAP_<tag>``
        tree, so resamples never reuse (and so never read stale output
        from) an earlier attempt's directory, and the whole subtree is
        trivially removable.  Defaults to ``<src>/L1d_init_scratch``.
    keep_jobs : bool
        If False, the entire ``scratch_dir`` is deleted once the
        population is built (the .npz holds the results).  Default True
        so the accepted individuals' L1d outputs survive for inspection.

    Returns
    -------
    dict with keys:
        x_phys   : (pop_size, 6) physical design vectors
        x_norm   : (pop_size, 6) normalised [1,2]^6 vectors (CMA-ES seed)
        objectives : (pop_size, 3) raw (t_hold, impact, delta_vs)
        attempts : (pop_size,) resamples needed per sector (1 = accepted
                   on the first draw)
    """
    rng = np.random.default_rng(seed)
    if max_workers is None:
        max_workers = min(pop_size, os.cpu_count() or 1)
    if scratch_dir is None:
        scratch_dir = os.path.join(_SRC, "L1d_init_scratch")

    test_gas_p1 = float(base_config_dict()['p1'])

    strata = build_lhs_strata(pop_size, rng)          # (n, 6) ints, fixed
    xi = rng.random((pop_size, N_DIMS))               # within-stratum offsets

    objectives = [None] * pop_size
    attempts = np.zeros(pop_size, dtype=int)
    pending = list(range(pop_size))
    round_idx = 0

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker) as ex:
        while pending:
            round_idx += 1
            u = (strata + xi) / pop_size
            phys = {i: lhs_unit_to_physical(u[i]) for i in pending}

            if verbose:
                print(f"[round {round_idx}] dispatching {len(pending)} "
                      f"L1d evaluation(s) on {max_workers} worker(s): "
                      f"sectors {pending}", flush=True)
            t0 = time.monotonic()

            futures = {
                # Unique per-attempt tag: slot index + attempts-so-far.
                # attempts[i] is still the PRE-increment count here, so
                # each (sector, attempt) pair maps to its own directory.
                ex.submit(evaluate_individual, phys[i],
                          f"init_s{i:03d}_t{int(attempts[i])}", test_gas_p1,
                          transducer_xs, mesh_scale, scratch_dir): i
                for i in pending
            }
            round_results = {}
            for fut in as_completed(futures):
                i = futures[fut]
                round_results[i] = fut.result()

            # Process the round in deterministic (index) order so the RNG
            # consumption - and therefore reproducibility under a fixed
            # seed - does NOT depend on the nondeterministic order in
            # which the parallel workers happen to finish.
            still_pending = []
            for i in pending:
                obj, ok = round_results[i]
                attempts[i] += 1
                if ok:
                    objectives[i] = obj
                else:
                    if attempts[i] >= max_attempts:
                        raise RuntimeError(
                            f"sector {i} still sentinel after "
                            f"{max_attempts} resamples; widen bounds, "
                            f"raise max_attempts, or inspect the design."
                        )
                    xi[i] = rng.random(N_DIMS)   # resample THIS sector only
                    still_pending.append(i)

            if verbose:
                n_ok = len(pending) - len(still_pending)
                print(f"[round {round_idx}] {n_ok}/{len(pending)} "
                      f"non-sentinel in {time.monotonic() - t0:.1f}s; "
                      f"{len(still_pending)} to resample", flush=True)
            pending = still_pending

    if not keep_jobs:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        if verbose:
            print(f"removed scratch dir {scratch_dir}", flush=True)

    u_final = (strata + xi) / pop_size
    x_phys = np.array([lhs_unit_to_physical(u_final[i])
                       for i in range(pop_size)])
    x_norm = np.array(variable_transformation(x_phys.tolist(), BOUNDS))
    return {
        "x_phys": x_phys,
        "x_norm": x_norm,
        "objectives": np.array(objectives, dtype=float),
        "attempts": attempts,
    }


# Sobol (scrambled) + rejection - "draw more from the sequence" resampling

def _next_pow2(n):
    """Smallest power of two >= n (Sobol balance is exact only at 2^m)."""
    return 1 << max(0, (int(n) - 1)).bit_length()


def initialise_population_sobol(pop_size=12, *, seed=None, max_workers=None,
                                oversample=4, max_eval_factor=64,
                                floor_accept=0.05, mesh_scale=MESH_SCALE_FACTOR,
                                transducer_xs=TRANSDUCER_XS, scratch_dir=None,
                                keep_jobs=True, verbose=True):
    """Build a ``pop_size`` all-non-sentinel population via a scrambled
    Sobol sequence with rejection - drawing MORE points from the same
    sequence whenever the accepted set falls short.

    A Sobol sequence is low-discrepancy and designed to be extended: the
    first N points are well spread for any N, and appending more keeps
    them so. Rejecting sentinels and drawing further points is therefore
    sound rather than a workaround.

    Each unit point u in [0,1]^6 is mapped to a physical design with the
    same conditional bound semantics as ``lhs_unit_to_physical``, so
    reservoir_p >= driver_p and the compression-ratio and p4 heuristics
    hold by construction. Every sentinel is then a genuine L1d failure
    rather than a pre-check rejection.

    Adaptive batch sizing
    ---------------------
    The acceptance (= feasible) rate is unknown a priori, so the first
    round draws ``oversample * pop_size`` points (rounded up to a power of
    two for Sobol balance).  Thereafter the empirical acceptance rate
    ``p̂`` from all evaluations so far sizes the next draw at roughly
    ``need / p̂`` (with ``p̂`` floored at ``floor_accept`` so a run of bad
    luck can't request an enormous batch).  A hard ceiling of
    ``max_eval_factor * pop_size`` total evaluations aborts with a
    diagnostic - that is the signal to escalate to a feasibility surrogate.

    Returns the same dict shape as :func:`initialise_population` plus
    ``n_evaluated`` and ``acceptance_rate`` diagnostics.  ``attempts`` is
    all-ones (each accepted individual is evaluated exactly once); the
    resample cost lives in ``n_evaluated`` instead.
    """
    if max_workers is None:
        max_workers = min(pop_size, os.cpu_count() or 1)
    if scratch_dir is None:
        scratch_dir = os.path.join(_SRC, "L1d_init_scratch")

    test_gas_p1 = float(base_config_dict()['p1'])
    sampler = qmc.Sobol(d=N_DIMS, scramble=True, seed=seed)
    max_eval = max_eval_factor * pop_size

    accepted_u = []          # accepted unit rows, in draw order (reproducible)
    accepted_obj = []        # matching (t_hold, impact, delta_vs)
    n_eval = 0
    round_idx = 0

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker) as ex:
        while len(accepted_u) < pop_size:
            round_idx += 1
            need = pop_size - len(accepted_u)

            # Size this round's draw.  First round: oversample blindly.
            # Later rounds: use the observed acceptance rate to aim for
            # `need` more, never below max_workers (keep the pool busy) or
            # `need` (don't under-draw), and rounded up to a power of two.
            if n_eval == 0:
                batch = oversample * pop_size
            else:
                p_hat = max(len(accepted_u) / n_eval, floor_accept)
                batch = math.ceil(need / p_hat)
            batch = _next_pow2(max(batch, need, max_workers))

            if n_eval + batch > max_eval:
                batch = max_eval - n_eval
            if batch <= 0:
                raise RuntimeError(
                    f"Sobol init hit the evaluation ceiling "
                    f"({max_eval} evals) with only {len(accepted_u)}/"
                    f"{pop_size} non-sentinel designs "
                    f"(acceptance rate {len(accepted_u)/max(n_eval,1):.1%}). "
                    f"The feasible region is too sparse for blind rejection "
                    f"- escalate to a feasibility surrogate or widen/condition "
                    f"the bounds."
                )

            # Sobol balance is exact only for power-of-two block sizes; we
            # keep `batch` a power of two but silence scipy's advisory
            # warning in case the ceiling clamp above trims it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U = sampler.random(batch)            # (batch, 6) in [0,1)

            if verbose:
                print(f"[round {round_idx}] need {need} more; drawing "
                      f"{batch} Sobol point(s) on {max_workers} worker(s) "
                      f"(evals so far: {n_eval})", flush=True)
            t0 = time.monotonic()

            phys = [lhs_unit_to_physical(U[j]) for j in range(batch)]
            futures = {
                ex.submit(evaluate_individual, phys[j],
                          f"sobol_{n_eval + j:05d}", test_gas_p1,
                          transducer_xs, mesh_scale, scratch_dir): j
                for j in range(batch)
            }
            results = {}
            for fut in as_completed(futures):
                j = futures[fut]
                results[j] = fut.result()

            # Walk the batch in draw order so acceptance is deterministic
            # and the kept set is the FIRST pop_size feasible Sobol points.
            n_ok_round = 0
            for j in range(batch):
                obj, ok = results[j]
                if ok:
                    n_ok_round += 1
                    if len(accepted_u) < pop_size:
                        accepted_u.append(U[j].copy())
                        accepted_obj.append(obj)
            n_eval += batch

            if verbose:
                print(f"[round {round_idx}] {n_ok_round}/{batch} "
                      f"non-sentinel in {time.monotonic() - t0:.1f}s; "
                      f"accepted {len(accepted_u)}/{pop_size} "
                      f"(cumulative acceptance {len(accepted_u)/n_eval:.1%})",
                      flush=True)

    if not keep_jobs:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        if verbose:
            print(f"removed scratch dir {scratch_dir}", flush=True)

    u_final = np.array(accepted_u[:pop_size])
    x_phys = np.array([lhs_unit_to_physical(u_final[i])
                       for i in range(pop_size)])
    x_norm = np.array(variable_transformation(x_phys.tolist(), BOUNDS))
    return {
        "x_phys": x_phys,
        "x_norm": x_norm,
        "objectives": np.array(accepted_obj[:pop_size], dtype=float),
        "attempts": np.ones(pop_size, dtype=int),
        "n_evaluated": n_eval,
        "acceptance_rate": pop_size / n_eval,
    }


def _print_summary(result):
    x_phys = result["x_phys"]
    obj = result["objectives"]
    att = result["attempts"]
    print("\nAccepted non-sentinel population")
    print("-" * 78)
    print(f"{'i':>2} {'%He':>6} {'driver_p':>11} {'p4':>11} "
          f"{'D_thr':>6} {'res_p':>11} {'buf':>6} | "
          f"{'t_hold':>8} {'impact':>7} {'dvs':>7} {'try':>3}")
    for i in range(len(x_phys)):
        he, dp, p4, dt, rp, bf = x_phys[i]
        th, im, dv = obj[i]
        print(f"{i:>2} {he:6.2f} {dp:11.3e} {p4:11.3e} {dt:6.4f} "
              f"{rp:11.3e} {bf:6.4f} | {th:8.5f} {im:7.1f} {dv:7.1f} "
              f"{att[i]:>3}")
    print("-" * 78)
    if "n_evaluated" in result:
        # Sobol path: the resample cost lives in n_evaluated, not attempts.
        n_eval = int(result["n_evaluated"])
        print(f"total evaluations: {n_eval} for {len(x_phys)} accepted "
              f"individuals (acceptance rate {result['acceptance_rate']:.1%}, "
              f"rejection overhead {n_eval - len(x_phys)})")
    else:
        print(f"total L1d evaluations: {int(att.sum())} "
              f"for {len(x_phys)} accepted individuals "
              f"(resample overhead {int(att.sum()) - len(x_phys)})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("lhs", "sobol"), default="sobol",
                   help="sampler: 'sobol' (scrambled Sobol + draw-more "
                        "rejection, default) or 'lhs' (legacy per-sector "
                        "within-stratum resampling)")
    p.add_argument("--pop-size", type=int, default=12,
                   help="population size (default 12)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible sampling + resampling")
    p.add_argument("--max-workers", type=int, default=None,
                   help="parallel L1d processes (default min(pop_size, ncpu))")
    p.add_argument("--max-attempts", type=int, default=50,
                   help="[lhs] give up on a sector after this many resamples")
    p.add_argument("--oversample", type=int, default=4,
                   help="[sobol] first-round draw = oversample*pop_size "
                        "(rounded up to a power of two); later rounds size "
                        "adaptively from the observed acceptance rate")
    p.add_argument("--max-eval-factor", type=int, default=64,
                   help="[sobol] abort after max_eval_factor*pop_size total "
                        "evaluations (signal to escalate to a surrogate)")
    p.add_argument("--mesh-scale", type=int, default=MESH_SCALE_FACTOR,
                   help="per-slug ncells multiplier passed to run_l1d")
    p.add_argument("--scratch-dir", type=str, default=None,
                   help="root for per-attempt L1d job dirs "
                        "(default <src>/L1d_init_scratch)")
    p.add_argument("--no-keep-jobs", dest="keep_jobs", action="store_false",
                   help="delete the scratch dir after building the population "
                        "(default: keep it for inspection)")
    p.add_argument("--out", type=str,
                   default=os.path.join(_SRC, "L1d_Outputs",
                                        "init_population.npz"),
                   help="output .npz path")
    p.add_argument("--launch-cmaes", action="store_true",
                   help="after building the population, launch a CMA-ES run "
                        "(src/main.py --seed-npz <out>) seeded with x_norm. "
                        "Ensure config/experiments.yaml's pop_size matches "
                        "--pop-size.")
    args = p.parse_args(argv)

    t_start = time.monotonic()
    if args.method == "sobol":
        result = initialise_population_sobol(
            pop_size=args.pop_size,
            seed=args.seed,
            max_workers=args.max_workers,
            oversample=args.oversample,
            max_eval_factor=args.max_eval_factor,
            mesh_scale=args.mesh_scale,
            scratch_dir=args.scratch_dir,
            keep_jobs=args.keep_jobs,
        )
    else:
        result = initialise_population(
            pop_size=args.pop_size,
            seed=args.seed,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
            mesh_scale=args.mesh_scale,
            scratch_dir=args.scratch_dir,
            keep_jobs=args.keep_jobs,
        )
    _print_summary(result)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        x_phys=result["x_phys"],
        x_norm=result["x_norm"],
        objectives=result["objectives"],
        attempts=result["attempts"],
        n_evaluated=np.array(result.get("n_evaluated", int(result["attempts"].sum()))),
        acceptance_rate=np.array(result.get("acceptance_rate",
                                            len(result["x_phys"]) / max(int(result["attempts"].sum()), 1))),
        method=np.array(args.method),
        seed=np.array(args.seed if args.seed is not None else -1),
    )
    print(f"\nsaved -> {args.out}")
    print(f"wall clock: {time.monotonic() - t_start:.1f}s")

    # Optionally pipe the accepted population straight into a CMA-ES run.
    # main.py reads the .npz's ``x_norm`` (normalised [1,2]^6 designs) and
    # seeds every experiment in config/experiments.yaml with it - so make
    # sure that config's pop_size matches --pop-size here (e.g. the
    # ArnoldCHT_AL block).  Runs in main.py's dispatcher mode (one fresh
    # subprocess per experiment).
    if args.launch_cmaes:
        import subprocess
        main_py = os.path.join(_SRC, "main.py")
        cmd = [sys.executable, main_py, "--seed-npz", os.path.abspath(args.out)]
        print(f"\nlaunching seeded CMA-ES run:\n  {' '.join(cmd)}\n", flush=True)
        # Inherit the environment (gdtk on PYTHONPATH); cwd at repo root so
        # main.py resolves config/experiments.yaml relative to its parent.
        raise SystemExit(subprocess.run(cmd, cwd=os.path.dirname(_SRC)).returncode)


if __name__ == "__main__":
    main()
