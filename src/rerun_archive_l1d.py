#!/usr/bin/env python3
"""Re-run the external non-dominated archive of one optimisation run in L1d4.

Purpose
-------
An optimisation run persists its archive as *normalised* objective values
(``summary/archive.csv``: ``f_0``/``f_1`` in [0, 1]) plus the normalised
design vector.  For the paper we want the archive members re-simulated
with the production L1d setup so the reported hold times and impact
speeds are traceable to a simulation we can point at on disk, not just to
an inverted normalisation.

Parity with the optimiser
-------------------------
This script calls ``problem.l1d_job.run_l1d`` - the exact function
``problem.evaluate._evaluate_l1d`` calls - and deliberately passes NO
``mesh_scale`` or ``transducer_xs`` override, so both take the module
defaults the optimisation itself used:

    MESH_SCALE_FACTOR = 1     per-slug ncells multiplier (39/46/34 x 1)
    TRANSDUCER_XS     = (1.5, 2.5) m
    test_gas_p1       = problem.config.base_config_dict()['p1']

Reproducing the setup by *reusing the call site* rather than restating its
arguments is what makes the parity claim hold even if those constants are
retuned later.

The heuristic pre-check (``_heuristic_passes``) is applied first, exactly
as in the optimiser: an archive member that fails it would have been a
sentinel during the run, and we want to see that rather than paper over it.

Evaluations run as processes rather than threads: ``run_l1d`` calls
``os.chdir()``, which mutates process-global state, so each needs its own
working directory.

Output
------
``<out-dir>/archive_rerun.csv`` - one row per archive member with the
design in physical units, the re-simulated ``(t_hold, impact_speed,
delta_vs1)``, and the optimisation's own recorded values (un-normalised
with the optimiser's ideal/nadir points) for side-by-side comparison.
The per-individual L1d job directories are kept under
``<out-dir>/L1d_Outputs/``, so the raw pressure and piston traces stay
available for inspection.

Usage
-----
    python3 src/rerun_archive_l1d.py --run al_cht_0122
    python3 src/rerun_archive_l1d.py --run al_cht_0122 --max-workers 6

Do NOT invoke with ``PYTHONPATH=src`` - that clobbers gdtk and every
l1d4-prep subprocess dies.  This script puts ``src`` on ``sys.path`` itself.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from problem import l1d_job                                    # noqa: E402
from problem.l1d_job import (                                  # noqa: E402
    run_l1d, parse_l1d_outputs, TRANSDUCER_XS,
)
from problem.config import (                                   # noqa: E402
    BOUNDS, base_config_dict, APPROX_IDEAL_2D, APPROX_NADIR_2D,
)
from problem.evaluate import _heuristic_passes, _PITOT3_FAILURE_SENTINEL  # noqa: E402
from problem.transforms import variable_untransformation, unnormalise_fitness  # noqa: E402

PROJECT_ROOT = Path(_SRC).parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "al_cht_recomb"

DESIGN_NAMES = ["percent_He", "driver_p", "p4", "D_throat",
                "reservoir_p", "buffer_length"]


# Worker - one L1d evaluation in its own process

def _init_worker(grace_sim_time_s=None):
    """Pool initialiser: quiet the L1d stdout, keep job dirs, set the grace.

    ``STREAM_L1D_OUTPUT = False`` stops N parallel L1d processes from
    interleaving their progress lines into an unreadable stream (and lets
    run_l1d capture stderr for the failure path).  ``KEEP_ALL_JOBS`` is
    left True - unlike an init sweep this is only 14 evaluations, and we
    want the traces.

    ``grace_sim_time_s`` overrides ``l1d_job.GRACE_SIM_TIME_S``, the
    watchdog's SUCCESS-AND-DONE budget: how much SIMULATED time the run is
    allowed past diaphragm rupture before being killed early.  This is a
    wall-clock economy measure, NOT a physics parameter - the mesh, gas
    models, loss regions and time-stepping are unaffected, the simulation
    is simply allowed to advance further.

    It has to be raised for designs whose piston reaches the buffer more
    than the default 5 ms after rupture: the run is otherwise terminated
    before contact, ``on_buffer`` never flips, and parse_l1d_outputs
    returns the failure sentinel for want of an impact speed.  Because the
    watchdog polls only every WATCHDOG_POLL_S (60 s) wall-clock seconds,
    how far a run overshoots the grace depends on machine load - so near
    the boundary the evaluation is not reproducible.  Setting the grace to
    T_FINISH removes success-based early termination altogether and makes
    the result load-independent.

    The FAIL-FAST gates are untouched: they fire only for runs that never
    rupture, so they cannot truncate a design that got this far.
    """
    l1d_job.STREAM_L1D_OUTPUT = False
    l1d_job.KEEP_ALL_JOBS = True
    if grace_sim_time_s is not None:
        l1d_job.GRACE_SIM_TIME_S = float(grace_sim_time_s)


def evaluate_member(x_phys, job_tag, output_root):
    """Evaluate one physical design; return ``(t_hold, impact, delta_vs, ok)``.

    Mirrors ``problem.evaluate._evaluate_l1d`` including the cheap
    heuristic pre-check, so a member that the optimiser would have
    sentinelled is reported as a sentinel here too.
    """
    driver_dict = dict(zip(DESIGN_NAMES, x_phys))
    if not _heuristic_passes(driver_dict):
        return 0.0, 350.0, float(_PITOT3_FAILURE_SENTINEL), False

    # NOTE: no mesh_scale / transducer_xs argument by design - see module
    # docstring.  These defaults ARE the optimisation configuration.
    t_hold, impact, delta_vs, ok = run_l1d(
        x_phys=x_phys,
        ind_number=job_tag,
        test_gas_p1=float(base_config_dict()["p1"]),
        output_root=output_root,
    )
    return float(t_hold), float(impact), float(delta_vs), bool(ok)


# Re-scoring an archive without re-running the CFD

_XBUF_RE = re.compile(r"x_buffer=(-?\d+\.\d+)")


def rescore_member(job_root, p_burst):
    """Recompute ``(t_hold, impact, delta_vs, ok)`` from existing L1d output.

    The objectives are derived entirely from files already on disk, so a
    change to how they are MEASURED (the hold-window rule, the contact
    test) can be applied to a finished archive without repeating hours of
    CFD.  The simulation itself is deterministic and unaffected.

    ``x_buffer`` is read back from the generated job script - that is the
    value L1d was actually given, quantised to the template's "%.6f", and
    is the threshold parse_l1d_outputs must compare against.
    """
    script = job_root / f"{job_root.name}.py"
    if not script.is_file():
        return None
    m = _XBUF_RE.search(script.read_text())
    if m is None:
        return None
    return parse_l1d_outputs(
        job_dir=str(job_root / job_root.name),
        p_burst=p_burst,
        transducer_xs=TRANSDUCER_XS,
        x_buffer=float(m.group(1)),
    )


# Archive loading

def load_archive(run_dir: Path) -> list[dict]:
    """Read ``summary/archive.csv`` into a list of member dicts.

    Each dict carries the normalised design, the normalised objectives,
    and the physical design recovered with the optimiser's own inverse
    transform (which handles the two conditional bounds: p4 is scaled
    relative to driver_p, and reservoir_p has driver_p as its lower bound).
    """
    path = run_dir / "summary" / "archive.csv"
    members = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            x_norm = [float(row[f"design_{i}"]) for i in range(6)]
            f_norm = (float(row["f_0"]), float(row["f_1"]))
            members.append({
                "gen_found": int(row["gen_found"]),
                "lineage_id": int(row["lineage_id"]),
                "x_norm": x_norm,
                "x_phys": variable_untransformation(x_norm, BOUNDS),
                "f_norm": f_norm,
                # Optimisation-time physical objectives, for comparison.
                "opt_phys": unnormalise_fitness(
                    f_norm, APPROX_IDEAL_2D, APPROX_NADIR_2D),
                "g_al_0": float(row.get("g_al_0", "nan")),
            })
    return members


# Driver

def _finish(args, out_dir, members, results, n, t0):
    """Print the run summary and write archive_rerun.csv."""
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r[3])
    print(f"\nDone in {elapsed/60:.1f} min.  {n_ok}/{n} non-sentinel.")

    csv_path = out_dir / "archive_rerun.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "member", "gen_found", "lineage_id",
            *DESIGN_NAMES,
            "t_hold_s", "impact_speed_ms", "delta_vs1_ms", "ok",
            "opt_t_hold_s", "opt_impact_speed_ms", "opt_g_al_0",
        ])
        for i, m in enumerate(members):
            t_hold, impact, delta_vs, ok = results[i]
            w.writerow([
                i, m["gen_found"], m["lineage_id"],
                *[f"{v:.10g}" for v in m["x_phys"]],
                f"{t_hold:.10g}", f"{impact:.10g}", f"{delta_vs:.10g}", int(ok),
                f"{m['opt_phys'][0]:.10g}", f"{m['opt_phys'][1]:.10g}",
                f"{m['g_al_0']:.10g}",
            ])
    print(f"Wrote {csv_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="al_cht_0122",
                   help="Run directory name under --results-dir "
                        "(default: al_cht_0122)")
    p.add_argument("--results-dir", default=None,
                   help="Parent directory holding the run (default: "
                        "Results/al_cht_recomb).  Completed runs also live "
                        "under Results/al_cht_recomb.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write archive_rerun.csv and the L1d job dirs "
                        "(default: <run_dir>/archive_rerun)")
    p.add_argument("--reparse", action="store_true",
                   help="Do not run L1d.  Recompute the objectives from the "
                        "job directories already under --out-dir, and rewrite "
                        "archive_rerun.csv.  Use after changing how an "
                        "objective is measured.")
    p.add_argument("--grace-sim-time-ms", type=float, default=None,
                   help="Override the watchdog SUCCESS-AND-DONE budget: "
                        "simulated ms allowed past diaphragm rupture before "
                        "the run is killed early (module default: "
                        f"{l1d_job.GRACE_SIM_TIME_S*1e3:g} ms).  Pass a value "
                        ">= t_finish to disable success-based early "
                        "termination entirely.  Affects wall clock only, "
                        "never the physics.")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel L1d worker processes (default: min(n_members, cpus))")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    run_dir = results_dir / args.run
    if not run_dir.is_dir():
        print(f"ERROR: no such run directory: {run_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "archive_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    members = load_archive(run_dir)
    n = len(members)
    workers = args.max_workers or min(n, os.cpu_count() or 1)

    print(f"Run          : {args.run}")
    print(f"Archive size : {n}")
    print(f"Workers      : {workers}")
    grace_s = (args.grace_sim_time_ms * 1e-3
               if args.grace_sim_time_ms is not None else None)
    eff_grace = grace_s if grace_s is not None else l1d_job.GRACE_SIM_TIME_S
    print(f"mesh_scale   : {l1d_job.MESH_SCALE_FACTOR}  (optimisation default)")
    print(f"grace        : {eff_grace*1e3:g} ms simulated past rupture"
          + ("  [DEFAULT]" if grace_s is None else "  [OVERRIDDEN]")
          + f"   (t_finish = {l1d_job.T_FINISH*1e3:g} ms)")
    print(f"transducers  : {l1d_job.TRANSDUCER_XS} m")
    print(f"test_gas_p1  : {base_config_dict()['p1']:.0f} Pa")
    print(f"Output       : {out_dir}\n")

    t0 = time.time()
    results = {}

    if args.reparse:
        for i, m in enumerate(members):
            root = out_dir / "L1d_Outputs" / f"{args.run}_arch{i:02d}"
            if not root.is_dir():
                root = out_dir / "L1d_Outputs" / f"DEAP_{args.run}_arch{i:02d}"
            res = rescore_member(root, m["x_phys"][2])
            if res is None:
                print(f"  member {i:2d}: no job directory at {root}",
                      file=sys.stderr)
                res = (0.0, 350.0, float(_PITOT3_FAILURE_SENTINEL), False)
            results[i] = (float(res[0]), float(res[1]), float(res[2]), bool(res[3]))
            t_hold, impact, delta_vs, ok = results[i]
            print(f"  [{i+1:2d}/{n}] member {i:2d} "
                  f"(lineage {m['lineage_id']:5d})  {'ok ' if ok else 'SENT'}  "
                  f"t_hold={t_hold*1e3:7.3f} ms  impact={impact:7.2f} m/s  "
                  f"dvs1={delta_vs:8.1f} m/s")
        _finish(args, out_dir, members, results, n, t0)
        return 0

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(grace_s,)) as pool:
        futures = {
            pool.submit(evaluate_member, m["x_phys"],
                        f"{args.run}_arch{i:02d}", str(out_dir)): i
            for i, m in enumerate(members)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:                      # noqa: BLE001
                print(f"  member {i:2d}: worker raised {e!r}", file=sys.stderr)
                results[i] = (0.0, 350.0, float(_PITOT3_FAILURE_SENTINEL), False)
            t_hold, impact, delta_vs, ok = results[i]
            tag = "ok " if ok else "SENT"
            print(f"  [{len(results):2d}/{n}] member {i:2d} "
                  f"(lineage {members[i]['lineage_id']:5d})  {tag}  "
                  f"t_hold={t_hold*1e3:7.3f} ms  "
                  f"impact={impact:7.2f} m/s  "
                  f"dvs1={delta_vs:8.1f} m/s", flush=True)

    _finish(args, out_dir, members, results, n, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
