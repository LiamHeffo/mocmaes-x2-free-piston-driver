"""Plot the results of a single optimisation run.

Reads one run directory as main.py wrote it and draws whatever it finds:
the external non-dominated archive, the hypervolume history, the feasible
offspring count, and, for the Augmented Lagrangian strategies, the
constraint tolerance and multiplier histories.

The optimiser already writes its own figures at the end of a run. This
script exists so a finished run can be re-plotted without re-running it,
and so runs can be compared by pointing it at each in turn.

Usage:

    python3 src/plot_run.py Results/al_cht_recomb/al_cht_0122
    python3 src/plot_run.py <run-dir> --outdir somewhere --format png
    python3 src/plot_run.py <run-dir> --separate

Nothing here is specific to a particular run or to any choice made in
post-processing: every panel is drawn from the run's own output, with no
filtering. Panels whose input files are absent are skipped, so it works on
a run from any sim_type and on a run that stopped early.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem.config import APPROX_IDEAL_2D, APPROX_NADIR_2D        # noqa: E402
from problem.transforms import unnormalise_fitness                 # noqa: E402

# Block headings in convergence_data.txt. The file holds several
# "Generation N: value" series one after another, so a reader that greps
# for "Generation" alone silently concatenates all of them.
HV_NADIR = "Hypervolume (nadir ref, higher = better) per generation:"
FEASIBLE = "Feasible offspring per generation"


def read_convergence_block(path: Path, heading: str) -> dict[int, float]:
    """Return {generation: value} for one block of convergence_data.txt.

    ``heading`` is matched as a prefix, so the caller does not have to
    reproduce a trailing count such as "(out of 24)".
    """
    out: dict[int, float] = {}
    if not path.is_file():
        return out
    in_block = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(heading):
            in_block = True
            continue
        m = re.match(r"Generation (\d+):\s*([-\d.eE+]+)$", s)
        if m:
            if in_block:
                out[int(m.group(1))] = float(m.group(2))
            continue
        # any other non-blank line ends the block
        if in_block:
            in_block = False
    return out


def read_archive(path: Path):
    """Return the archive's objectives in physical units.

    ``f_0`` and ``f_1`` are stored normalised against the reference points
    in problem.config; convert them back so the axes carry units.
    Returns (hold_time_ms, impact_speed_ms).
    """
    if not path.is_file():
        return np.empty(0), np.empty(0)
    hold, impact = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                f = (float(row["f_0"]), float(row["f_1"]))
            except (KeyError, TypeError, ValueError):
                continue
            t_s, v_ms = unnormalise_fitness(f, APPROX_IDEAL_2D, APPROX_NADIR_2D)
            hold.append(t_s * 1e3)
            impact.append(v_ms)
    return np.asarray(hold), np.asarray(impact)


def _cell(raw):
    """Parse one diagnostics cell to a float.

    The Augmented Lagrangian carries one value per constraint, so columns
    such as ``lam``, ``mu`` and ``g_al_proxy`` are written as JSON lists
    ("[0.0]") while others are plain floats. Take the first component: the
    shock-speed constraint is the only one these runs declare. A blank or
    unparseable cell becomes NaN so a gap plots as a gap.
    """
    if raw is None:
        return np.nan
    raw = raw.strip()
    if not raw:
        return np.nan
    if raw.startswith("["):
        try:
            vals = json.loads(raw)
        except ValueError:
            return np.nan
        return float(vals[0]) if vals else np.nan
    try:
        return float(raw)
    except ValueError:
        return np.nan


def read_per_gen(path: Path, columns):
    """Return {column: array} for the named columns of a per-generation CSV."""
    out = {c: [] for c in columns}
    out["generation"] = []
    if not path.is_file():
        return {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                gen = int(row["generation"])
            except (KeyError, TypeError, ValueError):
                continue
            out["generation"].append(gen)
            for c in columns:
                out[c].append(_cell(row.get(c)))
    return {k: np.asarray(v) for k, v in out.items()}


def nondominated(hold, impact):
    """Indices of the non-dominated set for (maximise hold, minimise impact)."""
    keep = []
    for i in range(len(hold)):
        dominated = any(
            (hold[j] >= hold[i] and impact[j] <= impact[i])
            and (hold[j] > hold[i] or impact[j] < impact[i])
            for j in range(len(hold))
        )
        if not dominated:
            keep.append(i)
    return np.asarray(sorted(keep, key=lambda i: hold[i]), dtype=int)


def panel_archive(ax, run_dir: Path) -> bool:
    hold, impact = read_archive(run_dir / "summary" / "archive.csv")
    if hold.size == 0:
        return False
    ax.scatter(hold, impact, s=18, alpha=0.65, edgecolor="none",
               label=f"{hold.size} members")
    # The archive is already non-dominated by construction, so this is
    # normally just a line through every point; it still earns its place on
    # a run that stopped early, where it is not.
    nd = nondominated(hold, impact)
    if nd.size > 1:
        label = None if nd.size == hold.size else f"front ({nd.size})"
        ax.plot(hold[nd], impact[nd], lw=1.2, alpha=0.9, label=label)
    ax.set_xlabel("driver hold time [ms]")
    ax.set_ylabel("piston impact speed [m/s]")
    ax.set_title("External archive")
    ax.legend(frameon=False, fontsize="small")
    ax.grid(alpha=0.25)
    return True


def panel_hypervolume(ax, run_dir: Path) -> bool:
    hv = read_convergence_block(
        run_dir / "convergence" / "convergence_data.txt", HV_NADIR)
    if not hv:
        return False
    gens = np.array(sorted(hv))
    ax.plot(gens, [hv[g] for g in gens], lw=1.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("hypervolume")
    ax.set_title("Hypervolume (nadir reference, higher is better)")
    ax.grid(alpha=0.25)
    return True


def panel_feasible(ax, run_dir: Path) -> bool:
    feas = read_convergence_block(
        run_dir / "convergence" / "convergence_data.txt", FEASIBLE)
    if not feas:
        return False
    gens = np.array(sorted(feas))
    ax.plot(gens, [feas[g] for g in gens], lw=0.9, alpha=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("feasible offspring")
    ax.set_title("Feasible offspring per generation")
    ax.grid(alpha=0.25)
    return True


def panel_constraint(ax, run_dir: Path) -> bool:
    """Tolerance and constraint residual, for the AL strategies.

    ``g_al`` is the residual against the current tolerance, so the
    tolerance itself is recovered as the offset that makes the raw
    measurement comparable across generations.
    """
    d = read_per_gen(run_dir / "al_diagnostics" / "al_per_gen.csv",
                     ["g_al_proxy", "g_al_min", "g_al_max", "lam"])
    if not d or d["generation"].size == 0:
        return False
    g = d["generation"]
    ax.fill_between(g, d["g_al_min"], d["g_al_max"], alpha=0.2,
                    label="per-individual range")
    ax.plot(g, d["g_al_proxy"], lw=1.2, label="proxy")
    ax.axhline(0.0, ls="--", lw=0.8, color="0.3", label="tolerance")
    ax.set_xlabel("generation")
    ax.set_ylabel("constraint residual [m/s]")
    ax.set_title("Augmented Lagrangian constraint")
    ax.legend(frameon=False, fontsize="small")
    ax.grid(alpha=0.25)
    return True


def panel_multiplier(ax, run_dir: Path) -> bool:
    d = read_per_gen(run_dir / "al_diagnostics" / "al_per_gen.csv",
                     ["lam", "mu"])
    if not d or d["generation"].size == 0:
        return False
    g = d["generation"]
    drawn = False
    for key, label in (("lam", "multiplier"), ("mu", "penalty")):
        y = d.get(key)
        if y is not None and np.isfinite(y).any():
            ax.plot(g, y, lw=1.2, label=label)
            drawn = True
    if not drawn:
        return False
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("generation")
    ax.set_ylabel("value")
    ax.set_title("Augmented Lagrangian state")
    ax.legend(frameon=False, fontsize="small")
    ax.grid(alpha=0.25)
    return True


PANELS = [
    ("archive", panel_archive),
    ("hypervolume", panel_hypervolume),
    ("feasible", panel_feasible),
    ("constraint", panel_constraint),
    ("al_state", panel_multiplier),
]


def run_header(run_dir: Path) -> str:
    """First few 'key = value' lines of summary/output.txt, for the title."""
    path = run_dir / "summary" / "output.txt"
    if not path.is_file():
        return run_dir.name
    wanted = ("Simulation Type", "pop size", "step size",
              "Number of generations")
    found = {}
    for line in path.read_text().splitlines():
        if line.startswith("*"):
            break
        k, _, v = line.partition("=")
        if k.strip() in wanted:
            found[k.strip()] = v.strip()
    bits = [run_dir.name] + [f"{k} {v}" for k, v in found.items()]
    return "   ".join(bits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path,
                    help="a run directory written by src/main.py")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to write (default: <run-dir>/plots)")
    ap.add_argument("--format", default="png", choices=("png", "eps", "pdf"),
                    help="output format (default png)")
    ap.add_argument("--separate", action="store_true",
                    help="one file per panel instead of a single sheet")
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    outdir = args.outdir or (run_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.separate:
        written = []
        for name, fn in PANELS:
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            if fn(ax, run_dir):
                fig.tight_layout()
                p = outdir / f"{name}.{args.format}"
                fig.savefig(p, dpi=200)
                written.append(p)
            else:
                print(f"  no data for '{name}', skipped", file=sys.stderr)
            plt.close(fig)
    else:
        available = []
        for name, fn in PANELS:
            fig, ax = plt.subplots()
            ok = fn(ax, run_dir)
            plt.close(fig)
            if ok:
                available.append((name, fn))
            else:
                print(f"  no data for '{name}', skipped", file=sys.stderr)
        if not available:
            raise SystemExit(f"no plottable data under {run_dir}")
        ncol = 2
        nrow = (len(available) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 3.9 * nrow))
        flat = np.atleast_1d(axes).ravel()
        for ax, (_, fn) in zip(flat, available):
            fn(ax, run_dir)
        for ax in flat[len(available):]:
            ax.axis("off")
        fig.suptitle(run_header(run_dir), fontsize="medium")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = outdir / f"{run_dir.name}.{args.format}"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written = [p]

    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
