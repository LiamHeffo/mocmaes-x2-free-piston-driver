"""
End-of-run post-processing plots for the Augmented-Lagrangian (CHT_AL /
ArnoldCHT_AL) runs.

These three figures answer questions that the per-generation CSVs do not
render directly:

* ``plot_archive_pareto`` - the external non-dominated archive
  (``strategy.external_archive``) scattered in the two normalised
  objectives.  This is "every point that was ever Pareto-optimal", which
  is a superset of the final parent front.

* ``plot_epsilon_and_gal`` - the AL constraint tolerance ε (``al_tol``)
  schedule together with the per-individual constraint value
  ``g_AL = delta_vs1 - ε`` across generations.  Shows whether the
  population is being driven toward the (moving) feasibility boundary.

* ``plot_diversity`` - a population-diversity metric (Deb's spread Δ)
  across generations, to reveal front collapse / degeneration.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Axis labels for the two normalised objectives, in fitness-tuple order.
# For the AL family fitness = (hold_time, impact_speed), both normalised
# to [0, 1] with 0 = best (see problem/evaluate.py + main.py header).
_OBJ0_LABEL = "Normalised driver hold time  (f₀, 0 = best)"
_OBJ1_LABEL = "Normalised piston impact speed  (f₁, 0 = best)"


def plot_archive_pareto(archive, out_path) -> Path | None:
    """Scatter the external non-dominated archive in normalised objective space.

    Parameters
    ----------
    archive : list of dict
        ``strategy.external_archive``.  Each entry has a ``"fitness"``
        tuple ``(f0, f1)`` in normalised space and a ``"gen_found"`` int.
    out_path : path-like
        Destination ``.eps`` file.

    Returns
    -------
    Path or None
        The written path, or None when the archive is empty (nothing to
        draw - the caller's ``archive.csv`` already records that state).

    The points are coloured by the generation at which each entry first
    entered the archive, so the temporal order of discovery is visible:
    darker = earlier, brighter = later.  Both axes are "lower is better",
    so the desirable frontier is the lower-left envelope.
    """
    out_path = Path(out_path)
    entries = [m for m in archive if len(m["fitness"]) == 2]
    if not entries:
        return None

    f0 = np.array([m["fitness"][0] for m in entries], dtype=float)
    f1 = np.array([m["fitness"][1] for m in entries], dtype=float)
    gen_found = np.array([m.get("gen_found", 0) for m in entries], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    sc = ax.scatter(f0, f1, c=gen_found, cmap="viridis",
                    s=36, edgecolor="k", linewidth=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Generation first archived")

    ax.set_xlabel(_OBJ0_LABEL)
    ax.set_ylabel(_OBJ1_LABEL)
    ax.set_title(f"External non-dominated archive  (N = {len(entries)})")
    # Solid light-grey grid rather than an alpha grid: the EPS/PostScript
    # backend flattens transparency to opaque, so alpha grids render as
    # heavy black lines.  A light solid grey is the EPS-safe equivalent.
    ax.grid(color="0.85", lw=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, format="eps")
    plt.close(fig)
    return out_path


def plot_epsilon_and_gal(epsilon_history, gal_history, out_path) -> Path | None:
    """Plot the ε (al_tol) schedule and per-individual g_AL across generations.

    Both quantities are in m/s and are drawn on a single shared linear
    y-axis (per the chosen layout).  Because ``g_AL = delta_vs1 - ε`` and
    ``delta_vs1`` typically floors well above ε, the g_AL cloud sits far
    above the ε line - that separation is itself informative (it is the
    residual shock-speed gap the AL is trying to close).

    Parameters
    ----------
    epsilon_history : list of (gen, epsilon)
        One entry per generation; ``epsilon`` is the scheduled ``al_tol``.
    gal_history : list of (gen, [g_al, ...])
        One entry per generation; the list holds each surviving parent's
        scalar g_AL for that generation (may be empty when every parent
        was a sentinel that generation).
    out_path : path-like
        Destination ``.eps`` file.

    Returns
    -------
    Path or None
        The written path, or None when there is no data at all.
    """
    out_path = Path(out_path)
    if not epsilon_history and not gal_history:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)

    # Per-individual g_AL scatter: expand each (gen, [values]) into points.
    xs, ys = [], []
    for gen, vals in gal_history:
        for v in vals:
            xs.append(gen)
            ys.append(v)
    # EPS has no transparency, so instead of an alpha cloud we use small
    # markers in a light blue - overlap still reads as a denser band.
    if xs:
        ax.scatter(xs, ys, s=8, color="#7fa8d4",
                   label="g_AL per parent", zorder=2)

    # Feasibility boundary: g_AL = 0 (delta_vs1 exactly at tolerance).
    ax.axhline(0.0, color="0.4", ls=":", lw=1.0, zorder=1,
               label="g_AL = 0 (feasible boundary)")

    # ε schedule as a line on the same axis.
    if epsilon_history:
        eg = [g for g, _ in epsilon_history]
        ev = [e for _, e in epsilon_history]
        ax.plot(eg, ev, color="C3", lw=1.8, zorder=3,
                label="ε schedule (al_tol)")

    ax.set_xlabel("Generation")
    ax.set_ylabel("m/s")
    ax.set_title("ε schedule and per-individual g_AL")
    # framealpha=1.0: the legend's default semi-transparent frame would
    # trip the EPS backend's "no transparency" flattening warning.
    ax.legend(loc="best", fontsize=8, framealpha=1.0)
    ax.grid(color="0.85", lw=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, format="eps")
    plt.close(fig)
    return out_path


def plot_diversity(diversity_history, out_path) -> Path | None:
    """Plot Deb's spread Δ of the parent front across generations.

    Parameters
    ----------
    diversity_history : list of (gen, delta)
        One entry per generation; ``delta`` is Deb's Δ on that
        generation's parent front (NaN when < 2 non-dominated points, in
        which case Δ is undefined and the point is left as a gap).
    out_path : path-like
        Destination ``.eps`` file.

    Returns
    -------
    Path or None
        The written path, or None when there is no data.

    Lower Δ = more uniform spacing along the front; a rising Δ or long
    NaN gaps flag the front collapsing toward a single point.
    """
    out_path = Path(out_path)
    if not diversity_history:
        return None

    gens = np.array([g for g, _ in diversity_history], dtype=float)
    delta = np.array([d if d is not None else np.nan
                      for _, d in diversity_history], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    ax.plot(gens, delta, color="C2", lw=1.4, marker="o", ms=3,
            zorder=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Deb's spread Δ  (lower = more uniform)")
    ax.set_title("Parent-front diversity across generations")
    ax.grid(color="0.85", lw=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, format="eps")
    plt.close(fig)
    return out_path
