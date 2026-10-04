"""
Objective-space scatter plots for the Pareto front history.

All functions accept an optional ``out_dir`` keyword argument.  When given,
the plot is written into that directory; otherwise it is written to the
current working directory (legacy behaviour).

Functions
---------
plot_objective_space(fitness_history, obj1, obj2, **kwargs)
    Two-axis Pareto scatter for the supported objective pairs:
        ('delta_vs1', 'hold_time')
        ('delta_vs1', 'impact_speed')
        ('hold_time',  'impact_speed')

plot_objective_space_3d(fitness_history, **kwargs)
    Three-axis Pareto scatter (delta_vs1, hold_time, impact_speed).

plot_objective_space_heatmap(fitness_history, **kwargs)
    Two-axis hold_time vs impact_speed scatter, with delta_vs1 encoded
    as a viridis colour gradient.

The last MU entries in fitness_history are treated as the current Pareto
front and highlighted.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _resolve_path(filename, out_dir):
    """Return a save path inside out_dir if given, else just the filename (cwd)."""
    if out_dir is None:
        return filename
    return str(Path(out_dir) / filename)


def plot_objective_space(fitness_history, obj1, obj2, **kwargs):
    """Scatter-plot the objective-space history and highlight the current front.

    Parameters
    ----------
    fitness_history : list of tuples
        All evaluated fitness values (normalised or physical).
    obj1, obj2 : str
        Names of the two objectives to plot (see module docstring).
    MU : int, optional
        Population size - the last MU entries are the current front (default 10).
    gen : int, optional
        Generation number for the filename (default 0).
    normalised : bool, optional
        If True, axes are clamped to [0, 1.1] (default True).
    out_dir : Path or str, optional
        Directory to save the plot in.  Defaults to current working directory.
    """
    MU         = kwargs.get('MU',         10)
    gen        = kwargs.get('gen',         0)
    normalised = kwargs.get('normalised',  True)
    out_dir    = kwargs.get('out_dir',     None)

    if obj1 == 'delta_vs1' and obj2 == 'hold_time':
        plt.figure(dpi=200)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Residual of Shock Speed")
        plt.ylabel("Normalised Hold Time")

        delta_vs_history   = [entry[0] for entry in fitness_history]
        hold_time_history  = [entry[1] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(hold_time_history)))

        plt.scatter(delta_vs_history,       hold_time_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(delta_vs_history[-MU:], hold_time_history[-MU:], facecolors='none', edgecolors='green')

        filename = f"pareto_dvs1_holdtime_gen_{gen:04d}.png"
        plt.savefig(_resolve_path(filename, out_dir))
        plt.close()

    elif obj1 == 'delta_vs1' and obj2 == 'impact_speed':
        plt.figure(dpi=200)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Residual of Shock Speed")
        plt.ylabel("Normalised Impact Speed")

        delta_vs_history      = [entry[0] for entry in fitness_history]
        impact_speed_history  = [entry[2] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(impact_speed_history)))

        plt.scatter(delta_vs_history,       impact_speed_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(delta_vs_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='orange')

        filename = f"pareto_dvs1_impactspeed_gen_{gen:04d}.png"
        plt.savefig(_resolve_path(filename, out_dir))
        plt.close()

    elif obj1 == 'hold_time' and obj2 == 'impact_speed':
        plt.figure(dpi=200)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Hold Time")
        plt.ylabel("Normalised Impact Speed")

        hold_time_history     = [entry[1] for entry in fitness_history]
        impact_speed_history  = [entry[2] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(impact_speed_history)))

        plt.scatter(hold_time_history,       impact_speed_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(hold_time_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='purple')

        filename = f"pareto_holdtime_impactspeed_gen_{gen:04d}.png"
        plt.savefig(_resolve_path(filename, out_dir))
        plt.close()


def plot_holdtime_impactspeed_2d(fitness_history, **kwargs):
    """2-objective Pareto scatter for the CHT_AL run mode.

    In CHT_AL the fitness tuple is 2-D (hold_time, impact_speed) - delta_vs1
    is no longer a Pareto objective but an Augmented-Lagrangian constraint.

    The purple-highlighted scatter shows the *surviving μ parent set* if
    ``parent_fitness`` is supplied, falling back to ``fitness_history[-MU:]``
    (the last offspring batch) otherwise.  The parent-set view is the
    one users typically expect - it is the post-selection Pareto front
    that seeds the next generation, not the pre-selection candidate pool
    that includes sentinel offspring about to be discarded.
    """
    MU             = kwargs.get('MU',             10)
    gen            = kwargs.get('gen',             0)
    normalised     = kwargs.get('normalised',      True)
    out_dir        = kwargs.get('out_dir',         None)
    parent_fitness = kwargs.get('parent_fitness',  None)

    plt.figure(dpi=200)
    plt.title("Pareto Frontier (CHT_AL)")
    plt.xlabel("Normalised Hold Time")
    plt.ylabel("Normalised Impact Speed")

    hold_time_history    = [entry[0] for entry in fitness_history]
    impact_speed_history = [entry[1] for entry in fitness_history]

    if normalised:
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
    else:
        plt.ylim((-0.005, 2 * np.max(impact_speed_history)))

    plt.scatter(hold_time_history, impact_speed_history,
                facecolors='none', edgecolors='lightblue',
                label='all evaluated offspring')

    if parent_fitness:
        parent_ht = [f[0] for f in parent_fitness]
        parent_is = [f[1] for f in parent_fitness]
        plt.scatter(parent_ht, parent_is,
                    facecolors='none', edgecolors='purple',
                    label=f'surviving parents (μ={len(parent_fitness)})')
    else:
        plt.scatter(hold_time_history[-MU:], impact_speed_history[-MU:],
                    facecolors='none', edgecolors='purple',
                    label='last offspring batch')

    plt.legend(loc='best', fontsize=8)

    filename = f"pareto_holdtime_impactspeed_gen_{gen:04d}.png"
    plt.savefig(_resolve_path(filename, out_dir))
    plt.close()


def plot_objective_space_3d(fitness_history, **kwargs):
    """3D scatter of the full normalised objective space.

    All sampled individuals are drawn as translucent grey points; the last
    MU entries (current Pareto front) are over-plotted in opaque green.

    The camera is positioned in the (-x, -y, +z) octant so that the
    (0, 0, 0) ideal corner is closest to the viewer.

    Parameters
    ----------
    fitness_history : list of tuples
        Each entry is (delta_vs1, hold_time, impact_speed), normalised.
    MU : int, optional
        Population size - the last MU entries are the current front.
    gen : int, optional
        Generation number for the filename.
    normalised : bool, optional
        If True, axes are clamped to [0, 1.1].
    out_dir : Path or str, optional
        Directory to save the plot in.
    """
    MU         = kwargs.get('MU',         10)
    gen        = kwargs.get('gen',         0)
    normalised = kwargs.get('normalised',  True)
    out_dir    = kwargs.get('out_dir',     None)

    delta_vs_history     = [entry[0] for entry in fitness_history]
    hold_time_history    = [entry[1] for entry in fitness_history]
    impact_speed_history = [entry[2] for entry in fitness_history]

    fig = plt.figure(dpi=200)  # paper-quality is dpi 300+, but per-gen plots
    # are regenerated 100s of times in long runs so memory cost matters; bump
    # this up if you need higher fidelity for a specific publication figure.
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title("Pareto Frontier (3D)")
    ax.set_xlabel("Normalised Residual of Shock Speed")
    ax.set_ylabel("Normalised Hold Time")
    ax.set_zlabel("Normalised Impact Speed")

    if normalised:
        ax.set_xlim((0, 1.1))
        ax.set_ylim((0, 1.1))
        ax.set_zlim((0, 1.1))

    # Camera in (-x, -y, +z) octant places (0, 0, 0) closest to the viewer.
    ax.view_init(elev=25, azim=-135)

    ax.scatter(delta_vs_history,       hold_time_history,       impact_speed_history,
               c='grey', alpha=0.15, s=10, depthshade=False)
    ax.scatter(delta_vs_history[-MU:], hold_time_history[-MU:], impact_speed_history[-MU:],
               c='green', alpha=1.0,  s=25, depthshade=False)

    filename = f"pareto_3d_gen_{gen:04d}.png"
    plt.savefig(_resolve_path(filename, out_dir))
    plt.close()


def plot_objective_space_heatmap(fitness_history, **kwargs):
    """2D scatter of hold_time vs impact_speed, coloured by delta_vs1.

    delta_vs1 (residual of shock speed) is encoded as the marker colour
    via a perceptually uniform colormap (viridis).  Final-population
    individuals are outlined in black to keep with the convention of
    distinguishing the current front.

    Parameters
    ----------
    fitness_history : list of tuples
        Each entry is (delta_vs1, hold_time, impact_speed), normalised.
    MU : int, optional
        Population size - the last MU entries are the current front.
    gen : int, optional
        Generation number for the filename.
    normalised : bool, optional
        If True, axes are clamped to [0, 1.1].
    out_dir : Path or str, optional
        Directory to save the plot in.
    """
    MU         = kwargs.get('MU',         10)
    gen        = kwargs.get('gen',         0)
    normalised = kwargs.get('normalised',  True)
    out_dir    = kwargs.get('out_dir',     None)

    delta_vs_history     = np.array([entry[0] for entry in fitness_history])
    hold_time_history    = np.array([entry[1] for entry in fitness_history])
    impact_speed_history = np.array([entry[2] for entry in fitness_history])

    plt.figure(dpi=200)
    plt.title("Pareto Frontier (Hold Time vs Impact Speed, coloured by Shock Speed)")
    plt.xlabel("Normalised Hold Time")
    plt.ylabel("Normalised Impact Speed")

    if normalised:
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(delta_vs_history)), float(np.max(delta_vs_history))

    sc = plt.scatter(hold_time_history, impact_speed_history,
                     c=delta_vs_history, cmap='viridis',
                     vmin=vmin, vmax=vmax, s=18, alpha=0.7)

    plt.scatter(hold_time_history[-MU:], impact_speed_history[-MU:],
                c=delta_vs_history[-MU:], cmap='viridis',
                vmin=vmin, vmax=vmax, s=40,
                edgecolors='black', linewidths=0.8)

    cbar = plt.colorbar(sc)
    cbar.set_label("Normalised Residual of Shock Speed")

    filename = f"pareto_heatmap_gen_{gen:04d}.png"
    plt.savefig(_resolve_path(filename, out_dir))
    plt.close()
