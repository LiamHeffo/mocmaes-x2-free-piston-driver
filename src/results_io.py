"""
Output directory layout and per-generation CSV writer.

setup_run_directory creates a numbered run folder under
<project_root>/Results/<category>/, e.g.

    Results/parent_value_with_recomb/pv_w_rec_0008/

setup_subfolders creates one subfolder per output type inside the
run directory.  All file writers in the project should write into
these subfolders rather than cwd.

write_population_csv emits one row per individual using the stable
POPULATION_FIELDS column order, so downstream tooling can rely on it.
"""

import csv
import re
from pathlib import Path


POPULATION_FIELDS = [
    "generation", "ind_number", "parent_idx", "chosen", "offspring_ind_number",
    "sigma", "hv_contribution",
    # Raw (physical) design variables
    "raw_pct_he", "raw_driver_p", "raw_p4",
    "raw_d_throat", "raw_reservoir_p", "raw_buffer_length",
    # Scaled (algorithm-space [1, 2]) design variables
    "scaled_pct_he", "scaled_driver_p", "scaled_p4",
    "scaled_d_throat", "scaled_reservoir_p", "scaled_buffer_length",
    # Raw (dimensional) objectives
    "raw_delta_vs1", "raw_hold_time", "raw_impact_speed",
    # Scaled (normalised [0, 1]) objectives
    "scaled_delta_vs1", "scaled_hold_time", "scaled_impact_speed",
]


def setup_run_directory(category, run_prefix, base=None):
    """Create a new numbered run directory under <base>/Results/<category>/.

    The new directory is named ``<run_prefix>_NNNN`` where NNNN is one greater
    than the largest existing such number, zero-padded to four digits.

    Parameters
    ----------
    category : str
        Folder name under ``Results/`` for this experiment family
        (e.g. ``"parent_value_with_recomb"``).
    run_prefix : str
        Prefix for the per-run subfolder (e.g. ``"pv_w_rec"``).
    base : Path or str, optional
        Project root.  Defaults to two levels up from this file
        (i.e. the repository root).

    Returns
    -------
    pathlib.Path
        The newly created run directory.
    """
    if base is None:
        base = Path(__file__).resolve().parent.parent
    base = Path(base)

    parent = base / "Results" / category
    parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(run_prefix)}_(\d+)$")
    existing = []
    for child in parent.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                existing.append(int(match.group(1)))

    next_n = (max(existing) + 1) if existing else 1
    run_dir = parent / f"{run_prefix}_{next_n:04d}"
    run_dir.mkdir()
    return run_dir


def setup_subfolders(run_dir, names):
    """Create the named subfolders under ``run_dir``; return ``{name: Path}``."""
    out = {}
    for name in names:
        path = Path(run_dir) / name
        path.mkdir(exist_ok=True)
        out[name] = path
    return out


def write_population_csv(path, rows):
    """Write one snapshot to ``path``, one row per individual.

    Unknown / not-yet-known cells (e.g. ``offspring_ind_number`` for the
    most recent generation) are written as empty strings.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=POPULATION_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k])
                             for k in POPULATION_FIELDS})
