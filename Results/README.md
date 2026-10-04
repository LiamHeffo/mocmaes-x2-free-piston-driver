# Optimisation runs

Seven runs, all of the configuration in `config/experiments.yaml`:
`ArnoldCHT_AL`, population 24, initial step size 0.1, 500 generations,
tolerance schedule `[3585, 100.0, 100, 250]`, mesh scale 1, each seeded from
`seed/init_population.npz`.

| run | wall clock | finished | archive members |
|---|---|---|---|
| al_cht_0113 | 34.2 h | 2026-09-08 | 53 |
| al_cht_0114 | 33.3 h | 2026-09-10 | 88 |
| al_cht_0115 | 35.8 h | 2026-09-12 | 100 |
| al_cht_0116 | 44.1 h | 2026-09-13 | 17 |
| al_cht_0119 | 32.4 h | 2026-09-19 | 100 |
| al_cht_0121 | 31.8 h | 2026-09-22 | 100 |
| al_cht_0122 | 41.7 h | 2026-09-23 | 100 |

Every member of every archive satisfies the final tolerance. The archive is
capped at 100 members.

## Contents of a run directory

```
summary/output.txt               configuration, timing, generation count
summary/archive.csv              the external non-dominated archive
convergence/convergence_data.txt hypervolume and feasible-count histories
al_diagnostics/                  per-generation Augmented Lagrangian state
arnold_diagnostics/              per-generation and per-call covariance updates
strategy_diagnostics/            per-generation strategy state
parents/                         parent snapshots, every tenth generation
population/population.tar.gz     per-generation offspring
archive_rerun/archive_rerun.csv  archive members re-simulated in L1d
```

`summary/archive.csv` holds, per member, the generation it was found, its
design vector in the normalised space, its two objectives normalised against
the reference points in `src/problem/config.py`, and its constraint
residual. `src/plot_run.py` converts the objectives back to hold time in ms
and impact speed in m/s.

Draw any run with:

    python3 src/plot_run.py Results/al_cht_recomb/al_cht_0122

The raw L1d traces behind `archive_rerun.csv` are about 3 GB per run and are
not distributed; regenerate them with `src/rerun_archive_l1d.py`.
