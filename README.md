# Multi-objective optimisation of the X2 free-piston driver

Multi-objective CMA-ES applied to the free-piston driver of the X2 expansion
tube at the University of Queensland. Each candidate design is evaluated by
running the L1d4 quasi-one-dimensional flow solver.

The optimiser maximises the **driver hold time** and minimises the **piston
impact speed**, subject to a constraint on the **primary shock speed**.

- *Hold time* is how long the driver pressure stays within +/-10% of the
  burst pressure at the primary diaphragm after rupture, measured from the
  L1d pressure trace.
- *Impact speed* is how fast the piston is travelling when it first contacts
  the buffer.
- *Shock speed* must reach 3585 m/s. This enters as a constraint rather than
  a third objective, handled by an Augmented Lagrangian whose tolerance
  tightens as the run proceeds.

## Design variables

Six, optimised in a normalised [1, 2]^6 space and transformed to physical
units for evaluation (`src/problem/transforms.py`):

| # | variable | range | role in the L1d model |
|---|---|---|---|
| 1 | driver helium fraction (balance argon) | 70 - 100 % | driver gas composition |
| 2 | driver gas fill pressure | 1 kPa - 2.736 MPa | driver slug initial pressure |
| 3 | primary diaphragm burst pressure, p4 | 150 kPa - 40 MPa | diaphragm `p_burst` |
| 4 | orifice-plate bore diameter | 50 - 85 mm | throat area at the diaphragm |
| 5 | reservoir fill pressure | 1 kPa - 8 MPa | air reservoir driving the piston |
| 6 | buffer stud length | 50 - 150 mm | piston standoff from the buffer |

Bounds are in `src/problem/config.py`. Variables 2 and 3 are coupled: the
diaphragm cannot burst below the compression the driver can deliver, so
feasibility requires `p4 >= 14.62 * driver_p`. The upper bound on the driver
pressure is `40 MPa / 14.62`, and the coupling is why `p4_treatment` exists
in the configuration. A bore of 85 mm equals the shock tube bore, which removes
the orifice plate entirely.

## Method

Two pieces of constraint handling are combined:

**Arnold covariance adaptation.** An offspring that violates a bound is not
repaired and not evaluated. Instead it is consumed before evaluation: the
direction of its violation shrinks the parent's covariance along that
direction (Arnold and Hansen 2012, Eq. 6 and 7), and the slot then
contributes no candidate that generation. Over successive generations the
sampling distribution learns the shape of the feasible region.

**Augmented Lagrangian for the shock speed.** The shock-speed requirement is
adapted as a constraint using the formulation of Atamna et al. (2017) and
Dufosse and Hansen (2020), via the implementation in `pycma`. Its tolerance
is held wide for the first 100 generations, then tightened to 100 m/s by
generation 250 and held there. Starting wide lets the search explore the
design space before the constraint bites.

Selection is the hypervolume-indicator-based multi-objective CMA-ES of Voss
et al. (2010), over an external non-dominated archive capped at 100 members.

## Requirements

- Python 3.10 or later, and `pip install -r requirements.txt`.
- **L1d4**, from the gdtk / Eilmer suite: https://gdtk.uqcloud.net. The
  `l1d4-prep` and `l1d4` executables must be on `$PATH`. Nothing here
  imports gdtk as a Python module, so only those two binaries are needed.

Do not put `src` on `PYTHONPATH`. The gdtk install configures its own
Python environment, and prepending `src` shadows it, which makes every
`l1d4-prep` subprocess fail on import. The entry points add `src` to
`sys.path` themselves.

## Running an optimisation

```
./scripts/run_l1d_cmaes.sh --seed-npz seed/init_population.npz
```

That runs the single experiment in `config/experiments.yaml`, seeded from the
initial population used for every distributed run. Output goes to
`Results/al_cht_recomb/al_cht_NNNN/`, numbered automatically, beside the
reference runs.

Expect roughly **30 to 45 hours** per run on a workstation using 12 cores:
500 generations of 24 individuals, each a separate L1d simulation, run in
parallel across worker processes.

To build a fresh initial population instead of reusing the distributed one:

```
./scripts/run_l1d_cmaes.sh --pop-size 24 --seed 20 --mesh-scale 1
```

Initial designs are drawn from a Sobol sequence and rejected until
the population contains only designs whose primary diaphragm actually
ruptures. The acceptance rate is about 2%, so this step itself costs over a
thousand L1d evaluations.

## Layout

```
config/experiments.yaml   the configuration every distributed run used
seed/init_population.npz  that run set's initial population (Sobol, seed 20)
gas_models/               L1d gas models for the driver mix and the reservoir
scripts/run_l1d_cmaes.sh  build a seed population, then launch a run
src/main.py               entry point: generation loop, selection, output
src/algorithm/            CMA-ES, the constraint handling, hypervolume
src/problem/              design transforms, feasibility, L1d job and parsing
src/plot_run.py           plot a finished run
Results/                  the seven distributed runs; see Results/README.md
tests/                    unit tests: python3 -m pytest tests/
```

## Results

Seven runs of the same configuration are distributed under
`Results/al_cht_recomb/`, 558 archived designs in total. Every one of them
satisfies the shock-speed constraint the runs converged to, so the archives
are usable as they stand. `Results/README.md` lists the runs and what
each directory holds.

Each run writes its own figures as it finishes, into `postprocessing/` and
the diagnostics directories. To re-plot a finished run without re-running
it:

```
python3 src/plot_run.py Results/al_cht_recomb/al_cht_0122
```

To re-simulate the archived designs of a finished run at a different mesh
resolution, see `src/rerun_archive_l1d.py`.

## References

- Arnold, D. V. and Hansen, N. (2012). A (1+1)-CMA-ES for constrained
  optimisation. *GECCO*.
- Atamna, A., Auger, A. and Hansen, N. (2017). Augmented Lagrangian
  constraint handling for CMA-ES. *FOGA*.
- Dufosse, P. and Hansen, N. (2020). Augmented Lagrangian, penalty
  techniques and surrogate modelling for constrained optimisation with
  CMA-ES. *GECCO*.
- Voss, T., Hansen, N. and Igel, C. (2010). Improved step size adaptation
  for the MO-CMA-ES. *GECCO*.
- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M. and
  Gagne, C. (2012). DEAP: evolutionary algorithms made easy. *Journal of
  Machine Learning Research*, 13, 2171-2175. Supplies the individual and
  fitness types, and the non-dominated sorting and hypervolume routines
  used in selection.
- Jacobs, P. A. and Gollan, R. J. The gdtk / Eilmer gas dynamics toolkit.
  https://gdtk.uqcloud.net
- Hodson, J. (2025). Master's thesis, The University of Queensland. Source
  of the X2 driver geometry and the volume-conserving break-point model used
  in `src/problem/l1d_geometry.py`.

## Licence

MIT. See `LICENSE`.
