"""
End-to-end smoke test for the Arnold CHT diagnostics pipeline (no L1d).

Drives several synthetic generations through the same lifecycle main.py
uses - generate() -> apply_arnold_infeasibility() -> drain_and_persist()
(per gen) -> synthetic eval -> update() - with a large sigma so that plenty
of offspring land infeasible and the diagnostic buffer fills.  Then it

* asserts the two per-generation CSVs were written with the documented
  schema, and
* calls the four standalone plotters and asserts each PNG appears exactly
  once.

This locks in the "write every gen, plot once at the end" contract the
diagnostics are built around.
"""
import os
import sys

import numpy as np


import main  # noqa: F401  - registers creator.Individual2D / FitnessMulti2D
from deap import creator
from algorithm.cmaes import StrategyMultiObjective, cht_method
from problem.config import BOUNDS
from problem.feasibility import evaluate_constraints, is_feasible
from problem.transforms import variable_transformation

import arnold_diagnostics as ad


def _feasible_norm_pop(n, rng):
    from init_population_l1d import lhs_unit_to_physical
    out = []
    while len(out) < n:
        phys = lhs_unit_to_physical(rng.random(6))
        x_norm = variable_transformation([phys], BOUNDS)[0]
        if is_feasible(evaluate_constraints(x_norm, BOUNDS)):
            out.append([float(v) for v in x_norm])
    return out


def _make_parents(x_norms):
    parents = []
    for i, x in enumerate(x_norms):
        ind = creator.Individual2D(x)
        ind.ind_number = i
        ind.bounds = BOUNDS
        ind.sim_type = 'ArnoldCHT_AL'
        ind.al_tol = 3585.0
        ind.fitness.values = (0.5 + 0.01 * i, 0.5 - 0.005 * i)
        ind._feasible = True
        ind._g = evaluate_constraints(x, BOUNDS)
        ind._g_al = np.array([0.0])
        parents.append(ind)
    return parents


def _run_synthetic(out_dir, n_gen=8, mu=6):
    rng = np.random.default_rng(7)
    pop = _make_parents(_feasible_norm_pop(mu, rng))
    n_constraints = len(evaluate_constraints(pop[0], BOUNDS))
    strat = StrategyMultiObjective(
        pop, sigma=0.6, mu=mu, lambda_=mu,           # big σ -> many infeasibles
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=3585.0, n_constraints=n_constraints,
    )
    assert cht_method('ArnoldCHT_AL') == 'arnold'

    def _check(ind):
        g = evaluate_constraints(ind, BOUNDS)
        return is_feasible(g), g

    LAMBDA = mu
    for gen in range(1, n_gen + 1):
        offspring = strat.generate(creator.Individual2D)
        strat.apply_arnold_infeasibility(offspring, feasibility_check=_check)
        n_feasible = sum(1 for o in offspring if o._feasible)
        # Drain to CSV every generation (mirrors main.py).
        ad.drain_and_persist(
            strat, gen=gen, out_dir=out_dir,
            n_infeasible=(LAMBDA - n_feasible), n_lambda=LAMBDA,
        )
        for k, o in enumerate(offspring):
            if o._feasible:
                o.fitness.values = (0.4 + 0.02 * k + 0.01 * gen, 0.4 - 0.01 * k)
                o._g_al = np.array([rng.uniform(-50, 50)])
        strat.update(offspring)
    return strat


def test_arnold_diag_csvs_written(tmp_path):
    out_dir = tmp_path / "arnold_diagnostics"
    _run_synthetic(out_dir)

    per_gen = out_dir / "arnold_per_gen.csv"
    per_call = out_dir / "arnold_per_call.csv"
    assert per_gen.exists(), "arnold_per_gen.csv not written"
    assert per_call.exists(), "arnold_per_call.csv not written"

    import csv
    with per_gen.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "per_gen.csv has no data rows"
    # Schema present and the heatmap source is populated.
    for col in ("generation", "infeasibility_rate", "n_lambda",
                "n_constraints", "per_constraint_violation_count"):
        assert col in rows[0], f"missing per_gen column {col}"
    # At least one generation recorded some violations (big σ guarantees it).
    import json
    total_viol = sum(sum(json.loads(r["per_constraint_violation_count"]))
                     for r in rows)
    assert total_viol > 0, "expected some constraint violations with σ=0.6"

    with per_call.open() as f:
        call_rows = list(csv.DictReader(f))
    assert call_rows, "per_call.csv has no data rows"
    for col in ("lineage_id", "condition_number_after", "active_js", "v_norms"):
        assert col in call_rows[0], f"missing per_call column {col}"


def test_arnold_plot_all_writes_four_figures(tmp_path):
    out_dir = tmp_path / "arnold_diagnostics"
    _run_synthetic(out_dir)

    figs = ad.plot_all(out_dir)
    names = {p.name for p in figs}
    expected = {
        "arnold_constraint_heatmap.png",
        "arnold_infeasibility_rate.png",
        "arnold_mean_vj.png",
        "arnold_condition_by_lineage.png",
    }
    assert expected.issubset(names), f"missing figures: {expected - names}"
    for p in figs:
        assert p.exists() and p.stat().st_size > 0, f"empty figure {p}"
