#!/usr/bin/env bash
# run_l1d_cmaes.sh - Build a seed population (if needed) then launch a
# seeded L1d CMA-ES optimisation run.
#
# Usage:
#   ./scripts/run_l1d_cmaes.sh [OPTIONS]
#
# Options (init step - ignored when --seed-npz is supplied):
#   --pop-size N          Population size (default: 24).  Must match the
#                         pop_size field in config/experiments.yaml.
#   --method METHOD       Quasi-random sampler: 'sobol' (default) or 'lhs'.
#   --seed N              Integer RNG seed for reproducible sampling.
#   --max-workers N       Parallel L1d worker processes (default: auto).
#   --max-eval-factor N   Abort init after N*pop_size total L1d evaluations
#                         (default: 64).
#   --mesh-scale N        Per-slug ncells multiplier passed to run_l1d
#                         (default: 1).
#   --out PATH            Where to save the init .npz
#                         (default: seed/init_population.npz).
#
# Options (shortcut - skip init):
#   --seed-npz PATH       Use an existing seed .npz (x_norm array) and skip
#                         the init step.
#
# Options (control):
#   --init-only           Build the seed population but do NOT launch CMA-ES.
#   -h, --help            Print this help and exit.
#
# Notes:
#   - Do NOT set PYTHONPATH=src before calling this script.  The gdtk
#     environment on PATH is already configured; adding src/ to PYTHONPATH
#     clobbers it and kills every l1d4-prep subprocess with an import error.
#     Python adds src/ automatically because main.py lives there.
#   - The pop_size passed to --pop-size MUST match the pop_size entry in
#     config/experiments.yaml - main.py uses the first MU rows of x_norm
#     and will pad or truncate if the sizes disagree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
SEED_NPZ=""
POP_SIZE=24
METHOD="sobol"
RNG_SEED=""
MAX_WORKERS=""
MAX_EVAL_FACTOR=128
MESH_SCALE=1
OUT="${PROJECT_ROOT}/seed/init_population.npz"
INIT_ONLY=false

# Argument parsing
usage() {
    # Print the header docblock (lines 2..first non-comment line).
    awk '/^[^#]/{exit} NR>1{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-npz)        SEED_NPZ="$2";           shift 2 ;;
        --pop-size)        POP_SIZE="$2";            shift 2 ;;
        --method)          METHOD="$2";              shift 2 ;;
        --seed)            RNG_SEED="$2";            shift 2 ;;
        --max-workers)     MAX_WORKERS="$2";         shift 2 ;;
        --max-eval-factor) MAX_EVAL_FACTOR="$2";     shift 2 ;;
        --mesh-scale)      MESH_SCALE="$2";          shift 2 ;;
        --out)             OUT="$2";                 shift 2 ;;
        --init-only)       INIT_ONLY=true;           shift   ;;
        -h|--help)         usage ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

# Step 1: build or locate the seed population
if [[ -n "${SEED_NPZ}" ]]; then
    if [[ ! -f "${SEED_NPZ}" ]]; then
        echo "ERROR: --seed-npz path does not exist: ${SEED_NPZ}" >&2
        exit 1
    fi
    echo "Using pre-existing seed population: ${SEED_NPZ}"
else
    echo "================================================================"
    echo "  Step 1 - building seed population"
    echo "  pop_size=${POP_SIZE}  method=${METHOD}  mesh_scale=${MESH_SCALE}"
    echo "================================================================"

    INIT_CMD=(
        python3 "${PROJECT_ROOT}/src/init_population_l1d.py"
        --pop-size        "${POP_SIZE}"
        --method          "${METHOD}"
        --max-eval-factor "${MAX_EVAL_FACTOR}"
        --mesh-scale      "${MESH_SCALE}"
        --out             "${OUT}"
    )
    [[ -n "${RNG_SEED}" ]]    && INIT_CMD+=(--seed "${RNG_SEED}")
    [[ -n "${MAX_WORKERS}" ]] && INIT_CMD+=(--max-workers "${MAX_WORKERS}")

    echo "Running: ${INIT_CMD[*]}"
    echo ""
    "${INIT_CMD[@]}"

    SEED_NPZ="${OUT}"
    echo ""
    echo "Seed population saved to: ${SEED_NPZ}"
fi

# Step 2: launch the CMA-ES run
if [[ "${INIT_ONLY}" == true ]]; then
    echo "Init-only mode - skipping CMA-ES launch."
    exit 0
fi

echo ""
echo "================================================================"
echo "  Step 2 - launching seeded CMA-ES run"
echo "  seed:   ${SEED_NPZ}"
echo "  config: ${PROJECT_ROOT}/config/experiments.yaml"
echo "================================================================"
echo ""

# exec replaces the shell process with Python, so exit codes propagate
# cleanly and there is no dangling shell wrapper in the process tree.
exec python3 "${PROJECT_ROOT}/src/main.py" --seed-npz "${SEED_NPZ}"
