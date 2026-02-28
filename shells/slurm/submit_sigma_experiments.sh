#!/usr/bin/env bash
# =============================================================================
# SLURM Submission Script for Perturb Sigma Experiments
# =============================================================================
# Submits 12 perturb-only experiments across 6 problems, each with a
# constant-sigma variant (prefix "bench") and a linear-decay + tanh variant
# (prefix "tanh_schedule").
#
# Usage:
#   # Submit all 12 experiments:
#   bash shells/slurm/submit_sigma_experiments.sh
#
#   # Dry run (print commands without submitting):
#   bash shells/slurm/submit_sigma_experiments.sh --dry-run
#
#   # Submit only one problem:
#   bash shells/slurm/submit_sigma_experiments.sh --problem energy
#
#   # Override SLURM settings:
#   bash shells/slurm/submit_sigma_experiments.sh --partition gpu --time 48:00:00
#
# =============================================================================

set -e

# ---- Defaults ----
DRY_RUN=false
PROBLEM_FILTER=""
PARTITION="preempt"
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="pco_bench_rhel7"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true; shift ;;
        --problem)       PROBLEM_FILTER="$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --time)          TIME="$2"; shift 2 ;;
        --mem)           MEM="$2"; shift 2 ;;
        --gres)          GRES="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --conda-env)     CONDA_ENV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/"

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Submit one job ----
submit_job() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] sbatch --job-name=$job_name"
        echo "          CMD: $cmd"
        echo ""
        return
    fi

    echo "Submitting: $job_name"

    sbatch \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --gres="$GRES" \
        --cpus-per-task=1 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV

echo '=============================================='
echo 'Job: $job_name'
echo 'CMD: $cmd'
echo 'Python:' \$(which python)
echo 'Start:' \$(date)
echo '=============================================='

$cmd

echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
"
}

# ---- Define all 12 experiments ----
# Format: problem_arg|problem_key|yaml_path|prefix|extra_args
#
# problem_arg  = value passed to --problem
# problem_key  = human-readable key (for job naming / filtering)
# yaml_path    = --method_path value
# prefix       = --prefix value
# extra_args   = problem-specific solver/data arguments

EXPERIMENTS=(
    # --- Budget Allocation ---
    # 1) Constant sigma=0.001
    "budgetalloc|budgetalloc|openpto/config/models/perturb_s0001.yaml|bench_perturb_s0001|--solver neural --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 2) Linear decay 0.01->0.0001 + tanh
    "budgetalloc|budgetalloc|openpto/config/models/perturb_linsched_budget.yaml|tanh_schedule_perturb_linsched_budget|--solver neural --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"

    # --- Bipartite Matching ---
    # 3) Constant sigma=0.02
    "bipartitematching|bipartitematching|openpto/config/models/perturb_s002.yaml|bench_perturb_s002|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    # 4) Linear decay 0.02->0.001 + tanh
    "bipartitematching|bipartitematching|openpto/config/models/perturb_linsched_bipartite.yaml|tanh_schedule_perturb_linsched_bipartite|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"

    # --- Knapsack (gen) ---
    # 5) Constant sigma=0.01
    "knapsack|knapsack|openpto/config/models/perturb_s001.yaml|bench_perturb_s001|--solver gurobi --n_epochs 300 --gpu ${GPU_ID}"
    # 6) Linear decay 0.01->0.001 + tanh
    "knapsack|knapsack|openpto/config/models/perturb_linsched_knapsack.yaml|tanh_schedule_perturb_linsched_knapsack|--solver gurobi --n_epochs 300 --gpu ${GPU_ID}"

    # --- Knapsack Energy ---
    # 7) Constant sigma=0.01 (same yaml as knapsack)
    "knapsack|knapsack_energy|openpto/config/models/perturb_s001.yaml|bench_perturb_s001|--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    # 8) Linear decay 0.01->0.001 + tanh (same yaml as knapsack)
    "knapsack|knapsack_energy|openpto/config/models/perturb_linsched_knapsack.yaml|tanh_schedule_perturb_linsched_knapsack|--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    # 8b) Constant sigma=0.3, no activation
    "knapsack|knapsack_energy|openpto/config/models/perturb_s03.yaml|bench_perturb_s03|--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"

    # --- Portfolio ---
    # 9) Constant sigma=0.0025
    "portfolio|portfolio|openpto/config/models/perturb_s00025.yaml|bench_perturb_s00025|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 10) Linear decay 0.0025->0.0001 + tanh
    "portfolio|portfolio|openpto/config/models/perturb_linsched_portfolio.yaml|tanh_schedule_perturb_linsched_portfolio|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"

    # --- Portfolio (sigmoid activation) ---
    # 11) Constant sigma=0.1 + sigmoid
    "portfolio|portfolio|openpto/config/models/perturb_sigmoid_s01.yaml|sigmoid_perturb_sigmoid_s01|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 12) Constant sigma=0.01 + sigmoid
    "portfolio|portfolio|openpto/config/models/perturb_sigmoid_s001.yaml|sigmoid_perturb_sigmoid_s001|--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"

    # --- Energy (n_samples=1) ---
    # 13) Constant sigma=0.01, n_samples=1
    "energy|energy|openpto/config/models/perturb_energy_n1_s001.yaml|bench_perturb_energy_n1_s001|--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 14) Linear decay 0.01->0.001, n_samples=1 + tanh
    "energy|energy|openpto/config/models/perturb_linsched_energy.yaml|tanh_schedule_perturb_linsched_energy|--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
)

# ---- Main ----
echo "=============================================="
echo "Perturb Sigma Experiments — SLURM Submission"
echo "=============================================="
echo "Partition: $PARTITION  Time: $TIME  Mem: $MEM  GRES: $GRES"
if [[ -n "$PROBLEM_FILTER" ]]; then
    echo "Filter: $PROBLEM_FILTER"
fi
if $DRY_RUN; then
    echo "*** DRY RUN MODE ***"
fi
echo "=============================================="
echo ""

submitted=0
for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r problem_arg problem_key yaml_path prefix extra_args <<< "$entry"

    # Apply problem filter if set
    if [[ -n "$PROBLEM_FILTER" && "$problem_key" != "$PROBLEM_FILTER" ]]; then
        continue
    fi

    # Build job name from the prefix
    job_name="pco_${problem_key}_${prefix}"

    # Build command
    cmd="python rethink_exp/main_results.py --problem=${problem_arg} --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

    submit_job "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "Total experiments: $submitted"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "All jobs submitted. Check with: squeue -u \$USER"
fi
