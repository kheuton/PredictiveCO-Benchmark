#!/usr/bin/env bash
# =============================================================================
# SLURM Submission Script for Learning-Rate Sweep Experiments
# =============================================================================
# Submits 24 perturb experiments across 6 problems, varying learning rates
# (1e-3 and 5e-3) on existing sigma configurations.
#
# Usage:
#   bash shells/slurm/submit_lr_experiments.sh              # submit all
#   bash shells/slurm/submit_lr_experiments.sh --dry-run     # print only
#   bash shells/slurm/submit_lr_experiments.sh --problem knapsack
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

# ---- Define all 24 experiments ----
# Format: problem_arg|problem_key|yaml_path|prefix|extra_args

EXPERIMENTS=(
    # ===================== Budget Allocation =====================
    # 1) Constant sigma=0.001, lr=1e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_s0001.yaml|bench_perturb_s0001_lr1e3|--solver neural --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 2) Constant sigma=0.001, lr=5e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_s0001.yaml|bench_perturb_s0001_lr5e3|--solver neural --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 3) Schedule 0.01->0.0001 + tanh, lr=1e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_linsched_budget.yaml|sched_budget_lr1e3|--solver neural --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 4) Schedule 0.01->0.0001 + tanh, lr=5e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_linsched_budget.yaml|sched_budget_lr5e3|--solver neural --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Bipartite Matching =====================
    # 5) Constant sigma=0.02, lr=1e-3
    "bipartitematching|bipartitematching|openpto/config/models/perturb_s002.yaml|bench_perturb_s002_lr1e3|--solver cvxpy --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    # 6) Schedule 0.02->0.001 + tanh, lr=1e-3
    "bipartitematching|bipartitematching|openpto/config/models/perturb_linsched_bipartite.yaml|sched_bipartite_lr1e3|--solver cvxpy --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    # 7) Constant sigma=0.5, lr=5e-3
    "bipartitematching|bipartitematching|openpto/config/models/perturb_s05.yaml|bench_perturb_s05_lr5e3|--solver cvxpy --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"

    # ===================== Knapsack (gen) =====================
    # 8) Constant sigma=0.01, lr=1e-3
    "knapsack|knapsack|openpto/config/models/perturb_s001.yaml|bench_perturb_s001_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID}"
    # 9) Constant sigma=0.01, lr=5e-3
    "knapsack|knapsack|openpto/config/models/perturb_s001.yaml|bench_perturb_s001_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID}"
    # 10) Schedule 0.01->0.001 + tanh, lr=1e-3
    "knapsack|knapsack|openpto/config/models/perturb_linsched_knapsack.yaml|sched_knapsack_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID}"
    # 11) Schedule 0.01->0.001 + tanh, lr=5e-3
    "knapsack|knapsack|openpto/config/models/perturb_linsched_knapsack.yaml|sched_knapsack_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID}"

    # ===================== Knapsack Energy =====================
    # 12) Constant sigma=0.01, lr=1e-3
    "knapsack|knapsack_energy|openpto/config/models/perturb_s001.yaml|bench_perturb_s001_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    # 13) Constant sigma=0.01, lr=5e-3
    "knapsack|knapsack_energy|openpto/config/models/perturb_s001.yaml|bench_perturb_s001_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    # 14) Schedule 0.01->0.001 + tanh, lr=1e-3
    "knapsack|knapsack_energy|openpto/config/models/perturb_linsched_knapsack.yaml|sched_knapsack_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    # 15) Schedule 0.01->0.001 + tanh, lr=5e-3
    "knapsack|knapsack_energy|openpto/config/models/perturb_linsched_knapsack.yaml|sched_knapsack_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"

    # ===================== Portfolio =====================
    # 16) Constant sigma=0.0025, lr=1e-3
    "portfolio|portfolio|openpto/config/models/perturb_s00025.yaml|bench_perturb_s00025_lr1e3|--solver cvxpy --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 17) Constant sigma=0.0025, lr=5e-3
    "portfolio|portfolio|openpto/config/models/perturb_s00025.yaml|bench_perturb_s00025_lr5e3|--solver cvxpy --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 18) Schedule 0.0025->0.0001 + tanh, lr=1e-3
    "portfolio|portfolio|openpto/config/models/perturb_linsched_portfolio.yaml|sched_portfolio_lr1e3|--solver cvxpy --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 19) Schedule 0.0025->0.0001 + tanh, lr=5e-3
    "portfolio|portfolio|openpto/config/models/perturb_linsched_portfolio.yaml|sched_portfolio_lr5e3|--solver cvxpy --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 20) Constant sigma=0.1 (no activation), lr=1e-3
    "portfolio|portfolio|openpto/config/models/perturb_s01.yaml|bench_perturb_s01_lr1e3|--solver cvxpy --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Energy (n_samples=1) =====================
    # 21) Constant sigma=0.01, n=1, lr=1e-3
    "energy|energy|openpto/config/models/perturb_energy_n1_s001.yaml|bench_energy_n1_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 22) Constant sigma=0.01, n=1, lr=5e-3
    "energy|energy|openpto/config/models/perturb_energy_n1_s001.yaml|bench_energy_n1_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 23) Schedule 0.01->0.001, n=1, tanh, lr=1e-3
    "energy|energy|openpto/config/models/perturb_linsched_energy.yaml|sched_energy_lr1e3|--solver gurobi --n_epochs 300 --lr 0.001 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 24) Schedule 0.01->0.001, n=1, tanh, lr=5e-3
    "energy|energy|openpto/config/models/perturb_linsched_energy.yaml|sched_energy_lr5e3|--solver gurobi --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
)

# ---- Main ----
echo "=============================================="
echo "LR Sweep Experiments — SLURM Submission"
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
