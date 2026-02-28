#!/usr/bin/env bash
# =============================================================================
# SLURM Submission Script for Wave 3 Experiments
# =============================================================================
# Submits 6 perturb experiments with new sigma/lr/n_samples/patience combos:
#   - Budget: sigma=0.001 n=1 lr=5e-3, sigma=0.3 n=1 lr=5e-3
#   - Knapsack (gen): sigma=0.5 lr=5e-3 patience=100
#   - Bipartite: sigma=0.02 lr=5e-3 patience=300
#   - Portfolio: sigma=0.3 lr=5e-3, sigma=0.02 lr=5e-3 patience=100
#
# Usage:
#   bash shells/slurm/submit_wave3_experiments.sh              # submit all
#   bash shells/slurm/submit_wave3_experiments.sh --dry-run    # print only
#   bash shells/slurm/submit_wave3_experiments.sh --problem portfolio
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

# ---- Define Wave 3 experiments ----
# Format: problem_arg|problem_key|yaml_path|prefix|extra_args

EXPERIMENTS=(
    # ===================== Budget Allocation (n=1 variants) =====================
    # 1) sigma=0.001, n=1, lr=5e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_s0001_n1.yaml|bench_perturb_s0001_n1_lr5e3|--solver neural --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 2) sigma=0.3, n=1, lr=5e-3
    "budgetalloc|budgetalloc|openpto/config/models/perturb_s03_n1.yaml|bench_perturb_s03_n1_lr5e3|--solver neural --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Knapsack (gen) =====================
    # 3) sigma=0.5, lr=5e-3, patience=100
    "knapsack|knapsack|openpto/config/models/perturb_s05.yaml|bench_perturb_s05_lr5e3_p100|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"

    # ===================== Bipartite Matching =====================
    # 4) sigma=0.02, lr=5e-3, patience=300
    "bipartitematching|bipartitematching|openpto/config/models/perturb_s002.yaml|bench_perturb_s002_lr5e3_p300|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 300 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"

    # ===================== Portfolio =====================
    # 5) sigma=0.3, lr=5e-3
    "portfolio|portfolio|openpto/config/models/perturb_s03.yaml|bench_perturb_s03_lr5e3|--solver cvxpy --n_epochs 300 --lr 0.005 --gpu ${GPU_ID} --data_dir ${DIR}"
    # 6) sigma=0.02, lr=5e-3, patience=100
    "portfolio|portfolio|openpto/config/models/perturb_s002.yaml|bench_perturb_s002_lr5e3_p100|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
)

# ---- Main ----
echo "=============================================="
echo "Wave 3 Experiments — SLURM Submission"
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
