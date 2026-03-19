#!/usr/bin/env bash
# ====================================================================
# Best-MSE sweep — all problems
# ====================================================================
# Find the best MSE regret achievable on every benchmark problem by
# sweeping 5 learning rates × 4 batch configs (full-batch + 3 minibatch).
#
# Goal: establish the strongest possible MSE baseline to support the
# paper message that MSE is near-optimal for linear-objective problems.
#
# Batch configs:
#   gd          -- full-batch gradient descent (--opt_name gd)
#   bs32/64/128 -- minibatch SGD (--opt_name sgd --batch_size N)
#
# Learning rates: 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
#
# Total: 7 problems × 5 LRs × 4 batch configs = 140 jobs
#
# Solvers:
#   knapsack (gen): heuristic (DP, best evaluation accuracy)
#   knapsack-real:  gurobi
#   energy:         gurobi
#   budgetalloc:    neural
#   cubic:          heuristic
#   bipartitematching: cvxpy
#   portfolio:      cvxpy
#
# Usage:
#   bash shells/slurm/submit_mse_best_sweep.sh --dry-run
#   bash shells/slurm/submit_mse_best_sweep.sh
#   bash shells/slurm/submit_mse_best_sweep.sh --problem knapsack
#   bash shells/slurm/submit_mse_best_sweep.sh --batch gd
#   bash shells/slurm/submit_mse_best_sweep.sh --problem energy --batch bs32
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
BATCH_FILTER=""
PARTITION="hugheslab,batch"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --problem)            PROB_FILTER="$2"; shift 2 ;;
        --batch)              BATCH_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Per-problem configuration (mirrors submit_surgery_sweep_all.sh)
# ====================================================================

PROBLEMS=(knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio)

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[knapsack-real]=energy
PROB_VERSION[energy]=energy
PROB_VERSION[budgetalloc]=real
PROB_VERSION[cubic]=gen
PROB_VERSION[bipartitematching]=cora
PROB_VERSION[portfolio]=real

declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[knapsack-real]=knapsack
PROB_ARG[energy]=energy
PROB_ARG[budgetalloc]=budgetalloc
PROB_ARG[cubic]=cubic
PROB_ARG[bipartitematching]=bipartitematching
PROB_ARG[portfolio]=portfolio

declare -A MSE_SOLVER
MSE_SOLVER[knapsack]=heuristic      # DP solver: better eval accuracy than gurobi
MSE_SOLVER[knapsack-real]=gurobi
MSE_SOLVER[energy]=gurobi
MSE_SOLVER[budgetalloc]=neural
MSE_SOLVER[cubic]=heuristic
MSE_SOLVER[bipartitematching]=cvxpy
MSE_SOLVER[portfolio]=cvxpy

declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[knapsack-real]=openpto/config/probs/knapsack-real.yaml
PROB_CONFIG[energy]=""
PROB_CONFIG[budgetalloc]=""
PROB_CONFIG[cubic]=""
PROB_CONFIG[bipartitematching]=""
PROB_CONFIG[portfolio]=""

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[knapsack-real]=400
INSTANCES[energy]=400
INSTANCES[budgetalloc]=400
INSTANCES[cubic]=250
INSTANCES[bipartitematching]=20
INSTANCES[portfolio]=400

declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[knapsack-real]=200
TESTINSTANCES[energy]=200
TESTINSTANCES[budgetalloc]=200
TESTINSTANCES[cubic]=400
TESTINSTANCES[bipartitematching]=6
TESTINSTANCES[portfolio]=200

declare -A WALLTIME
WALLTIME[knapsack]=1:00:00
WALLTIME[knapsack-real]=1:30:00
WALLTIME[energy]=4:00:00
WALLTIME[budgetalloc]=1:00:00
WALLTIME[cubic]=1:00:00
WALLTIME[bipartitematching]=1:00:00
WALLTIME[portfolio]=1:00:00

# ====================================================================
# Sweep parameters
# ====================================================================

LRS=(1e-3 5e-3 1e-2 5e-2 1e-1)

# Batch configs: label → (opt_name, batch_size_flag)
BATCH_LABELS=(gd bs32 bs64 bs128)
declare -A OPT_NAME
OPT_NAME[gd]=gd
OPT_NAME[bs32]=sgd
OPT_NAME[bs64]=sgd
OPT_NAME[bs128]=sgd

declare -A BATCH_SIZE_FLAG
BATCH_SIZE_FLAG[gd]=""
BATCH_SIZE_FLAG[bs32]="--batch_size 32"
BATCH_SIZE_FLAG[bs64]="--batch_size 64"
BATCH_SIZE_FLAG[bs128]="--batch_size 128"

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

# Prefix: mse_best_{batch}_{lr}  e.g. mse_best_gd_lr5e2, mse_best_bs32_lr1e2
make_prefix() {
    local batch="$1" lr="$2"
    # replace 'e' → 'e', remove decimal points for cleaner names
    local lr_tag="${lr//./_}"
    echo "mse_best_${batch}_lr${lr}"
}

is_completed() {
    local prob="$1" prefix="$2"
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/mse/${prefix}/results.npy"
    [[ -f "$out" ]]
}

n_submitted=0
n_skipped=0

submit_job() {
    local prob="$1" batch="$2" lr="$3"

    if [[ -n "$PROB_FILTER" && "$prob" != "$PROB_FILTER" ]]; then return; fi
    if [[ -n "$BATCH_FILTER" && "$batch" != "$BATCH_FILTER" ]]; then return; fi

    local prefix
    prefix=$(make_prefix "$batch" "$lr")

    if $SKIP_COMPLETED && is_completed "$prob" "$prefix"; then
        echo "  [skip] ${prob} ${batch} lr=${lr} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local prob_arg="${PROB_ARG[$prob]}"
    local solver="${MSE_SOLVER[$prob]}"
    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local walltime="${WALLTIME[$prob]}"
    local opt_name="${OPT_NAME[$batch]}"
    local bs_flag="${BATCH_SIZE_FLAG[$batch]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"

    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local job_name="mse_best_${prob}_${batch}_lr${lr}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    local cmd="python rethink_exp/main_results.py \
        --problem ${prob_arg} \
        --opt_model mse \
        --solver ${solver} \
        --opt_name ${opt_name} \
        --lr ${lr} \
        --n_epochs 300 \
        --instances ${instances} \
        --testinstances ${testinstances} \
        --seed 2023 \
        --prefix ${prefix} \
        ${bs_flag} \
        ${cfg_flag}"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${job_name}  (time=${walltime})"
        echo "  output: saved_records/$(prob_out_dir $prob)/mse/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${walltime}" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

# ====================================================================
# Main sweep: 7 problems × 5 LRs × 4 batch configs = 140 jobs
# ====================================================================

echo "=== Best-MSE sweep — all problems ==="
echo "Problems: ${PROBLEMS[*]}"
echo "LRs:      ${LRS[*]}"
echo "Batches:  ${BATCH_LABELS[*]}"
echo "Total:    $((${#PROBLEMS[@]} * ${#LRS[@]} * ${#BATCH_LABELS[@]})) jobs"
echo "Partition: $PARTITION"
echo ""

for prob in "${PROBLEMS[@]}"; do
    echo "--- ${prob} ---"
    for batch in "${BATCH_LABELS[@]}"; do
        for lr in "${LRS[@]}"; do
            submit_job "$prob" "$batch" "$lr"
        done
    done
    echo ""
done

echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
