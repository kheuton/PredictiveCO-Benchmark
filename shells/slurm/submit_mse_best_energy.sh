#!/usr/bin/env bash
# ====================================================================
# Best-MSE sweep — energy problem, with --skip_solver_eval
# ====================================================================
# Replaces the slow energy jobs from submit_mse_best_sweep.sh.
# Those jobs called Gurobi every epoch for val eval (35 min/epoch!).
# With --skip_solver_eval, training uses val MSE for early stopping;
# Gurobi is only called once at the very end for final test eval.
# Expected walltime: ~2h (vs 4h+ previously).
#
# 20 jobs: 5 LRs × 4 batch configs
#
# Usage:
#   bash shells/slurm/submit_mse_best_energy.sh --dry-run
#   bash shells/slurm/submit_mse_best_energy.sh
#   bash shells/slurm/submit_mse_best_energy.sh --batch gd
#   bash shells/slurm/submit_mse_best_energy.sh --no-skip-completed
# ====================================================================

set -e

DRY_RUN=false
BATCH_FILTER=""
PARTITION="hugheslab,batch"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --batch)              BATCH_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Config (energy-specific)
# ====================================================================

LRS=(1e-3 5e-3 1e-2 5e-2 1e-1)
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

WALLTIME="2:00:00"

is_completed() {
    local prefix="$1"
    local out="$SCRIPT_DIR/saved_records/energy-energy/mse/${prefix}/results.npy"
    [[ -f "$out" ]]
}

n_submitted=0
n_skipped=0

submit_job() {
    local batch="$1" lr="$2"

    if [[ -n "$BATCH_FILTER" && "$batch" != "$BATCH_FILTER" ]]; then return; fi

    local prefix="mse_best_${batch}_lr${lr}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "  [skip] energy ${batch} lr=${lr} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local opt_name="${OPT_NAME[$batch]}"
    local bs_flag="${BATCH_SIZE_FLAG[$batch]}"
    local job_name="mse_best_energy_${batch}_lr${lr}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    local cmd="python rethink_exp/main_results.py \
        --problem energy \
        --opt_model mse \
        --solver gurobi \
        --opt_name ${opt_name} \
        --lr ${lr} \
        --n_epochs 300 \
        --instances 400 \
        --testinstances 200 \
        --seed 2023 \
        --prefix ${prefix} \
        --skip_solver_eval \
        --solver_valfreq 5 \
        ${bs_flag}"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${job_name}  (time=${WALLTIME})"
        echo "  output: saved_records/energy-energy/mse/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${WALLTIME}" \
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
# Main
# ====================================================================

echo "=== Best-MSE sweep — energy (--skip_solver_eval) ==="
echo "LRs:     ${LRS[*]}"
echo "Batches: ${BATCH_LABELS[*]}"
echo "Total:   $((${#LRS[@]} * ${#BATCH_LABELS[@]})) jobs"
echo "Partition: $PARTITION"
echo ""

for batch in "${BATCH_LABELS[@]}"; do
    for lr in "${LRS[@]}"; do
        submit_job "$batch" "$lr"
    done
done

echo ""
echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
