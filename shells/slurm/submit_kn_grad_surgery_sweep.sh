#!/usr/bin/env bash
# ====================================================================
# Knapsack gradient surgery sweep
# ====================================================================
# Gradient surgery at each step:
#   g_update = (g_p - proj_{g_m}(g_p)) + surgery_weight * g_m
# where g_p = perturbed gradient, g_m = MSE gradient w.r.t. predictions.
#
# This removes the MSE-misaligned component of the perturbed gradient
# and injects a clean MSE direction, directly addressing the 99.1%
# orthogonality finding. Differs from λ-MSE reg: surgery first removes
# the corrupted portion of g_p before adding g_m.
#
# Sweep:
#   3 surgery_weights × 2 LRs = 6 jobs
#
# Fixed hyperparams:
#   sigma=0.5, n_samples=100, 300 epochs, DP solver, dense model
#
# Usage:
#   bash shells/slurm/submit_kn_grad_surgery_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_grad_surgery_sweep.sh
#   bash shells/slurm/submit_kn_grad_surgery_sweep.sh --weight 1.0
# ====================================================================

set -e

DRY_RUN=false
WEIGHT_FILTER=""
LR_FILTER=""
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --weight)             WEIGHT_FILTER="$2"; shift 2 ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        --time)               TIME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Completion check ----
is_completed() {
    local weight="$1"
    local lr="$2"
    local prefix="kn_grad_surgery_w${weight}_lr${lr}"
    local out_dir="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}"
    # In proportional mode with no explicit targets, stem = prop_t{ocv_target}_s{sigma_init}
    # For surgery runs we set --prop_targets 1.0 → stem = prop_t1.0_s1
    ls "${out_dir}/prop_t1.0_s1_dense.npz" 2>/dev/null | grep -q . && return 0
    return 1
}

# ---- Submit helper ----
n_submitted=0
n_skipped=0

submit_job() {
    local weight="$1"
    local lr="$2"
    local prefix="kn_grad_surgery_w${weight}_lr${lr}"

    if [[ -n "$WEIGHT_FILTER" && "$weight" != "$WEIGHT_FILTER" ]]; then
        return
    fi
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then
        return
    fi

    if $SKIP_COMPLETED && is_completed "$weight" "$lr"; then
        echo "  [skip] ${prefix} — already completed"
        (( n_skipped++ )) || true
        return
    fi

    local job_name="kn_surgery_w${weight}_lr${lr}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/perturb_adaptive_sweep.py \
        --problem knapsack \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --solver heuristic \
        --grad_surgery \
        --surgery_weight ${weight} \
        --prop_targets 1.0 \
        --prop_sigma_init 0.5 \
        --sigma_min 1e-4 \
        --sigma_max 1e3 \
        --n_samples 100 \
        --n_epochs 300 \
        --lr ${lr} \
        --pred_models dense \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}"
        echo "  cmd: ${cmd}"
        return
    fi

    sbatch \
        --job-name="$job_name" \
        --output="$log_file" \
        --partition="$PARTITION" \
        --cpus-per-task=2 \
        --mem="$MEM" \
        --time="$TIME" \
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
# Main sweep: 3 surgery_weights × 2 LRs = 6 jobs
# ====================================================================

WEIGHTS=("0.5" "1.0" "2.0")
LRS=("1e-2" "5e-3")

echo "=== Gradient surgery sweep ==="
echo "Surgery weights: ${WEIGHTS[*]}"
echo "LRs:             ${LRS[*]}"
echo ""

for weight in "${WEIGHTS[@]}"; do
    for lr in "${LRS[@]}"; do
        submit_job "$weight" "$lr"
    done
done

echo ""
echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
