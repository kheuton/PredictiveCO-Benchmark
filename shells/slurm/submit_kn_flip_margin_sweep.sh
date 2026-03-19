#!/usr/bin/env bash
# ====================================================================
# Knapsack FlipMarginSigmaPerturb sweep
# ====================================================================
# FlipMarginSigmaPerturb: per-(instance, item) sigma guided by z* vs z0
# agreement, with regret-weighted blending and coeff-relative hard ceiling.
#
# Sweep:
#   3 base hamming targets × 3 LRs = 9 jobs  (stem: fm_t)
#
# Fixed hyperparams:
#   error_target=0.20, correct_target=0.01
#   regret_threshold=0.01, sigma_coeff_max=5.0
#   sigma_min=1e-4, n_samples=100, 300 epochs
#
# Usage:
#   bash shells/slurm/submit_kn_flip_margin_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_flip_margin_sweep.sh
#   bash shells/slurm/submit_kn_flip_margin_sweep.sh --lr 1e-2
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
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
    local prefix="$1"
    local target="$2"
    local lr="$3"
    local out_dir="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}"
    # Match: fm_t<target>_s1_dense.npz
    local stem="fm_t${target}_s1"
    ls "${out_dir}/${stem}_dense.npz" 2>/dev/null | grep -q . && return 0
    return 1
}

# ---- Submit helper ----
n_submitted=0
n_skipped=0

submit_job() {
    local lr="$1"
    local target="$2"
    local prefix="kn_flip_margin_lr${lr}"

    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then
        return
    fi

    if $SKIP_COMPLETED && is_completed "$prefix" "$target" "$lr"; then
        echo "  [skip] ${prefix} t=${target} — already completed"
        (( n_skipped++ )) || true
        return
    fi

    local job_name="kn_fm_lr${lr}_t${target}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/perturb_adaptive_sweep.py \
        --problem knapsack \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --solver heuristic \
        --flip_margin \
        --hamming_targets ${target} \
        --fm_error_target 0.20 \
        --fm_correct_target 0.01 \
        --fm_regret_threshold 0.01 \
        --fm_sigma_coeff_max 5.0 \
        --sigma_min 1e-4 \
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
# Main sweep: 3 targets × 3 LRs = 9 jobs
# ====================================================================

LRS=("5e-2" "1e-2" "5e-3")
TARGETS=("0.050" "0.100" "0.200")

echo "=== FlipMarginSigmaPerturb sweep ==="
echo "Targets: ${TARGETS[*]}"
echo "LRs:     ${LRS[*]}"
echo ""

for lr in "${LRS[@]}"; do
    for target in "${TARGETS[@]}"; do
        submit_job "$lr" "$target"
    done
done

echo ""
echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
