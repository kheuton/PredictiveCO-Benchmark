#!/usr/bin/env bash
# ====================================================================
# Knapsack HammingProportionalSigmaPerturb sweep
# ====================================================================
# No-setpoint per-item sigma controller: sigma_i ∝ hamming_i^alpha.
# Removes the ballooning failure mode of all prior Hamming controllers
# by eliminating the feedback setpoint entirely.
#
# Sweep:
#   3 scales × 2 alphas × 2 LRs = 12 jobs
#
# Fixed hyperparams:
#   sigma_min=1e-4, n_samples=100, 300 epochs, DP solver
#   warmup=10 epochs, update_freq=20 epochs, sigma_ema=0.9
#
# Usage:
#   bash shells/slurm/submit_kn_hamming_prop_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_hamming_prop_sweep.sh
#   bash shells/slurm/submit_kn_hamming_prop_sweep.sh --alpha 0.5
# ====================================================================

set -e

DRY_RUN=false
ALPHA_FILTER=""
LR_FILTER=""
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --alpha)              ALPHA_FILTER="$2"; shift 2 ;;
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
    local alpha="$1"
    local scale="$2"
    local lr="$3"
    local prefix="kn_hamming_prop_a${alpha}_lr${lr}"
    local out_dir="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}"
    # Stem format: hprop_a{alpha}_s{scale}_s1_dense.npz
    # (alpha and scale appear in stem as set by build_sweep_grid)
    local stem="hprop_a${alpha}_s${scale}_s1"
    ls "${out_dir}/${stem}_dense.npz" 2>/dev/null | grep -q . && return 0
    return 1
}

# ---- Submit helper ----
n_submitted=0
n_skipped=0

submit_job() {
    local alpha="$1"
    local scale="$2"
    local lr="$3"
    local prefix="kn_hamming_prop_a${alpha}_lr${lr}"

    if [[ -n "$ALPHA_FILTER" && "$alpha" != "$ALPHA_FILTER" ]]; then
        return
    fi
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then
        return
    fi

    if $SKIP_COMPLETED && is_completed "$alpha" "$scale" "$lr"; then
        echo "  [skip] ${prefix} scale=${scale} — already completed"
        (( n_skipped++ )) || true
        return
    fi

    local job_name="kn_hprop_a${alpha}_s${scale}_lr${lr}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/perturb_adaptive_sweep.py \
        --problem knapsack \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --solver heuristic \
        --hamming_proportional \
        --hp_alpha ${alpha} \
        --hp_warmup 10 \
        --hp_update_freq 20 \
        --sigma_ema 0.9 \
        --prop_targets ${scale} \
        --sigma_min 1e-4 \
        --sigma_max 10.0 \
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
# Main sweep: 2 alphas × 3 scales × 2 LRs = 12 jobs
# ====================================================================

ALPHAS=("0.5" "1.0")
SCALES=("0.5" "1.0" "2.0")
LRS=("1e-2" "5e-3")

echo "=== HammingProportionalSigmaPerturb sweep ==="
echo "Alphas:  ${ALPHAS[*]}"
echo "Scales:  ${SCALES[*]}"
echo "LRs:     ${LRS[*]}"
echo ""

for alpha in "${ALPHAS[@]}"; do
    for scale in "${SCALES[@]}"; do
        for lr in "${LRS[@]}"; do
            submit_job "$alpha" "$scale" "$lr"
        done
    done
done

echo ""
echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
