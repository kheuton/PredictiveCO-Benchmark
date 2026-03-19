#!/usr/bin/env bash
# ====================================================================
# Diagnostic re-runs of per-item sigma variants A and C
# ====================================================================
# Same as the original per-item sweep but:
#   - 150 epochs (enough to see sigma ballooning)
#   - Prefixes: kn_per_item_diag_{A,C}_lr{lr}
#   - Outputs include sigma_vec_item (n_epochs, D) and hamming_vec_item
#     for the per-item heatmap figure.
#
# Variants: A (per-item) and C (per-inst-item), sigma_min=1e-4
# Targets: 0.050, 0.100, 0.200
# LRs: 1e-2 (best from original sweep)
#
# Total: 2 variants × 3 targets × 1 LR = 6 jobs
#
# Usage:
#   bash shells/slurm/submit_kn_per_item_diag.sh --dry-run
#   bash shells/slurm/submit_kn_per_item_diag.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="hugheslab,batch"
TIME="0:45:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

is_completed() {
    local prefix="$1"
    local stem="$2"
    local out_dir="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}"
    ls "${out_dir}/${stem}_dense.npz" 2>/dev/null | grep -q . && return 0
    return 1
}

n_submitted=0
n_skipped=0

submit_job() {
    local variant="$1"   # A or C
    local target="$2"
    local lr="$3"
    local flag stem prefix

    if [[ "$variant" == "A" ]]; then
        flag="--per_item"
        stem="pitem_t${target}_s1"
    else
        flag="--per_inst_item"
        stem="piitem_t${target}_s1"
    fi
    prefix="kn_per_item_diag_${variant}_lr${lr}"

    if $SKIP_COMPLETED && is_completed "$prefix" "$stem"; then
        echo "  [skip] ${prefix}/${stem} — already done"
        (( n_skipped++ )) || true
        return
    fi

    local job_name="kn_pitem_diag_${variant}_lr${lr}_t${target}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/perturb_adaptive_sweep.py \
        --problem knapsack \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --solver heuristic \
        ${flag} \
        --hamming_targets ${target} \
        --sigma_min 1e-4 \
        --n_samples 100 \
        --n_epochs 150 \
        --lr ${lr} \
        --pred_models dense \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${job_name}"
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

LR="1e-2"
TARGETS=("0.050" "0.100" "0.200")

echo "=== Per-item diagnostic re-runs (A + C, lr=${LR}, 150 epochs) ==="
echo ""

for target in "${TARGETS[@]}"; do
    submit_job "A" "$target" "$LR"
    submit_job "C" "$target" "$LR"
done

echo ""
echo "Submitted: $n_submitted   Skipped: $n_skipped"
