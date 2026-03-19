#!/usr/bin/env bash
# ====================================================================
# Cubic: gradient surgery × constant sigma sweep
# ====================================================================
# Goal: find whether surgery + constant sigma can close the gap with
# MSE on cubic (MSE=0.0011; adaptive-sigma surgery=0.126).
#
# 5 sigma values × 2 conditions (surgery / no surgery) = 10 jobs
# Sigma range covers the Y cost scale (std≈1.55, range ±3.5):
#   0.1, 0.3, 1.0, 3.0, 10.0
#
# All jobs: --opt_name sgd --lr 1e-2 --n_samples 100 --n_epochs 300
# Surgery: --grad_surgery --surgery_weight 1.0
#
# Output: saved_records/cubic-gen/perturb/<prefix>/results.npy
#
# Usage:
#   bash shells/slurm/submit_cubic_surgery_sigma_sweep.sh --dry-run
#   bash shells/slurm/submit_cubic_surgery_sigma_sweep.sh
#   bash shells/slurm/submit_cubic_surgery_sigma_sweep.sh --sigma 1.0
#   bash shells/slurm/submit_cubic_surgery_sigma_sweep.sh --condition surgery
# ====================================================================

set -e

DRY_RUN=false
SIGMA_FILTER=""
COND_FILTER=""
LR="1e-2"
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --sigma)              SIGMA_FILTER="$2"; shift 2 ;;
        --condition)          COND_FILTER="$2"; shift 2 ;;
        --lr)                 LR="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# sigma_label → config file stem (perturb_cub_s<label>_n100.yaml)
# sigma 0.1 → s01, 0.3 → s03, 1.0 → s10, 3.0 → s30, 10.0 → s100
declare -A SIGMA_LABEL
SIGMA_LABEL[0.1]=01
SIGMA_LABEL[0.3]=03
SIGMA_LABEL[0.5]=05
SIGMA_LABEL[1.0]=10
SIGMA_LABEL[3.0]=30
SIGMA_LABEL[10.0]=100

SIGMAS=(0.1 0.3 0.5 1.0 3.0 10.0)
CONDITIONS=(plain surgery)   # plain = no surgery; surgery = --grad_surgery

# LR tag used in prefix/job names: 1e-2 → lr1e2, 5e-3 → lr5e3
lr_tag() { echo "$1" | tr -d '.e-' | sed 's/^0*//'; }
LR_TAG=$(echo "$LR" | sed 's/e-/e-/; s/\./p/g; s/-//2')
# simpler: just strip non-alphanumeric chars
LR_TAG=$(echo "$LR" | tr -d '.')

# ---- Completion check ----
is_completed() {
    local prefix="$1"
    local out="$SCRIPT_DIR/saved_records/cubic-gen/perturb/${prefix}/results.npy"
    [[ -f "$out" ]] && return 0
    return 1
}

# ---- Submit helper ----
n_submitted=0
n_skipped=0

submit_job() {
    local sigma="$1"
    local condition="$2"    # "plain" or "surgery"

    if [[ -n "$SIGMA_FILTER" && "$sigma" != "$SIGMA_FILTER" ]]; then return; fi
    if [[ -n "$COND_FILTER"  && "$condition" != "$COND_FILTER" ]]; then return; fi

    local label="${SIGMA_LABEL[$sigma]}"
    local prefix="cub_surg_s${label}_${condition}_lr${LR_TAG}"
    local config="openpto/config/models/perturb_cub_s${label}_n100.yaml"
    local job_name="cub_s${label}_${condition}_lr${LR_TAG}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "  [skip] ${prefix} — already completed"
        (( n_skipped++ )) || true
        return
    fi

    local surgery_flags=""
    if [[ "$condition" == "surgery" ]]; then
        surgery_flags="--grad_surgery --surgery_weight 1.0"
    fi

    local cmd="python rethink_exp/main_results.py \
        --problem cubic \
        --opt_model perturb \
        --solver heuristic \
        --opt_name sgd \
        --lr ${LR} \
        --n_epochs 300 \
        --n_ptr_epochs 0 \
        --instances 400 \
        --seed 2023 \
        --method_path ${config} \
        --prefix ${prefix} \
        ${surgery_flags}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  sigma=${sigma}  condition=${condition}"
        echo "  output: saved_records/cubic-gen/perturb/${prefix}/results.npy"
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
# Main sweep: 5 sigmas × 2 conditions = 10 jobs
# ====================================================================

echo "=== Cubic surgery × constant sigma sweep ==="
echo "Sigmas:     ${SIGMAS[*]}"
echo "Conditions: ${CONDITIONS[*]}  (surgery = --grad_surgery --surgery_weight 1.0)"
echo "LR:         ${LR}"
echo "Partition:  $PARTITION"
echo ""

for sigma in "${SIGMAS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        submit_job "$sigma" "$condition"
    done
done

echo ""
echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
