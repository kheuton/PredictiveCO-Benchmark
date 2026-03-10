#!/usr/bin/env bash
# ====================================================================
# Knapsack benchmark-scale Hamming-Star (capped) controller sweep
# ====================================================================
# Per-instance dynamic target: target_i = min(hamming(z*_i,z0_i)/D, cap)
# Cap = hamming_target (inherited). Self-terminates per-instance as z0→z*.
#
# Scale: 320+80 train/val, 200 test (benchmark default)
# Grid:  4 caps × 3 LRs × 2 models = 24 jobs  (fully parallel)
# Solver: DP (heuristic), n_samples=100, 300 epochs, ~45 min/run
#
# Prefixes: kn_hstar_lr{lr}/
# Files:    hams_s{cap}_s1_{dense,poly}.npz
#
# Usage:
#   bash shells/slurm/submit_kn_hstar_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_hstar_sweep.sh
#   bash shells/slurm/submit_kn_hstar_sweep.sh --lr 1e-2
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
PARTITION="batch"
TIME="1:30:00"
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

# Filename stem: hams_s{cap}_s{sigma_init} where sigma_init=1 and cap is formatted as %.3f
cap_to_stem() {
    # e.g. 0.050 -> hams_s0.050_s1
    printf "hams_s%s_s1" "$1"
}

is_completed() {
    local prefix="$1" cap="$2" model="$3"
    local stem
    stem=$(cap_to_stem "$cap")
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}/${stem}_${model}.npz" ]]
}

submit_job() {
    local job_name="$1" cmd="$2"
    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
        echo ""
        return
    fi
    local jid
    jid=$(sbatch --parsable \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --cpus-per-task=2 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV
echo 'Job: $job_name  Start:' \$(date)
$cmd
echo 'End:' \$(date)
")
    echo "  $job_name → job $jid"
}

# ====================================================================
CAPS=("0.050" "0.100" "0.150" "0.200")
LRS=("5e-2" "1e-2" "5e-3")
MODELS=("dense" "poly")

BASE_ARGS="--problem knapsack \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--mode proportional --gpu -1 \
--solver heuristic --n_samples 100 --n_epochs 300 \
--per_instance --hamming_star"

echo "======================================================================"
echo "Knapsack benchmark-scale Hamming-Star (capped) sweep"
echo "4 caps × 3 LRs × 2 models = 24 jobs  |  DP, n=100, 300ep, ~45 min/run"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]]; then echo "LR filter: $LR_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for lr in "${LRS[@]}"; do
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

    prefix="kn_hstar_lr${lr}"

    for cap in "${CAPS[@]}"; do
        for model in "${MODELS[@]}"; do
            job_name="knb_hstar_lr${lr//-/}_c${cap//./_}_${model}"

            if $SKIP_COMPLETED && is_completed "$prefix" "$cap" "$model"; then
                echo "[SKIP] $job_name — already completed"
                skipped=$((skipped + 1))
                continue
            fi

            cmd="python -u rethink_exp/perturb_adaptive_sweep.py $BASE_ARGS \
--lr ${lr} --hamming_targets ${cap} --pred_models ${model} --prefix ${prefix}"
            submit_job "$job_name" "$cmd"
            submitted=$((submitted + 1))
        done
    done
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep knb_hstar"; fi
