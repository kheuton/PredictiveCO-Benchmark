#!/usr/bin/env bash
# ====================================================================
# Knapsack benchmark-scale Hamming-rate controller with target decay
# ====================================================================
# Same 5 targets × 3 LRs as the constant-target Hamming sweep, but:
#   - Epochs   1-100: constant Hamming target (warmup)
#   - Epochs 101-300: linear decay of target to 0
#
# Scale: 320+80 train/val, 200 test (Knapsack_7.pkl)
# Grid:  5 targets × 3 LRs = 15 jobs
# Solver: DP (heuristic), n_samples=100, 300 epochs, ~18 min/run
#
# Prefixes: kn_bench_hamming_decay_lr{lr}/
# Files:    hamd_t{target}_s1_dense.npz
#
# Usage:
#   bash shells/slurm/submit_kn_bench_hamming_decay.sh --dry-run
#   bash shells/slurm/submit_kn_bench_hamming_decay.sh
#   bash shells/slurm/submit_kn_bench_hamming_decay.sh --lr 1e-2
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
PARTITION="batch"
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

is_completed() {
    local prefix="$1" target="$2"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}/hamd_t${target}_s1_dense.npz" ]]
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
TARGETS=("0.050" "0.100" "0.150" "0.200" "0.300")
LRS=("5e-2" "1e-2" "5e-3")

BASE_ARGS="--problem knapsack \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--mode proportional --gpu -1 \
--solver heuristic --n_samples 100 --n_epochs 300 \
--pred_models dense \
--hamming \
--hamming_warmup_epochs 100 \
--hamming_decay_epochs 200"

echo "======================================================================"
echo "Knapsack benchmark-scale Hamming-rate sweep with target decay"
echo "  100 ep warmup → 200 ep linear decay to 0"
echo "  5 targets × 3 LRs = 15 jobs  |  DP, n=100, 300ep, ~18 min/run"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]]; then echo "LR filter: $LR_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for lr in "${LRS[@]}"; do
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

    prefix="kn_bench_hamming_decay_lr${lr}"

    for target in "${TARGETS[@]}"; do
        job_name="knb_hamd_lr${lr//-/}_t${target//./_}"

        if $SKIP_COMPLETED && is_completed "$prefix" "$target"; then
            echo "[SKIP] $job_name — already completed"
            skipped=$((skipped + 1))
            continue
        fi

        cmd="python -u rethink_exp/perturb_adaptive_sweep.py $BASE_ARGS \
--lr ${lr} --hamming_targets ${target} --prefix ${prefix}"
        submit_job "$job_name" "$cmd"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep knb_hamd"; fi
