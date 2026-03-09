#!/usr/bin/env bash
# ====================================================================
# Knapsack benchmark-scale MSE (2-stage) baseline
# ====================================================================
# Benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model only (poly deprioritised). Gurobi solver. 3 LRs.
#
# Grid: 3 LRs = 3 jobs
# Time: ~70 min each (Gurobi, 300 epochs). Limit: 2h.
#
# Prefixes: saved_records/knapsack-gen/mse/kn_bench_mse_dense_lr{lr}/
#
# Usage:
#   bash shells/slurm/submit_kn_bench_mse.sh --dry-run
#   bash shells/slurm/submit_kn_bench_mse.sh
#   bash shells/slurm/submit_kn_bench_mse.sh --lr 1e-2
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
PARTITION="batch"
TIME="2:00:00"
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
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/mse/${prefix}/results.npy" ]]
}

submit_job() {
    local job_name="$1"
    local cmd="$2"

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
echo '=============================================='
echo 'Job: $job_name  Partition: $PARTITION'
echo 'Start:' \$(date)
echo '=============================================='
$cmd
echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
")
    echo "  $job_name → job $jid"
}

# ====================================================================
# Benchmark-scale args: loads Knapsack_7.pkl (320+80 train/val, 200 test)
BASE_ARGS="--problem knapsack --opt_model mse --solver gurobi \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--n_epochs 300 --gpu -1 --loadnew False"

LRS=("5e-2" "1e-2" "5e-3")

echo "======================================================================"
echo "Knapsack benchmark-scale MSE baseline (Knapsack_7: 320+80 train/val)"
echo "Dense model, Gurobi solver, 300 epochs, 3 LRs = 3 jobs"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]]; then echo "LR filter: $LR_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for lr in "${LRS[@]}"; do
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

    prefix="kn_bench_mse_dense_lr${lr}"
    job_name="kn_bench_mse_lr${lr//-/}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python rethink_exp/main_results.py $BASE_ARGS \
--pred_model dense --lr ${lr} --prefix ${prefix}"
    submit_job "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn_bench_mse"; fi
