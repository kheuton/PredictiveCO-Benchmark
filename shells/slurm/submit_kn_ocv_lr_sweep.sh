#!/usr/bin/env bash
# ====================================================================
# Knapsack adaptive-sigma OCV_Y sweep — 3 LRs × 15 targets × 2 models
# ====================================================================
# Submits one SLURM job per LR.  Each job runs all 15 OCV_Y targets ×
# 2 prediction models (dense, poly) sequentially (~5 hrs per job).
# Uses the DP solver (CPU-only — no GPU requested).
#
# Usage:
#   bash shells/slurm/submit_kn_ocv_lr_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_ocv_lr_sweep.sh
#   bash shells/slurm/submit_kn_ocv_lr_sweep.sh --lr 1e-2      # single LR
#   bash shells/slurm/submit_kn_ocv_lr_sweep.sh --no-backup
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
PARTITION="batch"
BACKUP_PARTITION=""
USE_BACKUP=false
TIME="8:00:00"
MEM="16G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --backup-partition)   BACKUP_PARTITION="$2"; shift 2 ;;
        --no-backup)          USE_BACKUP=false; shift ;;
        --time)               TIME="$2"; shift 2 ;;
        --mem)                MEM="$2"; shift 2 ;;
        --conda-env)          CONDA_ENV="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# Check if a sweep run is complete: all 30 expected .npz files exist.
is_completed() {
    local prefix="$1"
    local out_dir="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}"
    [[ -d "$out_dir" ]] && \
        [[ "$(ls "$out_dir"/*.npz 2>/dev/null | wc -l)" -ge 30 ]]
}

submit_job() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  partition: $PARTITION"
        if $USE_BACKUP; then echo "  backup:    $BACKUP_PARTITION (afternotok)"; fi
        echo "  CMD: $cmd"
        echo ""
        return
    fi

    local primary_jid
    primary_jid=$(sbatch --parsable \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --cpus-per-task=4 \
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
    echo "  $job_name → $PARTITION (job $primary_jid)"

    if $USE_BACKUP && [[ -n "$BACKUP_PARTITION" ]]; then
        local backup_jid
        backup_jid=$(sbatch --parsable \
            --job-name="${job_name}" \
            --partition="$BACKUP_PARTITION" \
            --time="$TIME" \
            --mem="$MEM" \
            --cpus-per-task=4 \
            --output="$SCRIPT_DIR/logs/slurm/${job_name}_bk_%j.out" \
            --error="$SCRIPT_DIR/logs/slurm/${job_name}_bk_%j.err" \
            --dependency="afternotok:${primary_jid}" \
            --kill-on-invalid-dep=yes \
            --export=ALL \
            --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV
echo '=============================================='
echo 'Job: $job_name  Partition: $BACKUP_PARTITION (backup)'
echo 'Start:' \$(date)
echo '=============================================='
$cmd
echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
")
        echo "    ↳ backup → $BACKUP_PARTITION (job $backup_jid, afternotok:$primary_jid)"
    fi
}

# ====================================================================
# Sweep jobs — one per LR
# ====================================================================

BASE_ARGS="--problem knapsack --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--mode proportional \
--n_epochs 150 --n_samples 100 \
--pred_models dense poly \
--gpu -1"

LRS=("5e-2" "1e-2" "5e-3")

echo "======================================================================"
echo "Knapsack OCV_Y adaptive-sigma sweep — 3 LRs"
echo "15 targets × 2 models × 150 epochs per job (~5 hrs each)"
echo "======================================================================"
echo "Primary:  $PARTITION"
if $USE_BACKUP; then echo "Backup:   $BACKUP_PARTITION (afternotok)"; fi
echo "Time: $TIME  Mem: $MEM  CPUs: 4"
if [[ -n "$LR_FILTER" ]]; then echo "LR filter: $LR_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON (checks for 30 .npz files)"; fi
echo "======================================================================"
echo ""

submitted=0
skipped=0

for lr in "${LRS[@]}"; do
    if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then
        continue
    fi

    prefix="kn_ocv_lr${lr}"
    job_name="kn_ocv_lr${lr//-/}"   # remove dashes for SLURM name

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed (30 .npz files found)"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python -u rethink_exp/perturb_adaptive_sweep.py $BASE_ARGS --lr $lr --prefix $prefix"
    submit_job "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if $USE_BACKUP; then echo "Total SLURM jobs: $((submitted * 2)) (primary + backup)"; fi
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
echo ""
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Monitor with: squeue -u \$USER"
fi
