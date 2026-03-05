#!/usr/bin/env bash
# ====================================================================
# Wave 5 — Portfolio: σ=0 (exact CvxpyLayer gradients, no perturbation)
# ====================================================================
# Grid (portfolio only):
#   sigma = 0 (direct differentiation through QP solver)
#   lr    ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
#   pred_loss_weight = 0
#
# 5 experiments total
#
# Usage:
#   bash shells/slurm/submit_portfolio_wave5.sh --dry-run
#   bash shells/slurm/submit_portfolio_wave5.sh
#   bash shells/slurm/submit_portfolio_wave5.sh --no-backup
# ====================================================================

set -e

# ---- Defaults ----
DRY_RUN=false
PARTITION="preempt"
BACKUP_PARTITION="hugheslab,gpu"
USE_BACKUP=true
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                DRY_RUN=true; shift ;;
        --partition)              PARTITION="$2"; shift 2 ;;
        --backup-partition)       BACKUP_PARTITION="$2"; shift 2 ;;
        --no-backup)              USE_BACKUP=false; shift ;;
        --time)                   TIME="$2"; shift 2 ;;
        --mem)                    MEM="$2"; shift 2 ;;
        --gres)                   GRES="$2"; shift 2 ;;
        --gpu)                    GPU_ID="$2"; shift 2 ;;
        --conda-env)              CONDA_ENV="$2"; shift 2 ;;
        --no-skip-completed)      SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/"
CFGDIR="openpto/config/models"
YAML="${CFGDIR}/perturb_sigma0.yaml"

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Check if experiment already completed ----
is_completed() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/portfolio-real/perturb/${prefix}/results.npy" ]]
}

# ---- Submit one experiment (primary + optional backup) ----
submit_experiment() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
        echo ""
        return
    fi

    # Submit primary job
    local primary_jid
    primary_jid=$(sbatch --parsable \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --gres="$GRES" \
        --cpus-per-task=1 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV

echo '=============================================='
echo 'Job: $job_name  Partition: $PARTITION'
echo 'CMD: $cmd'
echo 'Start:' \$(date)
echo '=============================================='

$cmd

echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
")

    echo "  $job_name → $PARTITION (job $primary_jid)"

    # Submit backup with dependency
    if $USE_BACKUP && [[ -n "$BACKUP_PARTITION" ]]; then
        local backup_jid
        backup_jid=$(sbatch --parsable \
            --job-name="${job_name}" \
            --partition="$BACKUP_PARTITION" \
            --time="$TIME" \
            --mem="$MEM" \
            --gres="$GRES" \
            --cpus-per-task=1 \
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
echo 'CMD: $cmd'
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
# Experiment grid
# ====================================================================

EXPERIMENTS=()

declare -A LR_TAG
LR_TAG[0.0001]="lr1e4"
LR_TAG[0.0005]="lr5e4"
LR_TAG[0.001]="lr1e3"
LR_TAG[0.005]="lr5e3"
LR_TAG[0.01]="lr1e2"

LRS=("0.0001" "0.0005" "0.001" "0.005" "0.01")

N_EPOCHS=400
PATIENCE=80

# ----- Build experiment list -----
for lr in "${LRS[@]}"; do
    lr_tag="${LR_TAG[$lr]}"
    prefix="sd1_w5_s0_${lr_tag}"

    cmd_extra="--solver cvxpy --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr ${lr} --data_dir ${DIR} --gpu ${GPU_ID}"
    EXPERIMENTS+=("${YAML}|${prefix}|${cmd_extra}")
done

# ---- Main ----
echo "=============================================="
echo "Wave 5 — Portfolio: σ=0 (exact CvxpyLayer gradients)"
echo "Grid: 5 lr values = ${#EXPERIMENTS[@]} experiments"
echo "Schedule: n_epochs=${N_EPOCHS}, patience=${PATIENCE}"
echo "=============================================="
echo "Primary:  $PARTITION"
if $USE_BACKUP; then
    echo "Backup:   $BACKUP_PARTITION (afternotok dependency)"
else
    echo "Backup:   disabled"
fi
echo "Time: $TIME  Mem: $MEM  GRES: $GRES"
if $DRY_RUN; then echo "*** DRY RUN MODE ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "=============================================="
echo ""

submitted=0
skipped=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r yaml_path prefix extra_args <<< "$entry"

    job_name="pco_portfolio_${prefix}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python rethink_exp/main_results.py --problem=portfolio --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

    submit_experiment "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "=============================================="
echo "Submitted: $submitted experiments"
if $USE_BACKUP; then
    echo "  → $submitted primary ($PARTITION) + $submitted backup ($BACKUP_PARTITION)"
    echo "  → Total SLURM jobs: $((submitted * 2))"
fi
if [[ $skipped -gt 0 ]]; then
    echo "Skipped (completed): $skipped"
fi
echo ""
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Monitor with: squeue -u \$USER | grep pco_portfolio_sd1_w5"
fi
