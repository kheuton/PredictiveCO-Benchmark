#!/usr/bin/env bash
# =============================================================================
# Wave 2 — Cubic Sigma Annealing + Tanh Activation
# =============================================================================
# Grid (cubic only):
#   sigma range ∈ {0.1→0.01, 0.1→0.001, 0.05→0.005, 0.05→0.001, 0.5→0.01}
#   warmup      ∈ {0, 10, 30} epochs
#   lr          ∈ {1e-3, 5e-3, 1e-2}
#   activation  ∈ {none, tanh}
#   n_samples   = 25,  schedule = linear_decay
#   n_epochs    = 400,  patience = 80
#
# 5 ranges × 3 warmups × 3 LRs × 2 activations = 90 experiments (~22 min total)
#
# Usage:
#   bash shells/slurm/submit_cubic_wave2.sh --dry-run
#   bash shells/slurm/submit_cubic_wave2.sh
#   bash shells/slurm/submit_cubic_wave2.sh --no-backup
# =============================================================================

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
CFGDIR="openpto/config/models"

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Check if experiment already completed ----
is_completed() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/cubic-gen/perturb/${prefix}/results.npy" ]]
}

# ---- Submit one experiment (primary + optional backup) ----
submit_experiment() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  primary:  $PARTITION"
        if $USE_BACKUP; then
            echo "  backup:   $BACKUP_PARTITION (afternotok)"
        fi
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

# =============================================================================
# Experiment grid — Cubic × linear_decay schedules × activation
# =============================================================================

EXPERIMENTS=()

# LR → tag mapping
declare -A LR_TAG
LR_TAG[0.001]="lr1e3"
LR_TAG[0.005]="lr5e3"
LR_TAG[0.01]="lr1e2"

# Sigma ranges (tuned for cubic's sweet spot)
SIGMA_RANGES=("s01to001" "s01to0001" "s005to0005" "s005to0001" "s05to001")
WARMUPS=("0" "10" "30")
LRS=("0.001" "0.005" "0.01")
ACTIVATIONS=("none" "tanh")

N_EPOCHS=400
PATIENCE=80

# ----- Build experiment list -----
for range_tag in "${SIGMA_RANGES[@]}"; do
    for warmup in "${WARMUPS[@]}"; do
        for act in "${ACTIVATIONS[@]}"; do
            if [[ "$act" == "none" ]]; then
                act_suffix=""
                act_prefix=""
            else
                act_suffix="_tanh"
                act_prefix="tanh_"
            fi

            yaml="${CFGDIR}/perturb_cub_lin_${range_tag}_w${warmup}${act_suffix}.yaml"

            # Verify YAML exists
            if [[ ! -f "$yaml" ]]; then
                echo "WARNING: Missing config $yaml — skipping"
                continue
            fi

            for lr in "${LRS[@]}"; do
                lr_tag="${LR_TAG[$lr]}"
                prefix="sd1_cub_lin_${range_tag}_w${warmup}_${act_prefix}${lr_tag}"

                cmd_extra="--solver heuristic --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr ${lr} --instances 250 --testinstances 400 --gpu ${GPU_ID}"
                EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")
            done
        done
    done
done

# ---- Main ----
echo "=============================================="
echo "Wave 2 — Cubic Sigma Annealing + Tanh"
echo "Grid: 5 ranges × 3 warmups × 3 LRs × 2 activations = 90 experiments"
echo "Schedule: linear_decay, n_samples=25, n_epochs=${N_EPOCHS}, patience=${PATIENCE}"
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

    # Build job name
    job_name="pco_cubic_${prefix}"

    # Skip completed
    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    # Build command
    cmd="python rethink_exp/main_results.py --problem=cubic --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

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
    echo "Monitor with: squeue -u \$USER | grep pco_cubic"
fi
