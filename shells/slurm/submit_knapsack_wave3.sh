#!/usr/bin/env bash
# ====================================================================
# Wave 3 — Knapsack: Aggressive σ annealing (0.5 → small) over 100 epochs
# ====================================================================
# Rationale:
#   W1 best (σ=0.5, n=25) plateaus around ep 100 at val_regret ≈ 5.7,
#   then slowly improves to 4.375 by ep 215. Hypothesis: high σ enables
#   good exploration early, but prevents fine-tuning later. Aggressively
#   decaying σ after a warmup phase should yield exploration → exploitation.
#
# Grid:
#   σ_end             ∈ {0.05, 0.01}
#   warmup            ∈ {50, 100}  epochs holding σ=0.5
#   lr                ∈ {1e-3, 5e-3}
#   --- fixed ---
#   σ_start = 0.5, n_samples = 5, schedule = linear_decay
#   decay window = 100 epochs (σ_n_epochs = warmup + 100)
#   pred_loss_weight = 0
#
# 2 σ_end × 2 warmup × 2 lr = 8 experiments
#
# n_epochs=500, patience=150 (generous: allow second learning phase
# after annealing completes)
#
# Usage:
#   bash shells/slurm/submit_knapsack_wave3.sh --dry-run
#   bash shells/slurm/submit_knapsack_wave3.sh
#   bash shells/slurm/submit_knapsack_wave3.sh --no-backup
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
DIR="./openpto/data/knapsack"
CFGDIR="openpto/config/models"

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Check if experiment already completed ----
is_completed() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb/${prefix}/results.npy" ]]
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

# Tag mappings
declare -A LR_TAG
LR_TAG[0.001]="lr1e3"
LR_TAG[0.005]="lr5e3"

# Fixed settings
N_SAMPLES=5
N_EPOCHS=500
PATIENCE=150

SIGMA_ENDS=("0.05" "0.01")
WARMUPS=("50" "100")
LRS=("0.001" "0.005")

declare -A SE_TAG
SE_TAG[0.05]="se005"
SE_TAG[0.01]="se001"

declare -A W_TAG
W_TAG[50]="w50"
W_TAG[100]="w100"

# ----- Build experiment list -----
for se in "${SIGMA_ENDS[@]}"; do
    se_tag="${SE_TAG[$se]}"
    for warmup in "${WARMUPS[@]}"; do
        w_tag="${W_TAG[$warmup]}"
        yaml="${CFGDIR}/perturb_kn_w3_${se_tag}_${w_tag}.yaml"

        # Verify YAML exists
        if [[ ! -f "$yaml" ]]; then
            echo "WARNING: Missing config $yaml — skipping"
            continue
        fi

        for lr in "${LRS[@]}"; do
            lr_tag="${LR_TAG[$lr]}"
            prefix="sd1_kn_w3_${se_tag}_${w_tag}_${lr_tag}"

            cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr ${lr} --data_dir ${DIR} --gpu ${GPU_ID}"
            EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")
        done
    done
done

# ---- Main ----
echo "=============================================="
echo "Wave 3 — Knapsack: Aggressive σ annealing"
echo "Grid: ${#SIGMA_ENDS[@]} σ_end × ${#WARMUPS[@]} warmup × ${#LRS[@]} lr = ${#EXPERIMENTS[@]} experiments"
echo "Schedule: linear_decay, σ: 0.5 → {0.05, 0.01} over 100 ep"
echo "Warmup: {50, 100} ep | n_epochs=${N_EPOCHS} | patience=${PATIENCE}"
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

    job_name="pco_knapsack_${prefix}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python rethink_exp/main_results.py --problem=knapsack --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

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
    echo "Monitor with: squeue -u \$USER | grep pco_knapsack"
fi
