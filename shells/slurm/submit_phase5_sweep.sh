#!/usr/bin/env bash
# =============================================================================
# Phase 5: Sigma Decay + Softplus + Low-LR Portfolio
# =============================================================================
# Knapsack (σ=0.5, n=5, 500ep, patience=200):
#   5a: baseline (no activation, constant σ)  × lr={0.05,0.02,0.01,0.005}  → 4
#   5b: softplus activation                   × lr={0.05,0.02,0.01,0.005}  → 4
#   5c: linear decay σ 0.5→0.05, warmup=100   × lr={0.05,0.02,0.01,0.005} → 4
#   5d: decay + softplus                      × lr={0.02,0.01,0.005}       → 3
#
# Portfolio (no activation, 300ep, patience=100):
#   5e: σ=0.5, lr=5e-4, n={1,5}                                           → 2
#   5f: σ=0.01, lr=5e-4, n={1,5}                                          → 2
#
# Total: 19 experiments (38 SLURM jobs with backup)
#
# Usage:
#   bash shells/slurm/submit_phase5_sweep.sh --dry-run
#   bash shells/slurm/submit_phase5_sweep.sh
#   bash shells/slurm/submit_phase5_sweep.sh --problem knapsack
#   bash shells/slurm/submit_phase5_sweep.sh --wave 5a
# =============================================================================

set -e

# ---- Defaults ----
DRY_RUN=false
PROBLEM_FILTER=""
WAVE_FILTER=""
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
        --problem)                PROBLEM_FILTER="$2"; shift 2 ;;
        --wave)                   WAVE_FILTER="$2"; shift 2 ;;
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

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Map problem_key to saved_records directory name ----
get_data_dir_name() {
    local problem_key="$1"
    case "$problem_key" in
        portfolio)           echo "portfolio-real" ;;
        bipartitematching)   echo "bipartitematching-cora" ;;
        knapsack)            echo "knapsack-gen" ;;
        knapsack_energy)     echo "knapsack-energy" ;;
        budgetalloc)         echo "budgetalloc-real" ;;
        energy)              echo "energy-energy" ;;
        cubic)               echo "cubic-gen" ;;
        *)                   echo "$problem_key" ;;
    esac
}

# ---- Check if experiment already completed ----
is_completed() {
    local problem_key="$1"
    local prefix="$2"
    local data_dir_name
    data_dir_name=$(get_data_dir_name "$problem_key")
    [[ -f "$SCRIPT_DIR/saved_records/${data_dir_name}/perturb/${prefix}/results.npy" ]]
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
# Experiment definitions
# Format: wave|problem_arg|problem_key|yaml_path|prefix|extra_args
# =============================================================================

EXPERIMENTS=()

# =====================================================================
# Wave 5a: Knapsack baseline — σ=0.5, n=5, no activation, constant σ
# =====================================================================
for lr in 0.05 0.02 0.01 0.005; do
    case "$lr" in
        0.05)  lr_tag="lr05" ;;
        0.02)  lr_tag="lr02" ;;
        0.01)  lr_tag="lr01" ;;
        0.005) lr_tag="lr005" ;;
    esac
    yaml="${CFGDIR}/perturb_s05_n5.yaml"
    prefix="p5_s05_n5_${lr_tag}"
    EXPERIMENTS+=("5a|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
done

# =====================================================================
# Wave 5b: Knapsack softplus — σ=0.5, n=5, softplus activation
# =====================================================================
for lr in 0.05 0.02 0.01 0.005; do
    case "$lr" in
        0.05)  lr_tag="lr05" ;;
        0.02)  lr_tag="lr02" ;;
        0.01)  lr_tag="lr01" ;;
        0.005) lr_tag="lr005" ;;
    esac
    yaml="${CFGDIR}/perturb_s05_n5_softplus.yaml"
    prefix="p5_s05_n5_sp_${lr_tag}"
    EXPERIMENTS+=("5b|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
done

# =====================================================================
# Wave 5c: Knapsack linear decay — σ 0.5→0.05, warmup=100, n=5
# =====================================================================
for lr in 0.05 0.02 0.01 0.005; do
    case "$lr" in
        0.05)  lr_tag="lr05" ;;
        0.02)  lr_tag="lr02" ;;
        0.01)  lr_tag="lr01" ;;
        0.005) lr_tag="lr005" ;;
    esac
    yaml="${CFGDIR}/perturb_s05_n5_lindecay_w100.yaml"
    prefix="p5_s05_n5_decay_${lr_tag}"
    EXPERIMENTS+=("5c|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
done

# =====================================================================
# Wave 5d: Knapsack decay + softplus — σ 0.5→0.05, warmup=100, n=5
# =====================================================================
for lr in 0.02 0.01 0.005; do
    case "$lr" in
        0.02)  lr_tag="lr02" ;;
        0.01)  lr_tag="lr01" ;;
        0.005) lr_tag="lr005" ;;
    esac
    yaml="${CFGDIR}/perturb_s05_n5_lindecay_w100_softplus.yaml"
    prefix="p5_s05_n5_decay_sp_${lr_tag}"
    EXPERIMENTS+=("5d|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
done

# =====================================================================
# Wave 5e: Portfolio — σ=0.5, no activation, lr=5e-4, n={1,5}
# =====================================================================
for ns in 1 5; do
    yaml="${CFGDIR}/perturb_s05_n${ns}.yaml"
    prefix="p5_s05_n${ns}_lr0005"
    EXPERIMENTS+=("5e|portfolio|portfolio|${yaml}|${prefix}|--solver cvxpy --n_epochs 300 --lr 0.0005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}")
done

# =====================================================================
# Wave 5f: Portfolio — σ=0.01, no activation, lr=5e-4, n={1,5}
# =====================================================================
for ns in 1 5; do
    yaml="${CFGDIR}/perturb_s001_n${ns}.yaml"
    prefix="p5_s001_n${ns}_lr0005"
    EXPERIMENTS+=("5f|portfolio|portfolio|${yaml}|${prefix}|--solver cvxpy --n_epochs 300 --lr 0.0005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}")
done


# ---- Main ----
echo "=============================================="
echo "Phase 5: Decay + Softplus + Low-LR Portfolio"
echo "=============================================="
echo "Primary:  $PARTITION"
if $USE_BACKUP; then
    echo "Backup:   $BACKUP_PARTITION (afternotok dependency)"
else
    echo "Backup:   disabled"
fi
echo "Time: $TIME  Mem: $MEM  GRES: $GRES"
if [[ -n "$PROBLEM_FILTER" ]]; then echo "Problem filter: $PROBLEM_FILTER"; fi
if [[ -n "$WAVE_FILTER" ]]; then echo "Wave filter: $WAVE_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN MODE ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "=============================================="
echo ""

submitted=0
skipped=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r wave problem_arg problem_key yaml_path prefix extra_args <<< "$entry"

    # Apply wave filter
    if [[ -n "$WAVE_FILTER" && "$wave" != "$WAVE_FILTER" ]]; then
        continue
    fi

    # Apply problem filter
    if [[ -n "$PROBLEM_FILTER" && "$problem_key" != *"$PROBLEM_FILTER"* ]]; then
        continue
    fi

    # Build job name
    job_name="pco_${problem_key}_${prefix}"

    # Skip completed
    if $SKIP_COMPLETED && is_completed "$problem_key" "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    # Build command
    cmd="python rethink_exp/main_results.py --problem=${problem_arg} --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

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
    echo "Monitor with: squeue -u \$USER"
fi
