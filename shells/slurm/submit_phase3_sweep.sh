#!/usr/bin/env bash
# =============================================================================
# Phase 3: LR Grid Search at Good Sigmas, Low n
# =============================================================================
# Dual-partition submission: submits each job to preempt first, then submits a
# backup to hugheslab,gpu with --dependency=afternotok so it auto-starts only
# if the preempt job fails (gets preempted/cancelled).
#
# Key insights from Phases 1-2:
#   Portfolio:  σ=0.01 + sigmoid is promising (val=0.304 in 12ep, preempted)
#               Old best: σ=0.01, n=25, sigmoid, lr=0.01 → test=0.317
#               Need wider LR sweep at n=1 (fast) and n=5
#   Knapsack:   σ=0.5, n=25 got 3.58 but all n=1 runs hit 300-epoch ceiling
#               Need 500 epochs + wider LR range
#
# Grid design (48 experiments):
#   Portfolio (27 jobs, 300 epochs, patience=100):
#     3a: n=1 sigmoid × σ={0.005,0.01,0.02} × lr={0.001,0.002,0.005,0.01,0.02,0.05}  → 18
#     3b: n=5 sigmoid × σ=0.01 × lr={0.001,0.002,0.01,0.02,0.05}                      → 5
#     3c: n=1 normal  × σ=0.1  × lr={0.001,0.002,0.01,0.02}                            → 4
#   Knapsack (21 jobs, 500 epochs, patience=200):
#     3d: n=1 × σ={0.3,0.5,1.0} × lr={0.001,0.002,0.005,0.01,0.02}                    → 15
#     3e: n=5 × σ={0.3,0.5}     × lr={0.002,0.005,0.01}                                → 6
#
# Usage:
#   bash shells/slurm/submit_phase3_sweep.sh --dry-run
#   bash shells/slurm/submit_phase3_sweep.sh                        # dual: preempt + backup
#   bash shells/slurm/submit_phase3_sweep.sh --no-backup            # preempt only
#   bash shells/slurm/submit_phase3_sweep.sh --partition hugheslab,gpu --no-backup
#   bash shells/slurm/submit_phase3_sweep.sh --problem portfolio
#   bash shells/slurm/submit_phase3_sweep.sh --wave 3a
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

# Skip if another job already finished
if [ -f saved_records/*/perturb/${job_name##*_}/results.npy ] 2>/dev/null; then
    echo 'Results already exist, skipping'
    exit 0
fi

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
# =============================================================================
# Format: wave|problem_arg|problem_key|yaml_file|prefix|extra_args
#
# YAML files specify sigma, n_samples, noise, output_activation
# LR, epochs, patience, solver are set via extra_args

EXPERIMENTS=()

# =====================================================================
# Wave 3a: Portfolio sigmoid, n=1, LR grid  (18 jobs)
# σ = {0.005, 0.01, 0.02} × lr = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
# =====================================================================
for sigma_tag in s0005 s001 s002; do
    yaml="${CFGDIR}/perturb_${sigma_tag}_n1_sigmoid.yaml"
    for lr in 0.001 0.002 0.005 0.01 0.02 0.05; do
        lr_tag=$(echo "$lr" | sed 's/0\.//; s/^0*//')
        # Make lr_tag more readable
        case "$lr" in
            0.001) lr_tag="lr001" ;;
            0.002) lr_tag="lr002" ;;
            0.005) lr_tag="lr005" ;;
            0.01)  lr_tag="lr01" ;;
            0.02)  lr_tag="lr02" ;;
            0.05)  lr_tag="lr05" ;;
        esac
        prefix="p3_${sigma_tag}_n1_sig_${lr_tag}"
        EXPERIMENTS+=("3a|portfolio|portfolio|${yaml}|${prefix}|--solver cvxpy --n_epochs 300 --lr ${lr} --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}")
    done
done

# =====================================================================
# Wave 3b: Portfolio sigmoid, n=5, σ=0.01, LR grid  (5 jobs)
# lr = {0.001, 0.002, 0.01, 0.02, 0.05}  (skip 0.005 — done in P2)
# =====================================================================
yaml="${CFGDIR}/perturb_s001_n5_sigmoid.yaml"
for lr in 0.001 0.002 0.01 0.02 0.05; do
    case "$lr" in
        0.001) lr_tag="lr001" ;;
        0.002) lr_tag="lr002" ;;
        0.01)  lr_tag="lr01" ;;
        0.02)  lr_tag="lr02" ;;
        0.05)  lr_tag="lr05" ;;
    esac
    prefix="p3_s001_n5_sig_${lr_tag}"
    EXPERIMENTS+=("3b|portfolio|portfolio|${yaml}|${prefix}|--solver cvxpy --n_epochs 300 --lr ${lr} --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}")
done

# =====================================================================
# Wave 3c: Portfolio normal, n=1, σ=0.1, LR grid  (4 jobs)
# lr = {0.001, 0.002, 0.01, 0.02}  (skip 0.005 — done in P1)
# =====================================================================
yaml="${CFGDIR}/perturb_s01_n1.yaml"
for lr in 0.001 0.002 0.01 0.02; do
    case "$lr" in
        0.001) lr_tag="lr001" ;;
        0.002) lr_tag="lr002" ;;
        0.01)  lr_tag="lr01" ;;
        0.02)  lr_tag="lr02" ;;
    esac
    prefix="p3_s01_n1_${lr_tag}"
    EXPERIMENTS+=("3c|portfolio|portfolio|${yaml}|${prefix}|--solver cvxpy --n_epochs 300 --lr ${lr} --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}")
done

# =====================================================================
# Wave 3d: Knapsack normal, n=1, LR grid, 500 epochs  (15 jobs)
# σ = {0.3, 0.5, 1.0} × lr = {0.001, 0.002, 0.005, 0.01, 0.02}
# =====================================================================
for sigma_tag in s03 s05 s1; do
    yaml="${CFGDIR}/perturb_${sigma_tag}_n1.yaml"
    for lr in 0.001 0.002 0.005 0.01 0.02; do
        case "$lr" in
            0.001) lr_tag="lr001" ;;
            0.002) lr_tag="lr002" ;;
            0.005) lr_tag="lr005" ;;
            0.01)  lr_tag="lr01" ;;
            0.02)  lr_tag="lr02" ;;
        esac
        prefix="p3_${sigma_tag}_n1_${lr_tag}"
        EXPERIMENTS+=("3d|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
    done
done

# =====================================================================
# Wave 3e: Knapsack normal, n=5, LR grid, 500 epochs  (6 jobs)
# σ = {0.3, 0.5} × lr = {0.002, 0.005, 0.01}
# =====================================================================
for sigma_tag in s03 s05; do
    yaml="${CFGDIR}/perturb_${sigma_tag}_n5.yaml"
    for lr in 0.002 0.005 0.01; do
        case "$lr" in
            0.002) lr_tag="lr002" ;;
            0.005) lr_tag="lr005" ;;
            0.01)  lr_tag="lr01" ;;
        esac
        prefix="p3_${sigma_tag}_n5_${lr_tag}"
        EXPERIMENTS+=("3e|knapsack|knapsack|${yaml}|${prefix}|--solver gurobi --n_epochs 500 --lr ${lr} --patience 200 --gpu ${GPU_ID}")
    done
done


# ---- Main ----
echo "=============================================="
echo "Phase 3: LR Grid Search at Good Sigmas"
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
    echo "After preempt jobs finish, backup jobs on $BACKUP_PARTITION auto-cancel."
    echo "If preempted, backup jobs auto-start."
fi
