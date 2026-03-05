#!/usr/bin/env bash
# =============================================================================
# Wave 1 — Soft-Decision DPO: All 8 problems × 3 σ × 3 LR
# =============================================================================
# Grid:
#   σ  ∈ {0.05, 0.1, 0.5}
#   lr ∈ {1e-3, 5e-3, 1e-2}
#   n_samples = 25 (default; 1 for energy),  noise = normal,  no schedule,  no activation
#
# 8 problems × 9 HP combos = 72 experiments (144 SLURM jobs with backup)
#
# Usage:
#   bash shells/slurm/submit_softdec_wave1.sh --dry-run
#   bash shells/slurm/submit_softdec_wave1.sh
#   bash shells/slurm/submit_softdec_wave1.sh --problem portfolio
#   bash shells/slurm/submit_softdec_wave1.sh --no-backup
#   bash shells/slurm/submit_softdec_wave1.sh --n-samples 1   # override default n_samples
# =============================================================================

set -e

# ---- Defaults ----
DRY_RUN=false
PROBLEM_FILTER=""
PARTITION="preempt"
BACKUP_PARTITION="hugheslab,gpu"
USE_BACKUP=true
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true
N_SAMPLES=""  # empty = use per-problem defaults

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                DRY_RUN=true; shift ;;
        --problem)                PROBLEM_FILTER="$2"; shift 2 ;;
        --partition)              PARTITION="$2"; shift 2 ;;
        --backup-partition)       BACKUP_PARTITION="$2"; shift 2 ;;
        --no-backup)              USE_BACKUP=false; shift ;;
        --time)                   TIME="$2"; shift 2 ;;
        --mem)                    MEM="$2"; shift 2 ;;
        --gres)                   GRES="$2"; shift 2 ;;
        --gpu)                    GPU_ID="$2"; shift 2 ;;
        --conda-env)              CONDA_ENV="$2"; shift 2 ;;
        --no-skip-completed)      SKIP_COMPLETED=false; shift ;;
        --n-samples)              N_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/"
CFGDIR="openpto/config/models"

mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Map problem_key → saved_records directory name ----
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
        advertising)         echo "advertising-real" ;;
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
# Experiment grid
# =============================================================================

EXPERIMENTS=()

# ---- Per-problem default n_samples ----
# Energy is very slow per sample; default to 1.
get_n_samples() {
    local problem_key="$1"
    if [[ -n "$N_SAMPLES" ]]; then
        echo "$N_SAMPLES"  # CLI override
        return
    fi
    case "$problem_key" in
        energy) echo 1 ;;
        *)      echo 25 ;;
    esac
}

# ---- Resolve YAML for (sigma_tag, n_samples) ----
# Uses _n{N} suffix when n_samples != 25 (default)
get_yaml() {
    local sigma_tag="$1"
    local n_samples="$2"
    if [[ "$n_samples" -eq 25 ]]; then
        echo "${CFGDIR}/perturb_s${sigma_tag}.yaml"
    else
        echo "${CFGDIR}/perturb_s${sigma_tag}_n${n_samples}.yaml"
    fi
}

# LR → tag mapping
declare -A LR_TAG
LR_TAG[0.001]="lr1e3"
LR_TAG[0.005]="lr5e3"
LR_TAG[0.01]="lr1e2"

SIGMAS=("005" "01" "05")
LRS=("0.001" "0.005" "0.01")

# ----- Problem definitions -----
# Format: problem_arg|problem_key|solver|epochs|extra_args
PROBLEMS=(
    "portfolio|portfolio|cvxpy|300|--data_dir ${DIR} --gpu ${GPU_ID}"
    "knapsack|knapsack|gurobi|300|--gpu ${GPU_ID}"
    "knapsack|knapsack_energy|gurobi|300|--config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR} --gpu ${GPU_ID}"
    "bipartitematching|bipartitematching|cvxpy|300|--instances 20 --testinstances 6 --data_dir ${DIR} --gpu ${GPU_ID}"
    "budgetalloc|budgetalloc|neural|300|--data_dir ${DIR} --gpu ${GPU_ID}"
    "cubic|cubic|heuristic|300|--instances 250 --testinstances 400 --gpu ${GPU_ID}"
    "energy|energy|gurobi|300|--data_dir ${DIR} --gpu ${GPU_ID}"
    "advertising|advertising|ortools|150|--pred_model cvr --n_ptr_epochs 150 --batch_size 1 --data_dir ${DIR} --gpu ${GPU_ID}"
)

# ----- Build experiment list -----
for prob_entry in "${PROBLEMS[@]}"; do
    IFS='|' read -r problem_arg problem_key solver epochs extra_args <<< "$prob_entry"

    local_n_samples=$(get_n_samples "$problem_key")

    for sigma_tag in "${SIGMAS[@]}"; do
        yaml=$(get_yaml "$sigma_tag" "$local_n_samples")

        # Verify YAML exists
        if [[ ! -f "$yaml" ]]; then
            echo "WARNING: Missing config $yaml — skipping ${problem_key} s${sigma_tag} (n_samples=${local_n_samples})"
            continue
        fi

        for lr in "${LRS[@]}"; do
            lr_tag="${LR_TAG[$lr]}"
            prefix="sd1_s${sigma_tag}_${lr_tag}"

            cmd_extra="--solver ${solver} --n_epochs ${epochs} --lr ${lr} ${extra_args}"
            EXPERIMENTS+=("${problem_arg}|${problem_key}|${yaml}|${prefix}|${cmd_extra}")
        done
    done
done

# ---- Main ----
echo "=============================================="
echo "Wave 1 — Soft-Decision DPO"
echo "Grid: 8 problems × 3σ × 3lr = 72 experiments"
echo "=============================================="
echo "Primary:  $PARTITION"
if $USE_BACKUP; then
    echo "Backup:   $BACKUP_PARTITION (afternotok dependency)"
else
    echo "Backup:   disabled"
fi
echo "Time: $TIME  Mem: $MEM  GRES: $GRES"
if [[ -n "$PROBLEM_FILTER" ]]; then echo "Problem filter: $PROBLEM_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN MODE ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "=============================================="
echo ""

submitted=0
skipped=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r problem_arg problem_key yaml_path prefix extra_args <<< "$entry"

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
