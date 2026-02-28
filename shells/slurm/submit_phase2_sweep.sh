#!/usr/bin/env bash
# =============================================================================
# Phase 2: n_samples Scale-up & Sigmoid Activation
# =============================================================================
# Builds on Phase 1 findings: σ=0.1 best for portfolio (n=1), σ=0.3 best for
# knapsack (n=1). Phase 2 scales n_samples and tests sigmoid activation.
#
# Portfolio (target: test_regret < 0.24):
#   2a: Sigmoid activation + small σ (old best used σ=0.01, n=25, sigmoid)
#       σ = {0.005, 0.01, 0.02}, n = {5, 25}, sigmoid, lr = 0.005 → 6 jobs
#       σ = {0.005, 0.01, 0.02}, n = 25,      sigmoid, lr = 0.01  → 3 jobs
#   2b: Scale up Phase 1 winner (σ=0.1)
#       σ = 0.1, n = {5, 25}, normal, lr = 0.005 → 2 jobs
#
# Knapsack (target: test_regret < 3.0):
#   2c: Scale n_samples at sweet-spot sigmas
#       σ = {0.2, 0.3, 0.5}, n = {5, 10, 25}, lr = 0.005 → 9 jobs
#
# Total: 20 jobs (fits in one SLURM wave)
#
# Usage:
#   bash shells/slurm/submit_phase2_sweep.sh --dry-run
#   bash shells/slurm/submit_phase2_sweep.sh
#   bash shells/slurm/submit_phase2_sweep.sh --problem portfolio
#   bash shells/slurm/submit_phase2_sweep.sh --problem knapsack
#   bash shells/slurm/submit_phase2_sweep.sh --wave 2a
#   bash shells/slurm/submit_phase2_sweep.sh --resubmit-preempted --warm-restart
# =============================================================================

set -e

# ---- Defaults ----
DRY_RUN=false
PROBLEM_FILTER=""
WAVE_FILTER=""
PARTITION="preempt"
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="pco_bench_rhel7"
RESUBMIT_PREEMPTED=false
WARM_RESTART=false
SKIP_COMPLETED=true

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                DRY_RUN=true; shift ;;
        --problem)                PROBLEM_FILTER="$2"; shift 2 ;;
        --wave)                   WAVE_FILTER="$2"; shift 2 ;;
        --partition)              PARTITION="$2"; shift 2 ;;
        --time)                   TIME="$2"; shift 2 ;;
        --mem)                    MEM="$2"; shift 2 ;;
        --gres)                   GRES="$2"; shift 2 ;;
        --gpu)                    GPU_ID="$2"; shift 2 ;;
        --conda-env)              CONDA_ENV="$2"; shift 2 ;;
        --resubmit-preempted)     RESUBMIT_PREEMPTED=true; shift ;;
        --warm-restart)           WARM_RESTART=true; shift ;;
        --no-skip-completed)      SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/"

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
        advertising)         echo "advertising-real" ;;
        *)                   echo "$problem_key" ;;
    esac
}

# ---- Check if experiment already completed (has results.npy) ----
is_completed() {
    local problem_key="$1"
    local prefix="$2"
    local data_dir_name
    data_dir_name=$(get_data_dir_name "$problem_key")
    [[ -f "$SCRIPT_DIR/saved_records/${data_dir_name}/perturb/${prefix}/results.npy" ]]
}

# ---- Check if experiment was preempted ----
is_preempted() {
    local job_name="$1"
    local prefix="$2"
    local problem_key="$3"
    local data_dir_name
    data_dir_name=$(get_data_dir_name "$problem_key")
    
    for ef in "$SCRIPT_DIR/logs/slurm/${job_name}_"*.err; do
        [[ -f "$ef" ]] || continue
        if tail -c 500 "$ef" 2>/dev/null | grep -q "DUE TO PREEMPTION\|CANCELLED"; then
            if ! is_completed "$problem_key" "$prefix"; then
                return 0
            fi
        fi
    done
    return 1
}

# ---- Get checkpoint path if it exists ----
get_checkpoint() {
    local problem_key="$1"
    local prefix="$2"
    local data_dir_name
    data_dir_name=$(get_data_dir_name "$problem_key")
    local ckpt="$SCRIPT_DIR/saved_records/${data_dir_name}/perturb/${prefix}/checkpoints/tr_pred_best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "$ckpt"
    else
        echo ""
    fi
}

# ---- Submit one job ----
submit_job() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] sbatch --job-name=$job_name"
        echo "          CMD: $cmd"
        echo ""
        return
    fi

    echo "Submitting: $job_name"

    sbatch \
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
echo 'Job: $job_name'
echo 'CMD: $cmd'
echo 'Python:' \$(which python)
echo 'Start:' \$(date)
echo '=============================================='

$cmd

echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
"
}

# =============================================================================
# Experiment definitions
# =============================================================================
# Format: wave|problem_arg|problem_key|yaml_path|prefix|extra_args

EXPERIMENTS=(
    # ===================== Phase 2A: Portfolio sigmoid =====================
    # σ = {0.005, 0.01, 0.02}, n = {5, 25}, sigmoid, lr = 0.005
    "2a|portfolio|portfolio|openpto/config/models/perturb_s0005_n5_sigmoid.yaml|p2_s0005_n5_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s0005_n25_sigmoid.yaml|p2_s0005_n25_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s001_n5_sigmoid.yaml|p2_s001_n5_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s001_n25_sigmoid.yaml|p2_s001_n25_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s002_n5_sigmoid.yaml|p2_s002_n5_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s002_n25_sigmoid.yaml|p2_s002_n25_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # σ = {0.005, 0.01, 0.02}, n = 25, sigmoid, lr = 0.01
    "2a|portfolio|portfolio|openpto/config/models/perturb_s0005_n25_sigmoid.yaml|p2_s0005_n25_sigmoid_lr01|--solver cvxpy --n_epochs 300 --lr 0.01 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s001_n25_sigmoid.yaml|p2_s001_n25_sigmoid_lr01|--solver cvxpy --n_epochs 300 --lr 0.01 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2a|portfolio|portfolio|openpto/config/models/perturb_s002_n25_sigmoid.yaml|p2_s002_n25_sigmoid_lr01|--solver cvxpy --n_epochs 300 --lr 0.01 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Phase 2B: Portfolio n-scale =====================
    # Scale up Phase 1 winner: σ = 0.1, n = {5, 25}
    "2b|portfolio|portfolio|openpto/config/models/perturb_s01_n5.yaml|p2_s01_n5|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "2b|portfolio|portfolio|openpto/config/models/perturb_s01_n25.yaml|p2_s01_n25|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Phase 2C: Knapsack n-scale =====================
    # σ = {0.2, 0.3, 0.5}, n = {5, 10, 25}
    "2c|knapsack|knapsack|openpto/config/models/perturb_s02_n5.yaml|p2_s02_n5|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s02_n10.yaml|p2_s02_n10|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s02_n25.yaml|p2_s02_n25|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s03_n5.yaml|p2_s03_n5|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s03_n10.yaml|p2_s03_n10|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s03_n25.yaml|p2_s03_n25|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s05_n5.yaml|p2_s05_n5|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s05_n10.yaml|p2_s05_n10|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "2c|knapsack|knapsack|openpto/config/models/perturb_s05_n25.yaml|p2_s05_n25|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
)

# ---- Main ----
echo "=============================================="
echo "Phase 2: n_samples Scale-up & Sigmoid"
echo "=============================================="
echo "Partition: $PARTITION  Time: $TIME  Mem: $MEM  GRES: $GRES"
if [[ -n "$PROBLEM_FILTER" ]]; then
    echo "Problem filter: $PROBLEM_FILTER"
fi
if [[ -n "$WAVE_FILTER" ]]; then
    echo "Wave filter: $WAVE_FILTER"
fi
if $DRY_RUN; then
    echo "*** DRY RUN MODE ***"
fi
if $RESUBMIT_PREEMPTED; then
    echo "*** RESUBMIT PREEMPTED ONLY ***"
fi
if $WARM_RESTART; then
    echo "*** WARM RESTART: loading checkpoints ***"
fi
if $SKIP_COMPLETED; then
    echo "Skip completed: ON (use --no-skip-completed to override)"
fi
echo "=============================================="
echo ""

submitted=0
skipped_done=0
skipped_not_preempted=0
for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r wave problem_arg problem_key yaml_path prefix extra_args <<< "$entry"

    # Apply wave filter
    if [[ -n "$WAVE_FILTER" && "$wave" != "$WAVE_FILTER" ]]; then
        continue
    fi

    # Apply problem filter
    if [[ -n "$PROBLEM_FILTER" && "$problem_key" != "$PROBLEM_FILTER" ]]; then
        continue
    fi

    # Build job name
    job_name="pco_${problem_key}_${prefix}"

    # Skip already-completed experiments
    if $SKIP_COMPLETED && is_completed "$problem_key" "$prefix"; then
        echo "[SKIP] $job_name — already completed (results.npy exists)"
        skipped_done=$((skipped_done + 1))
        continue
    fi

    # If --resubmit-preempted, only submit jobs that were preempted
    if $RESUBMIT_PREEMPTED && ! is_preempted "$job_name" "$prefix" "$problem_key"; then
        skipped_not_preempted=$((skipped_not_preempted + 1))
        continue
    fi

    # Build command
    cmd="python rethink_exp/main_results.py --problem=${problem_arg} --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"

    # Add warm restart if checkpoint exists and --warm-restart is set
    if $WARM_RESTART; then
        ckpt=$(get_checkpoint "$problem_key" "$prefix")
        if [[ -n "$ckpt" ]]; then
            cmd="${cmd} --trained_path ${ckpt}"
            echo "  → warm restart from: ${ckpt##*/saved_records/}"
        fi
    fi

    submit_job "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "Total submitted: $submitted"
if [[ $skipped_done -gt 0 ]]; then
    echo "Skipped (completed): $skipped_done"
fi
if [[ $skipped_not_preempted -gt 0 ]]; then
    echo "Skipped (not preempted): $skipped_not_preempted"
fi
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "All jobs submitted. Check with: squeue -u \$USER"
fi
