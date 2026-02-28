#!/usr/bin/env bash
# =============================================================================
# Phase 1: Fast n_samples=1 Sigma Sweep
# =============================================================================
# ~47 experiments using n_samples=1 for fast screening across 6 problems.
# Tests sigma, Gumbel noise, and sigmoid activation.
#
# Estimated runtimes (with patience=100):
#   Bipartite:  ~2 min/job   (6 sigma + 3 gumbel + 2 sigmoid = 11 jobs)
#   Portfolio:  ~3 min/job   (6 sigma + 3 gumbel = 9 jobs)
#   Knapsack:   ~4 min/job   (6 sigma + 3 gumbel = 9 jobs)
#   Knap-energy: ~1 min/job  (4 sigma = 4 jobs)
#   Budget:    ~83 min/job   (5 sigma = 5 jobs)
#   Energy:   ~20 hr/job     (3 sigma = 3 jobs)
#
# Usage:
#   bash shells/slurm/submit_phase1_sweep.sh --dry-run
#   bash shells/slurm/submit_phase1_sweep.sh
#   bash shells/slurm/submit_phase1_sweep.sh --problem portfolio
#   bash shells/slurm/submit_phase1_sweep.sh --wave 1a          # sigma only
#   bash shells/slurm/submit_phase1_sweep.sh --wave 1b          # gumbel only
#   bash shells/slurm/submit_phase1_sweep.sh --wave 1c          # sigmoid only
#
# Resubmission:
#   # Resubmit only preempted jobs, warm-starting from saved checkpoint:
#   bash shells/slurm/submit_phase1_sweep.sh --resubmit-preempted --warm-restart
#   # Resubmit all incomplete jobs with warm restart:
#   bash shells/slurm/submit_phase1_sweep.sh --warm-restart
#   # Force resubmit everything (even completed):
#   bash shells/slurm/submit_phase1_sweep.sh --no-skip-completed
#
# Note: --warm-restart uses --trained_path to load the best saved weights.
#   This does NOT restore optimizer state, epoch counter, or best metric.
#   Training restarts from epoch 0 with a fresh Adam optimizer.
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
    
    # Check SLURM err logs for preemption
    for ef in "$SCRIPT_DIR/logs/slurm/${job_name}_"*.err; do
        [[ -f "$ef" ]] || continue
        if tail -c 500 "$ef" 2>/dev/null | grep -q "DUE TO PREEMPTION\|CANCELLED"; then
            # Also verify it didn't complete after being resubmitted
            if ! is_completed "$problem_key" "$prefix"; then
                return 0  # true: preempted and not yet completed
            fi
        fi
    done
    return 1  # false: not preempted (or already completed)
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
# Phase 1A — Sigma sweep (normal noise, n=1)
# =============================================================================
# Format: wave|problem_arg|problem_key|yaml_path|prefix|extra_args

EXPERIMENTS=(
    # ===================== Portfolio (priority) =====================
    # sigma = 0.01, 0.05, 0.1, 0.3, 0.5, 1.0
    "1a|portfolio|portfolio|openpto/config/models/perturb_s0001_n1.yaml|sweep_s001_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|portfolio|portfolio|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|portfolio|portfolio|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|portfolio|portfolio|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|portfolio|portfolio|openpto/config/models/perturb_s05_n1.yaml|sweep_s05_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|portfolio|portfolio|openpto/config/models/perturb_s1_n1.yaml|sweep_s1_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Bipartite Matching (priority) =====================
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s0001_n1.yaml|sweep_s001_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s05_n1.yaml|sweep_s05_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1a|bipartitematching|bipartitematching|openpto/config/models/perturb_s1_n1.yaml|sweep_s1_n1|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"

    # ===================== Knapsack gen (priority) =====================
    "1a|knapsack|knapsack|openpto/config/models/perturb_s0001_n1.yaml|sweep_s001_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1a|knapsack|knapsack|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1a|knapsack|knapsack|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1a|knapsack|knapsack|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1a|knapsack|knapsack|openpto/config/models/perturb_s05_n1.yaml|sweep_s05_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1a|knapsack|knapsack|openpto/config/models/perturb_s1_n1.yaml|sweep_s1_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"

    # ===================== Knapsack energy =====================
    "1a|knapsack|knapsack_energy|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    "1a|knapsack|knapsack_energy|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    "1a|knapsack|knapsack_energy|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
    "1a|knapsack|knapsack_energy|openpto/config/models/perturb_s05_n1.yaml|sweep_s05_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"

    # ===================== Budget Allocation =====================
    "1a|budgetalloc|budgetalloc|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver neural --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|budgetalloc|budgetalloc|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver neural --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|budgetalloc|budgetalloc|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver neural --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|budgetalloc|budgetalloc|openpto/config/models/perturb_s05_n1.yaml|sweep_s05_n1|--solver neural --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|budgetalloc|budgetalloc|openpto/config/models/perturb_s1_n1.yaml|sweep_s1_n1|--solver neural --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # ===================== Energy (slow — runs overnight) =====================
    "1a|energy|energy|openpto/config/models/perturb_s005_n1.yaml|sweep_s005_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|energy|energy|openpto/config/models/perturb_s01_n1.yaml|sweep_s01_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1a|energy|energy|openpto/config/models/perturb_s03_n1.yaml|sweep_s03_n1|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"

    # =============================================================================
    # Phase 1B — Gumbel noise (n=1, priority problems only)
    # =============================================================================
    # Portfolio
    "1b|portfolio|portfolio|openpto/config/models/perturb_s01_n1_gumbel.yaml|sweep_s01_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1b|portfolio|portfolio|openpto/config/models/perturb_s03_n1_gumbel.yaml|sweep_s03_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    "1b|portfolio|portfolio|openpto/config/models/perturb_s05_n1_gumbel.yaml|sweep_s05_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --data_dir ${DIR}"
    # Bipartite
    "1b|bipartitematching|bipartitematching|openpto/config/models/perturb_s01_n1_gumbel.yaml|sweep_s01_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1b|bipartitematching|bipartitematching|openpto/config/models/perturb_s03_n1_gumbel.yaml|sweep_s03_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1b|bipartitematching|bipartitematching|openpto/config/models/perturb_s05_n1_gumbel.yaml|sweep_s05_n1_gumbel|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    # Knapsack gen
    "1b|knapsack|knapsack|openpto/config/models/perturb_s01_n1_gumbel.yaml|sweep_s01_n1_gumbel|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1b|knapsack|knapsack|openpto/config/models/perturb_s03_n1_gumbel.yaml|sweep_s03_n1_gumbel|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"
    "1b|knapsack|knapsack|openpto/config/models/perturb_s05_n1_gumbel.yaml|sweep_s05_n1_gumbel|--solver gurobi --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID}"

    # =============================================================================
    # Phase 1C — Sigmoid activation for bipartite (n=1)
    # =============================================================================
    "1c|bipartitematching|bipartitematching|openpto/config/models/perturb_s01_n1_sigmoid.yaml|sweep_s01_n1_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
    "1c|bipartitematching|bipartitematching|openpto/config/models/perturb_s03_n1_sigmoid.yaml|sweep_s03_n1_sigmoid|--solver cvxpy --n_epochs 300 --lr 0.005 --patience 100 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
)

# ---- Main ----
echo "=============================================="
echo "Phase 1: Fast n=1 Sigma Sweep"
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
