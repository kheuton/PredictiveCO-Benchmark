#!/usr/bin/env bash
# ====================================================================
# Exp 3: Random seed sweep — variance of cold-start perturbed optimizer
# ====================================================================
# Run AFTER Step 0 completes. Update BEST_YAML and BEST_LR below
# to the best (sigma, LR) from fig_kn_plain_sigma_sweep.py.
#
# Benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model. DP solver, n_samples=100, 300 epochs.
# 25 independent random seeds. No MSE pretraining. Full-batch Adam.
#
# Grid: 25 seeds = 25 jobs  (~10 min each, 1h limit)
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_seed_{seed}/
# Figure:  python rethink_exp/fig_kn_seed_sweep.py
#
# Usage:
#   bash shells/slurm/submit_kn_seed_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_seed_sweep.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --time)               TIME="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# TODO: Update these after Step 0 completes (fig_kn_plain_sigma_sweep.py)
BEST_YAML="openpto/config/models/perturb_kn_s05_n100.yaml"
BEST_LR="5e-3"
# ====================================================================

SEEDS=($(seq 0 24))

BASE_ARGS="--problem knapsack --opt_model perturb --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--n_epochs 300 --n_ptr_epochs 0 \
--opt_name gd --batch_size 400 \
--pred_model dense \
--method_path ${BEST_YAML} \
--lr ${BEST_LR} \
--gpu -1 --loadnew False"

is_completed() {
    local prefix="$1"
    [[ -f "$RESULTS_BASE/${prefix}/results.npy" ]]
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
echo "======================================================================"
echo "Knapsack random seed sweep (Exp 3)"
echo "25 seeds  |  YAML: $BEST_YAML  LR: $BEST_LR"
echo "DP, n=100, 300ep, ~10min each"
echo "Partition: $PARTITION  Time: $TIME"
if $DRY_RUN;        then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for seed in "${SEEDS[@]}"; do
    prefix="kn_plain_seed_${seed}"
    job_name="kn_seed_${seed}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python rethink_exp/main_results.py $BASE_ARGS \
--seed ${seed} --prefix ${prefix}"
    submit_job "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn_seed"; fi
echo ""
echo "After jobs complete, run:"
echo "  python rethink_exp/fig_kn_seed_sweep.py"
