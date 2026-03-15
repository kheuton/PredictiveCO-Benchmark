#!/usr/bin/env bash
# ====================================================================
# Exp 4: LR schedule ablation (step decay vs constant LR)
# ====================================================================
# Run AFTER Step 0 completes. Update BEST_YAML below.
# Compares MultiStepLR (gamma=0.316 at ep100+ep200) vs constant LR.
#
# Benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model. DP solver, n_samples=100, 300 epochs.
# No MSE pretraining. Full-batch Adam.
#
# Grid: 3 LRs × 2 (sched/no-sched) = 6 jobs  (~10 min each, 1h limit)
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_lrsched_{tag}_lr{lr}/
# Figure:  python rethink_exp/fig_kn_lrsched_sweep.py
#
# Usage:
#   bash shells/slurm/submit_kn_lrsched_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_lrsched_sweep.sh
#   bash shells/slurm/submit_kn_lrsched_sweep.sh --lr 1e-2
#   bash shells/slurm/submit_kn_lrsched_sweep.sh --sched sched   # sched only
#   bash shells/slurm/submit_kn_lrsched_sweep.sh --sched nosched # no-sched only
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
SCHED_FILTER=""
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --sched)              SCHED_FILTER="$2"; shift 2 ;;
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
# TODO: Update after Step 0 completes (fig_kn_plain_sigma_sweep.py)
BEST_YAML="openpto/config/models/perturb_kn_s05_n100.yaml"
# ====================================================================

LRS=("5e-2" "1e-2" "5e-3")

BASE_ARGS="--problem knapsack --opt_model perturb --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--n_epochs 300 --n_ptr_epochs 0 \
--opt_name gd --batch_size 400 \
--pred_model dense \
--method_path ${BEST_YAML} \
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
echo "Knapsack LR schedule ablation (Exp 4)"
echo "3 LRs × 2 (sched/no-sched) = 6 jobs  |  DP, n=100, 300ep"
echo "YAML: $BEST_YAML"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]];   then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$SCHED_FILTER" ]]; then echo "Sched filter: $SCHED_FILTER"; fi
if $DRY_RUN;        then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for lr in "${LRS[@]}"; do
    [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]] && continue

    # No scheduling
    if [[ -z "$SCHED_FILTER" || "$SCHED_FILTER" == "nosched" ]]; then
        prefix="kn_plain_lrsched_nosched_lr${lr}"
        job_name="kn_lrs_no_lr${lr//-/}"

        if $SKIP_COMPLETED && is_completed "$prefix"; then
            echo "[SKIP] $job_name — already completed"
            skipped=$((skipped + 1))
        else
            cmd="python rethink_exp/main_results.py $BASE_ARGS \
--lr ${lr} --prefix ${prefix}"
            submit_job "$job_name" "$cmd"
            submitted=$((submitted + 1))
        fi
    fi

    # With step-decay scheduling (milestones at ep100, ep200)
    if [[ -z "$SCHED_FILTER" || "$SCHED_FILTER" == "sched" ]]; then
        prefix="kn_plain_lrsched_sched_lr${lr}"
        job_name="kn_lrs_sc_lr${lr//-/}"

        if $SKIP_COMPLETED && is_completed "$prefix"; then
            echo "[SKIP] $job_name — already completed"
            skipped=$((skipped + 1))
        else
            cmd="python rethink_exp/main_results.py $BASE_ARGS \
--lr ${lr} --use_lr_scheduling --lr_milestone_1 100 --lr_milestone_2 200 \
--prefix ${prefix}"
            submit_job "$job_name" "$cmd"
            submitted=$((submitted + 1))
        fi
    fi
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn_lrs"; fi
echo ""
echo "After jobs complete, run:"
echo "  python rethink_exp/fig_kn_lrsched_sweep.py"
