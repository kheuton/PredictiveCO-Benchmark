#!/usr/bin/env bash
# ====================================================================
# Exp 2: MSE regularization sweep (plain perturbed + λ·MSE loss term)
# ====================================================================
# Run AFTER Step 0 completes. Update BEST_SIGMA_YAML below before
# submitting (pick the best (sigma, LR) from fig_kn_plain_sigma_sweep.py).
#
# Benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model. DP solver, n_samples=100, 300 epochs.
# pred_loss_weight ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0} (set in YAML).
# No MSE pretraining. Full-batch Adam.
#
# Grid: 6 λ values × 3 LRs = 18 jobs  (~10 min each, 1h limit)
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_mse_reg_w{w}_lr{lr}/
#
# Usage:
#   bash shells/slurm/submit_kn_mse_reg_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_mse_reg_sweep.sh
#   bash shells/slurm/submit_kn_mse_reg_sweep.sh --lr 1e-2
#   bash shells/slurm/submit_kn_mse_reg_sweep.sh --weight w001
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
WEIGHT_FILTER=""
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --weight)             WEIGHT_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --time)               TIME="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# weight label → YAML file (sigma set inside YAML; update after Step 0)
declare -A WEIGHT_YAML
WEIGHT_YAML[w0001]="perturb_kn_reg_w0001_n100.yaml"
WEIGHT_YAML[w001]="perturb_kn_reg_w001_n100.yaml"
WEIGHT_YAML[w005]="perturb_kn_reg_w005_n100.yaml"
WEIGHT_YAML[w01]="perturb_kn_reg_w01_n100.yaml"
WEIGHT_YAML[w05]="perturb_kn_reg_w05_n100.yaml"
WEIGHT_YAML[w1]="perturb_kn_reg_w1_n100.yaml"
WEIGHT_LABELS=(w0001 w001 w005 w01 w05 w1)

LRS=("5e-2" "1e-2" "5e-3")

# Shared args
BASE_ARGS="--problem knapsack --opt_model perturb --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--n_epochs 300 --n_ptr_epochs 0 \
--opt_name gd --batch_size 400 \
--pred_model dense \
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
echo "Knapsack MSE regularization sweep (Exp 2)"
echo "6 weights × 3 LRs = 18 jobs  |  DP, n=100, 300ep, ~10min each"
echo "NOTE: Update sigma in YAML files to best sigma from Step 0 first!"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]];     then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$WEIGHT_FILTER" ]]; then echo "Weight filter: $WEIGHT_FILTER"; fi
if $DRY_RUN;         then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED;  then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for w_label in "${WEIGHT_LABELS[@]}"; do
    [[ -n "$WEIGHT_FILTER" && "$w_label" != "$WEIGHT_FILTER" ]] && continue

    yaml="openpto/config/models/${WEIGHT_YAML[$w_label]}"

    for lr in "${LRS[@]}"; do
        [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]] && continue

        prefix="kn_plain_mse_reg_${w_label}_lr${lr}"
        job_name="kn_reg_${w_label}_lr${lr//-/}"

        if $SKIP_COMPLETED && is_completed "$prefix"; then
            echo "[SKIP] $job_name — already completed"
            skipped=$((skipped + 1))
            continue
        fi

        cmd="python rethink_exp/main_results.py $BASE_ARGS \
--method_path ${yaml} \
--lr ${lr} --prefix ${prefix}"
        submit_job "$job_name" "$cmd"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn_reg"; fi
echo ""
echo "After jobs complete, run:"
echo "  python rethink_exp/fig_kn_mse_reg_sweep.py"
