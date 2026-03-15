#!/usr/bin/env bash
# ====================================================================
# Step 0: Plain perturbed sigma sweep (benchmark scale, DP solver)
# ====================================================================
# Benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model. DP solver, n_samples=100, 300 epochs.
# No adaptive sigma. No MSE pretraining (--n_ptr_epochs 0).
# Full-batch Adam (--opt_name gd --batch_size 400).
#
# Grid: 5 sigmas × 3 LRs = 15 jobs  (~10 min each, 1h limit)
# Sigmas: 0.1, 0.5, 1.0, 5.0, 10.0
# LRs:    5e-2, 1e-2, 5e-3
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_sigma_sweep_s{sigma}_lr{lr}/
#
# After jobs complete, run:
#   python rethink_exp/fig_kn_plain_sigma_sweep.py
# to find the best (sigma, LR) — use as input to Exp 2/3/4.
#
# Usage:
#   bash shells/slurm/submit_kn_plain_sigma_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_plain_sigma_sweep.sh
#   bash shells/slurm/submit_kn_plain_sigma_sweep.sh --lr 1e-2
#   bash shells/slurm/submit_kn_plain_sigma_sweep.sh --sigma s1
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
SIGMA_FILTER=""
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --sigma)              SIGMA_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --time)               TIME="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# sigma label → YAML file and float value
declare -A SIGMA_YAML
SIGMA_YAML[s01]="perturb_kn_s01_n100.yaml"
SIGMA_YAML[s05]="perturb_kn_s05_n100.yaml"
SIGMA_YAML[s1]="perturb_kn_s1_n100.yaml"
SIGMA_YAML[s5]="perturb_kn_s5_n100.yaml"
SIGMA_YAML[s10]="perturb_kn_s10_n100.yaml"
SIGMA_LABELS=(s01 s05 s1 s5 s10)

LRS=("5e-2" "1e-2" "5e-3")

# Shared args — loads Knapsack_7.pkl (320+80 train/val, 200 test)
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
echo "Knapsack plain perturbed sigma sweep (Step 0)"
echo "5 sigmas × 3 LRs = 15 jobs  |  DP, n=100, 300ep, ~10min each"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]];    then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$SIGMA_FILTER" ]]; then echo "Sigma filter: $SIGMA_FILTER"; fi
if $DRY_RUN;         then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED;  then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for sigma_label in "${SIGMA_LABELS[@]}"; do
    [[ -n "$SIGMA_FILTER" && "$sigma_label" != "$SIGMA_FILTER" ]] && continue

    yaml="openpto/config/models/${SIGMA_YAML[$sigma_label]}"

    for lr in "${LRS[@]}"; do
        [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]] && continue

        prefix="kn_plain_sigma_sweep_${sigma_label}_lr${lr}"
        job_name="kn_pln_${sigma_label}_lr${lr//-/}"

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
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn_pln"; fi
echo ""
echo "After jobs complete, run:"
echo "  python rethink_exp/fig_kn_plain_sigma_sweep.py"
