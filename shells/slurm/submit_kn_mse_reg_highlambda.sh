#!/usr/bin/env bash
# ====================================================================
# MSE regularization high-λ sweep + balanced λ=0.033
# ====================================================================
# λ ∈ {0.033, 20, 50, 100}, lr=5e-3 only (best lr from prior sweep)
# Total: 4 jobs
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_mse_reg_w{label}_lr5e-3/
#
# Usage:
#   bash shells/slurm/submit_kn_mse_reg_highlambda.sh --dry-run
#   bash shells/slurm/submit_kn_mse_reg_highlambda.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb"
mkdir -p "$SCRIPT_DIR/logs/slurm"

BASE_ARGS="--problem knapsack --opt_model perturb --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--n_epochs 300 --n_ptr_epochs 0 \
--opt_name gd --batch_size 400 \
--pred_model dense \
--gpu -1 --loadnew False"

LR="5e-3"

is_completed() { [[ -f "$RESULTS_BASE/${1}/results.npy" ]]; }

submit_job() {
    local job_name="$1" cmd="$2"
    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
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
$cmd
")
    echo "  $job_name → job $jid"
}

echo "======================================================================"
echo "MSE reg high-λ sweep (λ=0.033, 20, 50, 100) + balanced λ=0.033"
echo "Partition: $PARTITION   Time: $TIME   LR: $LR"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
echo "======================================================================"

submitted=0; skipped=0

declare -A JOBS
JOBS[w0033]="perturb_kn_reg_w0033_n100.yaml"
JOBS[w20]="perturb_kn_reg_w20_n100.yaml"
JOBS[w50]="perturb_kn_reg_w50_n100.yaml"
JOBS[w100]="perturb_kn_reg_w100_n100.yaml"
LABELS=(w0033 w20 w50 w100)

for w_label in "${LABELS[@]}"; do
    yaml="${JOBS[$w_label]}"
    prefix="kn_plain_mse_reg_${w_label}_lr${LR}"
    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "  [skip] $prefix"
        skipped=$((skipped+1)); continue
    fi
    cmd="python rethink_exp/main_results.py $BASE_ARGS \
--method_path openpto/config/models/${yaml} \
--lr $LR --prefix $prefix"
    submit_job "mse_reg_${w_label}_lr${LR}" "$cmd"
    submitted=$((submitted+1))
done

echo ""
echo "Submitted: $submitted   Skipped: $skipped"
