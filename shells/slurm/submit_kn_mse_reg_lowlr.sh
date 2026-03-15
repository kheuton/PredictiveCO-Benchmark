#!/usr/bin/env bash
# ====================================================================
# Extended MSE regularization sweep — lower learning rates + λ=5,10
# ====================================================================
# Existing λ values: add lr ∈ {1e-3, 5e-4}  (2 new LRs × 6 λ = 12 jobs)
# New λ values (5, 10): run lr ∈ {5e-3, 1e-3, 5e-4}      (3 × 2 = 6 jobs)
# Total: 18 jobs
#
# Results: saved_records/knapsack-gen/perturb/kn_plain_mse_reg_w{w}_lr{lr}/
#
# Usage:
#   bash shells/slurm/submit_kn_mse_reg_lowlr.sh --dry-run
#   bash shells/slurm/submit_kn_mse_reg_lowlr.sh
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
echo "MSE reg lower-LR sweep + λ=5,10"
echo "Partition: $PARTITION   Time: $TIME"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
echo "======================================================================"

submitted=0; skipped=0

# --- Existing lambdas: add lr=1e-3 and lr=5e-4 ---
declare -A EXISTING_YAML
EXISTING_YAML[w0001]="perturb_kn_reg_w0001_n100.yaml"
EXISTING_YAML[w001]="perturb_kn_reg_w001_n100.yaml"
EXISTING_YAML[w005]="perturb_kn_reg_w005_n100.yaml"
EXISTING_YAML[w01]="perturb_kn_reg_w01_n100.yaml"
EXISTING_YAML[w05]="perturb_kn_reg_w05_n100.yaml"
EXISTING_YAML[w1]="perturb_kn_reg_w1_n100.yaml"
EXISTING_LABELS=(w0001 w001 w005 w01 w05 w1)
NEW_LRS=("1e-3" "5e-4")

for w_label in "${EXISTING_LABELS[@]}"; do
    yaml="${EXISTING_YAML[$w_label]}"
    for lr in "${NEW_LRS[@]}"; do
        prefix="kn_plain_mse_reg_${w_label}_lr${lr}"
        if $SKIP_COMPLETED && is_completed "$prefix"; then
            echo "  [skip] $prefix"
            skipped=$((skipped+1)); continue
        fi
        cmd="python rethink_exp/main_results.py $BASE_ARGS \
--method_path openpto/config/models/${yaml} \
--lr $lr --prefix $prefix"
        submit_job "mse_reg_${w_label}_lr${lr}" "$cmd"
        submitted=$((submitted+1))
    done
done

# --- New lambdas: w5 and w10, lr=5e-3, 1e-3, 5e-4 ---
declare -A NEW_YAML
NEW_YAML[w5]="perturb_kn_reg_w5_n100.yaml"
NEW_YAML[w10]="perturb_kn_reg_w10_n100.yaml"
NEW_LABELS=(w5 w10)
ALL_LRS=("5e-3" "1e-3" "5e-4")

for w_label in "${NEW_LABELS[@]}"; do
    yaml="${NEW_YAML[$w_label]}"
    for lr in "${ALL_LRS[@]}"; do
        prefix="kn_plain_mse_reg_${w_label}_lr${lr}"
        if $SKIP_COMPLETED && is_completed "$prefix"; then
            echo "  [skip] $prefix"
            skipped=$((skipped+1)); continue
        fi
        cmd="python rethink_exp/main_results.py $BASE_ARGS \
--method_path openpto/config/models/${yaml} \
--lr $lr --prefix $prefix"
        submit_job "mse_reg_${w_label}_lr${lr}" "$cmd"
        submitted=$((submitted+1))
    done
done

echo ""
echo "Submitted: $submitted   Skipped: $skipped"
