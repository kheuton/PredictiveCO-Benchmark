#!/usr/bin/env bash
# ====================================================================
# Re-run best perturbed config to save model checkpoint for loss
# landscape interpolation experiment.
#
# Best config: per-instance OCV_Y, lr=1e-2, t=0.066, DP, n=100, 300ep
# Output: kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt
#
# Usage:
#   bash shells/slurm/submit_kn_landscape_ckpt.sh --dry-run
#   bash shells/slurm/submit_kn_landscape_ckpt.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

JOB_NAME="kn_landscape_ckpt"
CKPT_PATH="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt"

# Delete existing .npz so the skip-if-exists check doesn't fire
NPZ_PATH="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense.npz"

CMD="python -u rethink_exp/perturb_adaptive_sweep.py \
--problem knapsack \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--mode proportional --gpu -1 \
--solver heuristic --n_samples 100 --n_epochs 300 \
--pred_models dense \
--per_instance \
--lr 1e-2 \
--prop_targets 0.066 \
--prefix kn_bench_dp_per_instance_lr1e-2"

echo "======================================================================"
echo "Loss landscape checkpoint run"
echo "  Best perturbed config: per-instance OCV_Y, lr=1e-2, t=0.066"
echo "  Will save: pi_t0.066_s1_dense_best_pred.pt"
echo "Partition: $PARTITION  Time: $TIME"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
echo "======================================================================"

if $DRY_RUN; then
    echo "[DRY RUN] $JOB_NAME"
    echo "  Would delete: $NPZ_PATH"
    echo "  CMD: $CMD"
    exit 0
fi

# Remove existing .npz to force re-run (checkpoint won't be overwritten otherwise)
if [[ -f "$NPZ_PATH" ]]; then
    echo "Removing existing .npz to force re-run: $NPZ_PATH"
    rm "$NPZ_PATH"
fi

jid=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task=2 \
    --output="$SCRIPT_DIR/logs/slurm/${JOB_NAME}_%j.out" \
    --error="$SCRIPT_DIR/logs/slurm/${JOB_NAME}_%j.err" \
    --export=ALL \
    --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV
echo 'Job: $JOB_NAME  Start:' \$(date)
$CMD
echo 'End:' \$(date)
")
echo "  $JOB_NAME → job $jid"
echo "  Monitor: squeue -u \$USER | grep kn_landscape"
echo "  Checkpoint will be at: $CKPT_PATH"
