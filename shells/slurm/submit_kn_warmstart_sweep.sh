#!/usr/bin/env bash
# ====================================================================
# Warm-start perturbed method from MSE checkpoint.
#
# Runs the best perturbed config (pi OCV_Y, lr=1e-2, t=0.066, DP n=100)
# initialized from the MSE best checkpoint instead of random weights.
# Compare against existing cold-start result:
#   kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense.npz
#
# Output: kn_bench_warmstart/pi_t0.066_s1_dense.npz
#
# Usage:
#   bash shells/slurm/submit_kn_warmstart_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_warmstart_sweep.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="batch"
TIME="30:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

MSE_CKPT="$SCRIPT_DIR/saved_records/knapsack-gen/mse/kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt"
JOB_NAME="kn_ws_warm"

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
--warmstart_ckpt $MSE_CKPT \
--prefix kn_bench_warmstart"

echo "======================================================================"
echo "Knapsack warm-start: perturbed method initialized from MSE checkpoint"
echo "  Config: per-instance OCV_Y, lr=1e-2, t=0.066, DP n=100, 300ep"
echo "  Init:   MSE best weights (lr=5e-2)"
echo "  Output: kn_bench_warmstart/pi_t0.066_s1_dense.npz"
echo "  Compare against: kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense.npz"
echo "Partition: $PARTITION  Time: $TIME"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
echo "======================================================================"

if $DRY_RUN; then
    echo "[DRY RUN] $JOB_NAME"
    echo "  CMD: $CMD"
    exit 0
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
echo ""
echo "Monitor: squeue -u \$USER | grep kn_ws"
echo "Visualize when done:"
echo "  conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_warmstart.py"
