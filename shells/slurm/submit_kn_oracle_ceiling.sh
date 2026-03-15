#!/usr/bin/env bash
# Submit oracle performance-ceiling experiment for knapsack.
# Trains a dense MLP on noise-free oracle data (same DGP, noise_width=0),
# then evaluates regret on the benchmark test set (Knapsack_7.pkl).
#
# Usage:
#   bash shells/slurm/submit_kn_oracle_ceiling.sh [--dry-run]

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_DIR}/slurm_logs/kn_oracle_ceiling"
mkdir -p "${LOG_DIR}"

JOB_NAME="kn_oracle_ceiling"

CMD="python rethink_exp/fig_kn_oracle_ceiling.py \
  --n_oracle 10000 \
  --n_epochs 2000 \
  --lr 0.05 \
  --n_layers 2 \
  --n_hidden 32"

SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --output="${LOG_DIR}/${JOB_NAME}_%j.out"
  --error="${LOG_DIR}/${JOB_NAME}_%j.err"
  --partition=batch
  --cpus-per-task=2
  --mem=8G
  --time=0:30:00
)

SCRIPT="#!/usr/bin/env bash
#SBATCH ${SBATCH_ARGS[*]}

set -euo pipefail
source ~/.bashrc
conda activate pco_bench_rhel7
cd ${REPO_DIR}
${CMD}
"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== DRY RUN ==="
  echo "SBATCH options: ${SBATCH_ARGS[*]}"
  echo "CMD: ${CMD}"
else
  echo "$SCRIPT" | sbatch
  echo "Submitted: ${JOB_NAME}"
fi
