#!/usr/bin/env bash
# Run per-instance adaptive sigma sweep at 3 learning rates.
# 15 log-spaced targets × 2 models × 3 LRs = 90 runs, executed sequentially.
#
# Usage:
#   bash shells/run_pi_lr_sweep.sh

set -e
cd "$(dirname "$0")/.."

CONDA_ENV="pco_bench_rhel7"
BASE_ARGS="--problem cubic --mode proportional --per_instance \
           --n_epochs 300 --n_samples 100 --pred_models dense poly"

for lr in 5e-2 1e-2 5e-3; do
    prefix="pi_lr${lr}"
    echo "=============================="
    echo "LR = ${lr}  prefix = ${prefix}"
    echo "=============================="
    conda run -n "$CONDA_ENV" python -u rethink_exp/perturb_adaptive_sweep.py \
        $BASE_ARGS --lr "$lr" --prefix "$prefix"
done

echo "All done."
