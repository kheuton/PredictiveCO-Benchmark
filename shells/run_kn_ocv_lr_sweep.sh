#!/usr/bin/env bash
# Adaptive sigma sweep for knapsack at 3 LRs (global OCV_Y controller).
# Uses heuristic DP solver for fast iteration.
# Run sequentially to avoid CPU contention.

set -e
cd "$(dirname "$0")/.."

CONDA_ENV="pco_bench_rhel7"
BASE_ARGS="--problem knapsack --solver heuristic \
           --config_path openpto/config/probs/knapsack_small.yaml \
           --mode proportional \
           --n_epochs 150 --n_samples 100 \
           --pred_models dense poly"

for lr in 5e-2 1e-2 5e-3; do
    prefix="kn_ocv_lr${lr}"
    echo "=============================="
    echo "LR = ${lr}  prefix = ${prefix}"
    echo "=============================="
    conda run -n "$CONDA_ENV" python -u rethink_exp/perturb_adaptive_sweep.py \
        $BASE_ARGS --lr "$lr" --prefix "$prefix"
done

echo "All done."
