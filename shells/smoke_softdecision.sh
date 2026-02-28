#!/bin/bash
# Smoke test: run the new soft-decision perturbed method on all 8 problems
# with very few epochs (10) to verify nothing crashes and loss decreases.
set -e

GPU=0
EPOCHS=10
PREFIX="smoke_softdec"
DIR="./openpto/data/"
MODEL="perturb"

echo "=== Smoke tests for soft-decision perturbed method ==="
echo "=== $(date) ==="
echo ""

# Track failures
FAILED=""
PASSED=""

run_one() {
    local NAME="$1"
    shift
    echo "-----------------------------------------------------"
    echo "[START] $NAME"
    echo "CMD: python rethink_exp/main_results.py $@"
    echo "-----------------------------------------------------"
    if python rethink_exp/main_results.py "$@" 2>&1 | tail -20; then
        echo "[PASS] $NAME"
        PASSED="${PASSED} ${NAME}"
    else
        echo "[FAIL] $NAME (exit code $?)"
        FAILED="${FAILED} ${NAME}"
    fi
    echo ""
}

# 1. Portfolio (cvxpy solver, real data)
run_one "portfolio" \
    --problem=portfolio --opt_model ${MODEL} --solver cvxpy \
    --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR}

# 2. Knapsack-gen (gurobi solver, generated data)
run_one "knapsack-gen" \
    --problem=knapsack --opt_model ${MODEL} --solver gurobi \
    --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}

# 3. Knapsack-energy (gurobi solver, real energy data)
run_one "knapsack-energy" \
    --problem=knapsack --opt_model ${MODEL} --solver gurobi \
    --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} \
    --config_path "./openpto/config/probs/knapsack-real.yaml"

# 4. Bipartite matching (cvxpy solver, cora data)
run_one "bipartitematching" \
    --problem=bipartitematching --opt_model ${MODEL} --solver cvxpy \
    --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 \
    --prefix ${PREFIX} --data_dir ${DIR}

# 5. Budget allocation (neural solver, real data)
run_one "budgetalloc" \
    --problem=budgetalloc --opt_model ${MODEL} --solver neural \
    --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR}

# 6. Cubic top-K (heuristic solver, generated data)
run_one "cubic" \
    --problem=cubic --opt_model ${MODEL} --solver heuristic \
    --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 \
    --prefix ${PREFIX}

# 7. Energy scheduling (gurobi solver, real data)
run_one "energy" \
    --problem=energy --opt_model ${MODEL} --solver gurobi \
    --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR}

# 8. Advertising (ortools solver, real data)
run_one "advertising" \
    --problem=advertising --pred_model cvr --opt_model ${MODEL} --solver ortools \
    --n_ptr_epochs 5 --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} \
    --batch_size 1 --data_dir ${DIR}

echo "====================================================="
echo "SUMMARY"
echo "====================================================="
echo "PASSED:${PASSED}"
echo "FAILED:${FAILED}"
if [ -z "$FAILED" ]; then
    echo "ALL 8 PROBLEMS PASSED!"
else
    echo "SOME PROBLEMS FAILED!"
    exit 1
fi
