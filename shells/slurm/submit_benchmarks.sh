#!/usr/bin/env bash
# =============================================================================
# SLURM Submission Script for PredictiveCO Benchmark
# =============================================================================
# Submits one SLURM job per (problem, method) pair. Also submits perturb
# experiments with multiple sigma values.
#
# Usage:
#   # Submit ALL problems, ALL methods (including perturb with sigma sweep):
#   ./shells/slurm/submit_benchmarks.sh --all
#
#   # Single problem, all methods:
#   ./shells/slurm/submit_benchmarks.sh --problem portfolio
#
#   # Single problem, single method:
#   ./shells/slurm/submit_benchmarks.sh --problem portfolio --method mse
#
#   # Only perturb experiments (all problems):
#   ./shells/slurm/submit_benchmarks.sh --all --perturb-only
#
#   # Only existing baselines (no perturb):
#   ./shells/slurm/submit_benchmarks.sh --all --no-perturb
#
#   # Custom sigma values (comma-separated):
#   ./shells/slurm/submit_benchmarks.sh --all --perturb-only --sigmas "0.01,0.1,0.5,1.0,2.0"
#
#   # Dry run (print sbatch commands without submitting):
#   ./shells/slurm/submit_benchmarks.sh --all --dry-run
#
# =============================================================================

set -e

# ---- Defaults ----
PROBLEM=""
METHOD=""
ALL_PROBLEMS=false
DRY_RUN=false
PERTURB_ONLY=false
NO_PERTURB=false
SIGMAS="0.1,0.5,1.0"
PARTITION="preempt"
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="ieo_rhel7"
PREFIX="bench"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)       PROBLEM="$2"; shift 2 ;;
        --method)        METHOD="$2"; shift 2 ;;
        --all)           ALL_PROBLEMS=true; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --perturb-only)  PERTURB_ONLY=true; shift ;;
        --no-perturb)    NO_PERTURB=true; shift ;;
        --sigmas)        SIGMAS="$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --time)          TIME="$2"; shift 2 ;;
        --mem)           MEM="$2"; shift 2 ;;
        --gres)          GRES="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --conda-env)     CONDA_ENV="$2"; shift 2 ;;
        --prefix)        PREFIX="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! $ALL_PROBLEMS && [[ -z "$PROBLEM" ]]; then
    echo "Error: must specify --problem <name> or --all"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/"

# Create log directory
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Problem definitions ----
# Each problem is encoded as a function that emits (method, extra_args) lines.
# This directly mirrors the existing benchmark shell scripts.

get_methods_portfolio() {
    local E=300
    cat <<EOF
mse|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --loadnew True
dfl|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
blackbox|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
identity|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
cpLayer|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
spo|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
nce|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pointLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
listLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pairLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
lodl|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --method_path "openpto/config/models/lodl50k.yaml" --data_dir ${DIR}
EOF
}

get_perturb_base_portfolio() {
    echo "--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
}

get_methods_knapsack() {
    local E=300
    cat <<EOF
mse|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --loadnew True
dfl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 10000
blackbox|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}"
identity|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}"
cpLayer|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}"
spo|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1
nce|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1
pointLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1
listLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1
pairLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1
lodl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --method_path "openpto/config/models/lodl/quad5.yaml"
EOF
}

get_perturb_base_knapsack() {
    echo "--solver gurobi --n_epochs 300 --gpu ${GPU_ID}"
}

get_methods_knapsack_energy() {
    local E=300
    local PTH="./openpto/config/probs/knapsack-real.yaml"
    cat <<EOF
mse|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --loadnew True
dfl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 10000
blackbox|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR}
identity|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR}
cpLayer|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR}
spo|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR}
nce|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
pointLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
listLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
pairLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
lodl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --config_path ${PTH} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
EOF
}

get_perturb_base_knapsack_energy() {
    echo "--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --config_path ./openpto/config/probs/knapsack-real.yaml --data_dir ${DIR}"
}

get_methods_cubic() {
    local E=300
    cat <<EOF
mse|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --loadnew True
dfl|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}"
blackbox|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}"
identity|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}"
spo|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}"
nce|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --batch_size 1
pointLTR|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --batch_size 1
listLTR|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --batch_size 1
pairLTR|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --batch_size 1
lodl|--solver heuristic --n_epochs $E --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400 --prefix "${PREFIX}" --method_path "openpto/config/models/lodl/lodl5k.yaml"
EOF
}

get_perturb_base_cubic() {
    echo "--solver heuristic --n_epochs 300 --gpu ${GPU_ID} --lr 5e-2 --instances 250 --testinstances 400"
}

get_methods_budgetalloc() {
    local E=300
    cat <<EOF
mse|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
dfl|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 10000
blackbox|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
identity|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
spo|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
nce|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
pointLTR|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
listLTR|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
pairLTR|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1
lodl|--solver neural --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 1 --method_path "openpto/config/models/lodl5k.yaml"
EOF
}

get_perturb_base_budgetalloc() {
    echo "--solver neural --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
}

get_methods_energy() {
    local E=300
    cat <<EOF
mse|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --loadnew True
dfl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR} --batch_size 10000
blackbox|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
identity|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
spo|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --data_dir ${DIR}
nce|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pointLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pairLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
listLTR|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
lodl|--solver gurobi --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR} --method_path "openpto/config/models/lodl/lodl5k.yaml"
EOF
}

get_perturb_base_energy() {
    echo "--solver gurobi --n_epochs 300 --gpu ${GPU_ID} --data_dir ${DIR}"
}

get_methods_bipartitematching() {
    local E=300
    cat <<EOF
bce|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --data_dir ${DIR} --loadnew True
dfl|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --data_dir ${DIR}
blackbox|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --data_dir ${DIR}
identity|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --data_dir ${DIR}
cpLayer|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --data_dir ${DIR}
spo|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
nce|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pointLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
listLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
pairLTR|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
lodl|--solver cvxpy --n_epochs $E --gpu ${GPU_ID} --instances 20 --testinstances 6 --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR} --method_path "openpto/config/models/lodl/lodl5k.yaml"
EOF
}

get_perturb_base_bipartitematching() {
    echo "--solver cvxpy --n_epochs 300 --gpu ${GPU_ID} --instances 20 --testinstances 6 --data_dir ${DIR}"
}

get_methods_advertising() {
    local E=150
    local PTR=150
    cat <<EOF
bce|--pred_model cvr --solver ortools --n_ptr_epochs $PTR --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR} --loadnew True
dfl|--pred_model cvr --solver ortools --n_ptr_epochs $PTR --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
identity|--pred_model cvr --solver ortools --n_ptr_epochs $PTR --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
blackbox|--pred_model cvr --solver ortools --n_ptr_epochs $PTR --n_epochs $E --gpu ${GPU_ID} --prefix "${PREFIX}" --batch_size 1 --data_dir ${DIR}
EOF
}

get_perturb_base_advertising() {
    echo "--pred_model cvr --solver ortools --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU_ID} --batch_size 1 --data_dir ${DIR}"
}

# ---- List of all problems ----
ALL_PROBLEM_NAMES="portfolio knapsack knapsack_energy cubic budgetalloc energy bipartitematching advertising"

# Map problem key to --problem arg (knapsack_energy uses --problem=knapsack with a config override)
get_problem_arg() {
    case "$1" in
        knapsack_energy) echo "knapsack" ;;
        *) echo "$1" ;;
    esac
}

# ---- Submit one job ----
submit_job() {
    local problem_key="$1"
    local method="$2"
    local extra_args="$3"
    local job_prefix="$4"  # for perturb sigma labeling

    local problem_arg
    problem_arg=$(get_problem_arg "$problem_key")

    # Build sanitized job name
    local safe_method="${method//+/p}"
    local job_name="${job_prefix}_${problem_key}_${safe_method}"

    local cmd="python rethink_exp/main_results.py --problem=${problem_arg} --opt_model ${method} ${extra_args}"

    if $DRY_RUN; then
        echo "[DRY RUN] sbatch --job-name=$job_name"
        echo "          CMD: $cmd"
        echo ""
        return
    fi

    echo "Submitting: $job_name"

    sbatch \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --gres="$GRES" \
        --cpus-per-task=1 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV

echo '=============================================='
echo 'Job: $job_name'
echo 'Problem: $problem_key  Method: $method'
echo 'CMD: $cmd'
echo 'Python:' \$(which python)
echo 'Start:' \$(date)
echo '=============================================='

$cmd

echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
"
}

# ---- Submit perturb jobs for one problem ----
submit_perturb_jobs() {
    local problem_key="$1"
    local base_args
    base_args=$(eval "get_perturb_base_${problem_key}")

    IFS=',' read -ra SIGMA_ARRAY <<< "$SIGMAS"
    for sigma in "${SIGMA_ARRAY[@]}"; do
        # Map sigma to yaml filename tag: 0.1 -> 01, 0.5 -> 05, 1.0 -> 10
        # Remove the dot, keep all digits (preserving leading zeros for fractional part)
        local sigma_tag
        sigma_tag=$(echo "$sigma" | sed 's/\.//')
        # Remove a single leading zero only if the result has 2+ digits
        # 0.1 -> "01", 0.5 -> "05", 1.0 -> "10", 0.01 -> "001", 2.0 -> "20"
        local yaml_path="openpto/config/models/perturb_s${sigma_tag}.yaml"
        local perturb_prefix="${PREFIX}_perturb_s${sigma_tag}"

        # Check if the yaml exists; if not, we'll create it on-the-fly via the default
        # and pass sigma through the yaml. For known sigmas we have pre-made yamls.
        local method_path_arg=""
        if [[ -f "$SCRIPT_DIR/$yaml_path" ]]; then
            method_path_arg="--method_path $yaml_path"
        else
            echo "WARNING: $yaml_path not found. Creating it from default.yaml..."
            # Copy default.yaml and replace the perturb section to include sigma
            python3 -c "
import sys
with open('$SCRIPT_DIR/openpto/config/models/default.yaml') as f:
    content = f.read()
content = content.replace('perturb: \n  reduction: mean', 'perturb: \n  reduction: mean\n  sigma: ${sigma}')
with open('$SCRIPT_DIR/$yaml_path', 'w') as f:
    f.write(content)
print('Created $yaml_path with sigma=${sigma}')
"
            method_path_arg="--method_path $yaml_path"
        fi

        local full_args="${base_args} --prefix \"${perturb_prefix}\" ${method_path_arg}"
        submit_job "$problem_key" "perturb" "$full_args" "pco"
    done
}

# ---- Submit all methods for one problem ----
submit_problem() {
    local problem_key="$1"

    # Submit baseline methods
    if ! $PERTURB_ONLY; then
        while IFS='|' read -r method extra_args; do
            # If a specific method filter is set, skip non-matching
            if [[ -n "$METHOD" && "$method" != "$METHOD" ]]; then
                continue
            fi
            submit_job "$problem_key" "$method" "$extra_args" "pco"
        done < <(eval "get_methods_${problem_key}")
    fi

    # Submit perturb jobs
    if ! $NO_PERTURB; then
        if [[ -z "$METHOD" || "$METHOD" == "perturb" ]]; then
            submit_perturb_jobs "$problem_key"
        fi
    fi
}

# ---- Main ----
if $ALL_PROBLEMS; then
    PROBLEMS_TO_RUN=($ALL_PROBLEM_NAMES)
elif [[ -n "$PROBLEM" ]]; then
    # Normalize problem name (allow "knapsack-energy" -> "knapsack_energy")
    PROBLEM="${PROBLEM//-/_}"
    PROBLEMS_TO_RUN=("$PROBLEM")
else
    echo "Error: must specify --problem <name> or --all"
    exit 1
fi

echo "=============================================="
echo "PredictiveCO Benchmark SLURM Submission"
echo "=============================================="
echo "Problems: ${PROBLEMS_TO_RUN[*]}"
echo "Sigmas:   $SIGMAS"
echo "Prefix:   $PREFIX"
echo "Partition: $PARTITION"
echo "Time: $TIME  Mem: $MEM  GRES: $GRES"
if $DRY_RUN; then
    echo "*** DRY RUN MODE ***"
fi
echo "=============================================="
echo ""

for prob in "${PROBLEMS_TO_RUN[@]}"; do
    echo "--- Problem: $prob ---"
    submit_problem "$prob"
    echo ""
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "All jobs submitted. Check with: squeue -u \$USER"
fi
