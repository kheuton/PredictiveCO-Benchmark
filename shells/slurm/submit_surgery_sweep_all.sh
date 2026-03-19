#!/usr/bin/env bash
# ====================================================================
# Multi-problem gradient surgery sweep  (constant sigma)
# ====================================================================
# Tests 5 experiments × 7 CO problems = 35 jobs.
# All perturb runs use main_results.py --opt_name sgd with a fixed-sigma
# YAML config (no adaptive controller).
#
#   Exp 1 (surg_all_mse):       MSE baseline            lr=5e-2
#   Exp 2 (surg_all_plain):     Plain perturb            lr=5e-3
#   Exp 3 (surg_all_w1_lr1e2): Surgery w=1.0            lr=1e-2
#   Exp 4 (surg_all_w1_lr5e3): Surgery w=1.0            lr=5e-3
#   Exp 5 (surg_all_w0_lr1e2): Denoised DPO w=0.0       lr=1e-2
#
# Scientific questions:
#   Exp 3 vs 1: Does surgery beat MSE?
#   Exp 3 vs 2: Does surgery improve over plain perturb?
#   Exp 3 vs 4: Is lr=1e-2 required for surgery?
#   Exp 3 vs 5: Denoising alone (w=0) vs denoising + injecting g_m (w=1)?
#
# Sigma per problem (constant, baked into METHOD_PATH config):
#   knapsack (gen):  sigma=0.5, n=100  (perturb_kn_s05_n100.yaml)
#   knapsack-real:   sigma=0.5, n=10   (perturb_s05_n10.yaml)
#   energy:          sigma=0.5, n=10   (perturb_s05_n10.yaml)
#   budgetalloc:     sigma=0.5, n=10   (perturb_s05_n10.yaml)
#   cubic:           sigma=10.0, n=100 (perturb_cub_s100_n100.yaml)
#   bipartitematching: sigma=0.5, n=20 (perturb_s05_n20.yaml)
#   portfolio:       sigma=0.5, n=20   (perturb_s05_n20.yaml)
#
# Usage:
#   bash shells/slurm/submit_surgery_sweep_all.sh --dry-run
#   bash shells/slurm/submit_surgery_sweep_all.sh
#   bash shells/slurm/submit_surgery_sweep_all.sh --problem knapsack
#   bash shells/slurm/submit_surgery_sweep_all.sh --exp surg_all_w1_lr1e2
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
EXP_FILTER=""
PARTITION="hugheslab,batch"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --problem)            PROB_FILTER="$2"; shift 2 ;;
        --exp)                EXP_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Per-problem configuration
# ====================================================================

# knapsack-real = knapsack problem on real energy data (prob_version: energy),
# distinct from both knapsack-gen (synthetic) and energy (scheduling task).
PROBLEMS=(knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio)

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[knapsack-real]=energy
PROB_VERSION[energy]=energy
PROB_VERSION[budgetalloc]=real
PROB_VERSION[cubic]=gen
PROB_VERSION[bipartitematching]=cora
PROB_VERSION[portfolio]=real

# --problem flag passed to Python (knapsack-real still uses --problem knapsack)
declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[knapsack-real]=knapsack
PROB_ARG[energy]=energy
PROB_ARG[budgetalloc]=budgetalloc
PROB_ARG[cubic]=cubic
PROB_ARG[bipartitematching]=bipartitematching
PROB_ARG[portfolio]=portfolio

# Solver for perturb experiments (2-5)
declare -A PERTURB_SOLVER
PERTURB_SOLVER[knapsack]=heuristic
PERTURB_SOLVER[knapsack-real]=gurobi
PERTURB_SOLVER[energy]=gurobi
PERTURB_SOLVER[budgetalloc]=neural
PERTURB_SOLVER[cubic]=heuristic
PERTURB_SOLVER[bipartitematching]=cvxpy
PERTURB_SOLVER[portfolio]=cvxpy

# Solver for MSE baseline; knapsack-gen uses gurobi (consistent with 0.064 baseline)
declare -A MSE_SOLVER
MSE_SOLVER[knapsack]=gurobi
MSE_SOLVER[knapsack-real]=gurobi
MSE_SOLVER[energy]=gurobi
MSE_SOLVER[budgetalloc]=neural
MSE_SOLVER[cubic]=heuristic
MSE_SOLVER[bipartitematching]=cvxpy
MSE_SOLVER[portfolio]=cvxpy

# Problem config path (--config_path); empty = auto-detected from --problem
declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[knapsack-real]=openpto/config/probs/knapsack-real.yaml
PROB_CONFIG[energy]=""
PROB_CONFIG[budgetalloc]=""
PROB_CONFIG[cubic]=""
PROB_CONFIG[bipartitematching]=""
PROB_CONFIG[portfolio]=""

# Method config (--method_path): fixed sigma + n_samples per problem
declare -A METHOD_PATH
METHOD_PATH[knapsack]=openpto/config/models/perturb_kn_s05_n100.yaml
METHOD_PATH[knapsack-real]=openpto/config/models/perturb_s05_n10.yaml
METHOD_PATH[energy]=openpto/config/models/perturb_s05_n10.yaml
METHOD_PATH[budgetalloc]=openpto/config/models/perturb_s05_n10.yaml
METHOD_PATH[cubic]=openpto/config/models/perturb_cub_s100_n100.yaml
METHOD_PATH[bipartitematching]=openpto/config/models/perturb_s05_n20.yaml
METHOD_PATH[portfolio]=openpto/config/models/perturb_s05_n20.yaml

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[knapsack-real]=400
INSTANCES[energy]=400
INSTANCES[budgetalloc]=400
INSTANCES[cubic]=250
INSTANCES[bipartitematching]=20
INSTANCES[portfolio]=400

# Test instances; must match benchmark defaults (cubic=400, bipartite=6, others=200)
declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[knapsack-real]=200
TESTINSTANCES[energy]=200
TESTINSTANCES[budgetalloc]=200
TESTINSTANCES[cubic]=400
TESTINSTANCES[bipartitematching]=6
TESTINSTANCES[portfolio]=200

# Walltime for perturb experiments
declare -A PERTURB_WALLTIME
PERTURB_WALLTIME[knapsack]=1:00:00
PERTURB_WALLTIME[knapsack-real]=2:00:00
PERTURB_WALLTIME[energy]=24:00:00
PERTURB_WALLTIME[budgetalloc]=2:00:00
PERTURB_WALLTIME[cubic]=1:00:00
PERTURB_WALLTIME[bipartitematching]=2:00:00
PERTURB_WALLTIME[portfolio]=2:00:00

declare -A MSE_WALLTIME_MAP
MSE_WALLTIME_MAP[knapsack]=1:00:00
MSE_WALLTIME_MAP[knapsack-real]=1:00:00
MSE_WALLTIME_MAP[energy]=4:00:00
MSE_WALLTIME_MAP[budgetalloc]=1:00:00
MSE_WALLTIME_MAP[cubic]=1:00:00
MSE_WALLTIME_MAP[bipartitematching]=1:00:00
MSE_WALLTIME_MAP[portfolio]=1:00:00

# ====================================================================
# Helpers
# ====================================================================

# Output dir base: {prob_arg}-{version}, e.g. knapsack-gen, knapsack-energy
prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

is_completed_mse() {
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $1)/mse/$2/results.npy"
    [[ -f "$out" ]]
}

is_completed_perturb() {
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $1)/perturb/$2/results.npy"
    [[ -f "$out" ]]
}

n_submitted=0
n_skipped=0

# ====================================================================
# Submit helpers
# ====================================================================

submit_mse() {
    local prob="$1"
    local prefix="surg_all_mse"

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "$prefix" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_mse "$prob" "$prefix"; then
        echo "  [skip] ${prob} ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local prob_arg="${PROB_ARG[$prob]}"
    local solver="${MSE_SOLVER[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local walltime="${MSE_WALLTIME_MAP[$prob]}"
    local job_name="surg_mse_${prob}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local cmd="python rethink_exp/main_results.py \
        --problem ${prob_arg} \
        --opt_model mse \
        --solver ${solver} \
        --lr 5e-2 \
        --n_epochs 300 \
        --instances ${instances} \
        --testinstances ${testinstances} \
        --seed 2023 \
        --prefix ${prefix} \
        ${cfg_flag}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${walltime})"
        echo "  output: saved_records/$(prob_out_dir $prob)/mse/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${walltime}" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

submit_perturb() {
    local prob="$1"
    local prefix="$2"
    local extra_flags="$3"   # --grad_surgery --surgery_weight W  (or empty for plain)

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "$prefix" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_perturb "$prob" "$prefix"; then
        echo "  [skip] ${prob} ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local prob_arg="${PROB_ARG[$prob]}"
    local solver="${PERTURB_SOLVER[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local method="${METHOD_PATH[$prob]}"
    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local walltime="${PERTURB_WALLTIME[$prob]}"
    local job_name="surg_${prefix##surg_all_}_${prob}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local cmd="python rethink_exp/main_results.py \
        --problem ${prob_arg} \
        --opt_model perturb \
        --solver ${solver} \
        --n_epochs 300 \
        --instances ${instances} \
        --testinstances ${testinstances} \
        --seed 2023 \
        --method_path ${method} \
        --prefix ${prefix} \
        ${cfg_flag} \
        ${extra_flags}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${walltime})"
        echo "  output: saved_records/$(prob_out_dir $prob)/perturb/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="$walltime" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

# ====================================================================
# Main sweep: 7 problems × 5 experiments = 35 jobs
# ====================================================================

echo "=== Multi-problem gradient surgery sweep (constant sigma) ==="
echo "Problems:    ${PROBLEMS[*]}"
echo "             (knapsack-real = knapsack on real energy data)"
echo "Experiments: surg_all_mse | surg_all_plain | surg_all_w1_lr1e2 | surg_all_w1_lr5e3 | surg_all_w0_lr1e2"
echo "Partition:   $PARTITION"
echo ""

for prob in "${PROBLEMS[@]}"; do
    if [[ -n "$PROB_FILTER" && "$prob" != "$PROB_FILTER" ]]; then continue; fi

    echo "--- ${prob} (sigma=${METHOD_PATH[$prob]##*_s}, method=${METHOD_PATH[$prob]##*/}) ---"

    # Exp 1: MSE baseline
    submit_mse "$prob"

    # Exp 2: Plain perturb, no surgery, lr=5e-3
    submit_perturb "$prob" "surg_all_plain" "--lr 5e-3"

    # Exp 3: Surgery w=1.0, lr=1e-2
    submit_perturb "$prob" "surg_all_w1_lr1e2" "--grad_surgery --surgery_weight 1.0 --lr 1e-2"

    # Exp 4: Surgery w=1.0, lr=5e-3
    submit_perturb "$prob" "surg_all_w1_lr5e3" "--grad_surgery --surgery_weight 1.0 --lr 5e-3"

    # Exp 5: Denoised DPO w=0.0, lr=1e-2 (g_p^perp only, no g_m injection)
    submit_perturb "$prob" "surg_all_w0_lr1e2" "--grad_surgery --surgery_weight 0.0 --lr 1e-2"

    echo ""
done

echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
