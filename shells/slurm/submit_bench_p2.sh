#!/usr/bin/env bash
# ====================================================================
# Benchmark re-run — Phase 2: Method-specific HP sweep
# ====================================================================
# For each method that has a tunable HP, sweeps that HP using the best
# (lr, batch) config found in Phase 1.
#
# Reads bench_p1_best.json (produced by collect_bench_p1.py) to pick
# the optimal (lr, batch) per method × task.
#
# Methods with HPs:
#   dfl      : dflalpha  ∈ {0.001, 0.01, 0.1*, 1.0, 10.0}
#   blackbox : lambd     ∈ {0.01, 0.05, 0.1*, 0.5, 1.0}
#   qptl     : tau       ∈ {0.1, 0.5, 1.0*, 5.0, 10.0}
#   listLTR  : tau       ∈ {0.1, 0.5, 1*, 5, 10}
#   lodl     : num_samples ∈ {100, 250, 500*, 1000, 2000}
#   perturb  : sigma     ∈ {0.1, 0.5, 1.0*, 2.0, 5.0}
#   perturb  : n_samples ∈ {5, 10*, 25, 50, 100}
#
# (* = Phase 1 default, included for completeness)
#
# Usage:
#   bash shells/slurm/submit_bench_p2.sh --dry-run
#   bash shells/slurm/submit_bench_p2.sh
#   bash shells/slurm/submit_bench_p2.sh --problem knapsack
#   bash shells/slurm/submit_bench_p2.sh --method dfl
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
METHOD_FILTER=""
PARTITION="preempt"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true
BEST_JSON="bench_p1_best.json"
MANIFEST_FILE="sweep_manifest_p2.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --problem)           PROB_FILTER="$2"; shift 2 ;;
        --method)            METHOD_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --best-json)         BEST_JSON="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Load Phase 1 best configs ----
if [[ ! -f "$SCRIPT_DIR/$BEST_JSON" ]]; then
    echo "ERROR: $BEST_JSON not found. Run collect_bench_p1.py first."
    exit 1
fi

# ====================================================================
# Problem configuration (same as Phase 1)
# ====================================================================

PROBLEMS=(knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio asurv cook_county speed_humps)

declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[knapsack-real]=knapsack
PROB_ARG[energy]=energy
PROB_ARG[budgetalloc]=budgetalloc
PROB_ARG[cubic]=cubic
PROB_ARG[bipartitematching]=bipartitematching
PROB_ARG[portfolio]=portfolio
PROB_ARG[asurv]=asurv
PROB_ARG[cook_county]=cook_county
PROB_ARG[speed_humps]=speed_humps

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[knapsack-real]=energy
PROB_VERSION[energy]=energy
PROB_VERSION[budgetalloc]=real
PROB_VERSION[cubic]=gen
PROB_VERSION[bipartitematching]=cora
PROB_VERSION[portfolio]=real
PROB_VERSION[asurv]=real
PROB_VERSION[cook_county]=real
PROB_VERSION[speed_humps]=real

declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[knapsack-real]=openpto/config/probs/knapsack-real.yaml
PROB_CONFIG[energy]=""
PROB_CONFIG[budgetalloc]=""
PROB_CONFIG[cubic]=""
PROB_CONFIG[bipartitematching]=""
PROB_CONFIG[portfolio]=""
PROB_CONFIG[asurv]=openpto/config/probs/asurv.yaml
PROB_CONFIG[cook_county]=openpto/config/probs/cook_county.yaml
PROB_CONFIG[speed_humps]=openpto/config/probs/speed_humps.yaml

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[knapsack-real]=400
INSTANCES[energy]=400
INSTANCES[budgetalloc]=400
INSTANCES[cubic]=250
INSTANCES[bipartitematching]=20
INSTANCES[portfolio]=400
INSTANCES[asurv]=400             # silently ignored; dataset is fixed
INSTANCES[cook_county]=400       # silently ignored; dataset is fixed
INSTANCES[speed_humps]=400       # silently ignored; dataset is fixed

declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[knapsack-real]=200
TESTINSTANCES[energy]=200
TESTINSTANCES[budgetalloc]=200
TESTINSTANCES[cubic]=400
TESTINSTANCES[bipartitematching]=6
TESTINSTANCES[portfolio]=200
TESTINSTANCES[asurv]=200         # silently ignored
TESTINSTANCES[cook_county]=200   # silently ignored
TESTINSTANCES[speed_humps]=200   # silently ignored

declare -A SOLVER_PTO
SOLVER_PTO[knapsack]=heuristic
SOLVER_PTO[knapsack-real]=gurobi
SOLVER_PTO[energy]=gurobi
SOLVER_PTO[budgetalloc]=neural
SOLVER_PTO[cubic]=heuristic
SOLVER_PTO[bipartitematching]=cvxpy
SOLVER_PTO[portfolio]=cvxpy
SOLVER_PTO[asurv]=heuristic
SOLVER_PTO[cook_county]=heuristic
SOLVER_PTO[speed_humps]=heuristic

declare -A SOLVER_PNO
SOLVER_PNO[knapsack]=heuristic
SOLVER_PNO[knapsack-real]=gurobi
SOLVER_PNO[energy]=gurobi
SOLVER_PNO[budgetalloc]=neural
SOLVER_PNO[cubic]=heuristic
SOLVER_PNO[bipartitematching]=cvxpy
SOLVER_PNO[portfolio]=cvxpy
SOLVER_PNO[asurv]=heuristic
SOLVER_PNO[cook_county]=heuristic
SOLVER_PNO[speed_humps]=heuristic

declare -A SOLVER_CVXPY
SOLVER_CVXPY[knapsack]=heuristic
SOLVER_CVXPY[bipartitematching]=cvxpy
SOLVER_CVXPY[portfolio]=cvxpy

get_walltime() {
    local prob="$1" group="$2"
    case "$group" in
        pto)
            case "$prob" in
                energy) echo "3:00:00" ;;
                *)       echo "1:30:00" ;;
            esac ;;
        pno)
            case "$prob" in
                energy)        echo "26:00:00" ;;
                budgetalloc)   echo "4:00:00" ;;
                *)             echo "2:00:00" ;;
            esac ;;
        lodl)
            case "$prob" in
                energy)        echo "26:00:00" ;;
                *)             echo "4:00:00" ;;
            esac ;;
    esac
}

# ====================================================================
# Method-specific HP definitions
# ====================================================================
# Each HP sweep entry: (method, hp_name, hp_values, yaml_key_or_flag,
#                       solver_group, method_problems, base_yaml)

# Methods that have tunable HPs in Phase 2
# Format stored below in arrays:
#   P2_METHODS: list of method names
#   For each method: HP name, values, solver group, scope

# dfl: dflalpha
DFL_VALS=(0.001 0.01 0.1 1.0 10.0)

# blackbox: lambd
BB_VALS=(0.01 0.05 0.1 0.5 1.0)

# qptl: tau  (knapsack/bipartitematching/portfolio only)
QPTL_VALS=(0.1 0.5 1.0 5.0 10.0)

# listLTR: tau
LISTLTR_VALS=(0.1 0.5 1 5 10)

# lodl: num_samples
LODL_VALS=(100 250 500 1000 2000)

# perturb: sigma (fixed n_samples=10)
PERTURB_SIGMA_VALS=(0.1 0.5 1.0 2.0 5.0)

# pg: sigma (finite difference width)
PG_SIGMA_VALS=(0.01 0.05 0.1 0.5 1.0)

# perturb: n_samples (using best sigma from sigma sweep, or default sigma=1.0 for now)
PERTURB_N_VALS=(5 10 25 50 100)

# ====================================================================
# Helper to read best config from JSON
# ====================================================================

get_best_lr() {
    local method="$1" prob="$2"
    python3 -c "
import json, sys
with open('$SCRIPT_DIR/$BEST_JSON') as f:
    d = json.load(f)
cfg = d.get('$method', {}).get('$prob', {})
print(cfg.get('lr', '1e-2'))
"
}

get_best_batch() {
    local method="$1" prob="$2"
    python3 -c "
import json, sys
with open('$SCRIPT_DIR/$BEST_JSON') as f:
    d = json.load(f)
cfg = d.get('$method', {}).get('$prob', {})
print(cfg.get('batch', 'default'))
"
}

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

is_completed() {
    local prob="$1" method="$2" prefix="$3"
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
    [[ -f "$out" ]]
}

has_checkpoint() {
    local prob="$1" method="$2" prefix="$3"
    local ckpt="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/checkpoints/checkpoint_latest.pt"
    [[ -f "$ckpt" ]]
}

n_submitted=0
n_skipped=0

MANIFEST_TMP="$SCRIPT_DIR/.manifest_p2_tmp.jsonl"
: > "$MANIFEST_TMP"

add_manifest_entry() {
    local prob="$1" method="$2" prefix="$3"
    printf '{"prob":"%s","opt_model":"%s","prefix":"%s"}\n' \
        "$prob" "$method" "$prefix" >> "$MANIFEST_TMP"
}

# Main submit function
submit_p2_job() {
    local prob="$1" method="$2" hp_tag="$3" method_path="$4" solver_group="$5"
    local extra_args="${6:-}"

    if [[ -n "$PROB_FILTER"   && "$prob"   != "$PROB_FILTER"   ]]; then return; fi
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then return; fi

    # Get best lr and batch from Phase 1
    local lr
    local batch_label
    lr=$(get_best_lr "$method" "$prob")
    batch_label=$(get_best_batch "$method" "$prob")

    # If not found in JSON, fall back to defaults
    [[ -z "$lr" || "$lr" == "None" ]] && lr="1e-2"
    [[ -z "$batch_label" || "$batch_label" == "None" ]] && batch_label="default"

    # Determine opt_name and batch_size flag
    local opt_name bs_flag=""
    if [[ "$batch_label" == "default" ]]; then
        # Use method-specific default
        case "$method" in
            spo|nce|pointLTR|pairLTR|listLTR) opt_name=sgd; bs_flag="--batch_size 1" ;;
            *) opt_name=gd ;;
        esac
    else
        opt_name=sgd
        local alt_bs=32
        [[ "$prob" == "bipartitematching" ]] && alt_bs=4
        bs_flag="--batch_size ${alt_bs}"
    fi

    # Determine solver
    local solver
    if [[ "$method" == "qptl" || "$method" == "cpLayer" ]]; then
        solver="${SOLVER_CVXPY[$prob]:-cvxpy}"
    elif [[ "$solver_group" == "pto" ]]; then
        solver="${SOLVER_PTO[$prob]}"
    else
        solver="${SOLVER_PNO[$prob]}"
    fi

    # Extra args: PtO on energy
    if [[ "$solver_group" == "pto" && "$prob" == "energy" ]]; then
        extra_args="$extra_args --skip_solver_eval"
    fi

    local prob_arg="${PROB_ARG[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local walltime
    walltime=$(get_walltime "$prob" "$solver_group")

    # Prefix encodes the hp value
    local prefix="bench_p2_${method}_${hp_tag}_${batch_label}_lr${lr}"

    add_manifest_entry "$prob" "$method" "$prefix"

    if $SKIP_COMPLETED && is_completed "$prob" "$method" "$prefix"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip] ${prob} ${method} ${hp_tag}" || true
        return
    fi

    local cmd="python rethink_exp/main_results.py \
        --problem ${prob_arg} \
        --opt_model ${method} \
        --solver ${solver} \
        --opt_name ${opt_name} \
        --lr ${lr} \
        --n_epochs 300 \
        --patience 40 \
        --instances ${instances} \
        --testinstances ${testinstances} \
        --seed 2023 \
        --n_ptr_epochs 0 \
        --prefix ${prefix} \
        --method_path ${method_path} \
        ${bs_flag} \
        ${cfg_flag} \
        ${extra_args}"

    local job_name="bp2_${prob}_${method}_${hp_tag}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${job_name}  (lr=${lr}, batch=${batch_label}, time=${walltime})"
        echo "  output: saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
        echo "  cmd: ${cmd}"
        return
    fi

    sbatch --job-name="$job_name" \
           --output="$log_file" \
           --partition="$PARTITION" \
           --cpus-per-task=2 \
           --mem="$MEM" \
           --time="${walltime}" \
           --requeue \
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
# Helper: generate a method YAML for a given HP value on-the-fly
# Instead of creating many YAML files, we generate them at submit time.
# ====================================================================

ensure_yaml() {
    # Create a method YAML based on default.yaml but with one field changed.
    # Returns the path.
    local base_yaml="$1"
    local method="$2"
    local field="$3"
    local value="$4"
    local out_path="openpto/config/models/bench_p2_${method}_${field}_${value}.yaml"

    if [[ ! -f "$SCRIPT_DIR/$out_path" ]]; then
        python3 - <<PYEOF
import ruamel.yaml as yaml
with open("$SCRIPT_DIR/$base_yaml") as f:
    d = yaml.safe_load(f.read())
d["$method"]["$field"] = $value
with open("$SCRIPT_DIR/$out_path", "w") as f:
    yaml.dump(d, f)
PYEOF
    fi
    echo "$out_path"
}

# ====================================================================
# Phase 2 sweeps
# ====================================================================

echo "=== Benchmark Phase 2 sweep ==="
echo "Reading best configs from: $BEST_JSON"
echo "Partition: $PARTITION"
echo ""

BASE_YAML="openpto/config/models/default.yaml"
PERTURB_BASE="openpto/config/models/perturb_s1_n10.yaml"

# ---- dfl: dflalpha ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "dfl" ]]; then
    echo "--- dfl: dflalpha sweep ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${DFL_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "dfl" "dflalpha" "$val")
            submit_p2_job "$prob" "dfl" "alpha${val}" "$SCRIPT_DIR/$yaml_path" "pto"
        done
    done
    echo ""
fi

# ---- blackbox: lambd ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "blackbox" ]]; then
    echo "--- blackbox: lambd sweep ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${BB_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "blackbox" "lambd" "$val")
            submit_p2_job "$prob" "blackbox" "lam${val}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

# ---- qptl: tau ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "qptl" ]]; then
    echo "--- qptl: tau sweep ---"
    for prob in knapsack bipartitematching portfolio; do
        for val in "${QPTL_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "qptl" "tau" "$val")
            submit_p2_job "$prob" "qptl" "tau${val}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

# ---- listLTR: tau ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "listLTR" ]]; then
    echo "--- listLTR: tau sweep ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${LISTLTR_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "listLTR" "tau" "$val")
            submit_p2_job "$prob" "listLTR" "tau${val}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

# ---- lodl: num_samples ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "lodl" ]]; then
    echo "--- lodl: num_samples sweep ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${LODL_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "lodl" "num_samples" "$val")
            submit_p2_job "$prob" "lodl" "ns${val}" "$SCRIPT_DIR/$yaml_path" "lodl"
        done
    done
    echo ""
fi

# ---- perturb: sigma sweep (fixed n_samples=10) ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "perturb" ]]; then
    echo "--- perturb: sigma sweep (n_samples=10) ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${PERTURB_SIGMA_VALS[@]}"; do
            # Find or use existing sigma YAML
            # Encode sigma in tag: replace . with p
            sig_tag="s${val//./p}"
            yaml_path=$(ensure_yaml "$PERTURB_BASE" "perturb" "sigma" "$val")
            submit_p2_job "$prob" "perturb" "${sig_tag}_n10" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""

    # ---- perturb: n_samples sweep (fixed sigma=1.0) ----
    echo "--- perturb: n_samples sweep (sigma=1.0) ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${PERTURB_N_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$PERTURB_BASE" "perturb" "n_samples" "$val")
            submit_p2_job "$prob" "perturb" "s1p0_n${val}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

# ---- pg: sigma ----
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "pg" ]]; then
    echo "--- pg: sigma sweep ---"
    for prob in knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio asurv cook_county; do
        for val in "${PG_SIGMA_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "pg" "sigma" "$val")
            submit_p2_job "$prob" "pg" "s${val//./p}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

# ---- dad: stein_weight ----
DAD_STEIN_VALS=(0.1 0.5 1.0 2.0 5.0)
if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "dad" ]]; then
    echo "--- dad: stein_weight sweep ---"
    for prob in "${PROBLEMS[@]}"; do
        for val in "${DAD_STEIN_VALS[@]}"; do
            yaml_path=$(ensure_yaml "$BASE_YAML" "dad" "stein_weight" "$val")
            submit_p2_job "$prob" "dad" "sw${val//./p}" "$SCRIPT_DIR/$yaml_path" "pno"
        done
    done
    echo ""
fi

echo "Submitted: $n_submitted   Skipped (done): $n_skipped"

# Write manifest JSON (convert jsonl temp file → JSON array)
python3 -c "
import json
entries = [json.loads(l) for l in open('$MANIFEST_TMP') if l.strip()]
with open('$SCRIPT_DIR/$MANIFEST_FILE', 'w') as f:
    json.dump(entries, f, indent=2)
print(f'Manifest written to $MANIFEST_FILE ({len(entries)} entries)')
"
rm -f "$MANIFEST_TMP"
