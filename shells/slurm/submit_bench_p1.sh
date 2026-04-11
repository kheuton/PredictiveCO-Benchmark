#!/usr/bin/env bash
# ====================================================================
# Benchmark re-run — Phase 1: LR × Batch sweep
# ====================================================================
# Runs all 13 methods × 7 tasks × 3 LRs × 2 batch configs ≈ 498 jobs.
# Each job uses --requeue so preempted jobs are automatically restarted;
# the ExpManager checkpoint/resume support picks up where it left off.
#
# Methods covered:
#   PtO:  mse, dfl, identity
#   PnO:  spo, nce, blackbox, pointLTR, pairLTR, listLTR, lodl, perturb
#         qptl, cpLayer  (knapsack/bipartitematching/portfolio only)
#
# Tasks: knapsack, knapsack-real, energy, budgetalloc, cubic,
#        bipartitematching, portfolio
#
# Usage:
#   bash shells/slurm/submit_bench_p1.sh --dry-run
#   bash shells/slurm/submit_bench_p1.sh
#   bash shells/slurm/submit_bench_p1.sh --problem knapsack
#   bash shells/slurm/submit_bench_p1.sh --method mse
#   bash shells/slurm/submit_bench_p1.sh --method mse --problem knapsack
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
METHOD_FILTER=""
PARTITION="preempt"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true
MANIFEST_FILE="sweep_manifest_p1.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --problem)           PROB_FILTER="$2"; shift 2 ;;
        --method)            METHOD_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Problem configuration
# ====================================================================

PROBLEMS=(knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio)

declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[knapsack-real]=knapsack
PROB_ARG[energy]=energy
PROB_ARG[budgetalloc]=budgetalloc
PROB_ARG[cubic]=cubic
PROB_ARG[bipartitematching]=bipartitematching
PROB_ARG[portfolio]=portfolio

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[knapsack-real]=energy
PROB_VERSION[energy]=energy
PROB_VERSION[budgetalloc]=real
PROB_VERSION[cubic]=gen
PROB_VERSION[bipartitematching]=cora
PROB_VERSION[portfolio]=real

declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[knapsack-real]=openpto/config/probs/knapsack-real.yaml
PROB_CONFIG[energy]=""
PROB_CONFIG[budgetalloc]=""
PROB_CONFIG[cubic]=""
PROB_CONFIG[bipartitematching]=""
PROB_CONFIG[portfolio]=""

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[knapsack-real]=400     # silently ignored; dataset is fixed
INSTANCES[energy]=400            # silently ignored; dataset is fixed
INSTANCES[budgetalloc]=400
INSTANCES[cubic]=250
INSTANCES[bipartitematching]=20
INSTANCES[portfolio]=400

declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[knapsack-real]=200 # silently ignored
TESTINSTANCES[energy]=200        # silently ignored
TESTINSTANCES[budgetalloc]=200
TESTINSTANCES[cubic]=400
TESTINSTANCES[bipartitematching]=6
TESTINSTANCES[portfolio]=200

# ---- Per-problem solvers per method group ----
# PtO methods use fast solvers; PnO methods use the same (but gurobi for energy)
declare -A SOLVER_PTO   # MSE, DFL, Identity
SOLVER_PTO[knapsack]=heuristic
SOLVER_PTO[knapsack-real]=gurobi
SOLVER_PTO[energy]=gurobi
SOLVER_PTO[budgetalloc]=neural
SOLVER_PTO[cubic]=heuristic
SOLVER_PTO[bipartitematching]=cvxpy
SOLVER_PTO[portfolio]=cvxpy

declare -A SOLVER_PNO   # SPO, NCE, LTR, Blackbox, LODL, perturb
SOLVER_PNO[knapsack]=heuristic
SOLVER_PNO[knapsack-real]=gurobi
SOLVER_PNO[energy]=gurobi
SOLVER_PNO[budgetalloc]=neural
SOLVER_PNO[cubic]=heuristic
SOLVER_PNO[bipartitematching]=cvxpy
SOLVER_PNO[portfolio]=cvxpy

declare -A SOLVER_CVXPY  # QPTL, cpLayer — only valid for 3 tasks
SOLVER_CVXPY[knapsack]=heuristic    # qptl uses heuristic solver for knapsack
SOLVER_CVXPY[bipartitematching]=cvxpy
SOLVER_CVXPY[portfolio]=cvxpy

# ====================================================================
# Walltime table (by method group + problem)
# ====================================================================

# Walltime function: returns SLURM walltime for a given (prob, method_group)
# method_group: pto | pno | lodl | energy_pto | energy_pno
get_walltime() {
    local prob="$1" group="$2"
    case "$group" in
        pto)
            case "$prob" in
                energy) echo "3:00:00" ;;   # skip_solver_eval; setup ~35min
                *)       echo "1:30:00" ;;
            esac ;;
        pno)
            case "$prob" in
                energy)        echo "26:00:00" ;;  # Gurobi in training loop
                budgetalloc)   echo "4:00:00" ;;   # neural solver
                *)             echo "2:00:00" ;;
            esac ;;
        lodl)
            case "$prob" in
                energy)        echo "26:00:00" ;;
                budgetalloc)   echo "4:00:00" ;;
                *)             echo "4:00:00" ;;   # slow surrogate training
            esac ;;
    esac
}

# ====================================================================
# Batch configs
# ====================================================================
LRS=(1e-2 5e-3 1e-3)

# Batch configs: label → opt_name + batch_size flag
# Groups: default (full-batch gd) and alt (bs=32 sgd)
# Exception: SPO/NCE/LTR default is bs=1; their alt is bs=32
# Exception: bipartitematching alt is bs=4 (only 20 training instances)

# ====================================================================
# Method definitions
# ====================================================================
# Each method entry: name, solver_group (pto/pno/lodl),
#   default_opt, default_bs, alt_opt, alt_bs,
#   method_path, extra_args, problems (space-sep or "all")

# Format: METHOD_PROBLEMS[method] = "all" or space-separated problem names
declare -A METHOD_PROBLEMS
METHOD_PROBLEMS[mse]="all"
METHOD_PROBLEMS[dfl]="all"
METHOD_PROBLEMS[identity]="all"
METHOD_PROBLEMS[spo]="all"
METHOD_PROBLEMS[nce]="all"
METHOD_PROBLEMS[blackbox]="all"
METHOD_PROBLEMS[pointLTR]="all"
METHOD_PROBLEMS[pairLTR]="all"
METHOD_PROBLEMS[listLTR]="all"
METHOD_PROBLEMS[lodl]="all"
METHOD_PROBLEMS[perturb]="all"
METHOD_PROBLEMS[pg]="knapsack knapsack-real energy cubic bipartitematching portfolio"
METHOD_PROBLEMS[qptl]="knapsack bipartitematching portfolio"
METHOD_PROBLEMS[cpLayer]="knapsack bipartitematching portfolio"

declare -A METHOD_SOLVER_GROUP
METHOD_SOLVER_GROUP[mse]=pto
METHOD_SOLVER_GROUP[dfl]=pto
METHOD_SOLVER_GROUP[identity]=pto
METHOD_SOLVER_GROUP[spo]=pno
METHOD_SOLVER_GROUP[nce]=pno
METHOD_SOLVER_GROUP[blackbox]=pno
METHOD_SOLVER_GROUP[pointLTR]=pno
METHOD_SOLVER_GROUP[pairLTR]=pno
METHOD_SOLVER_GROUP[listLTR]=pno
METHOD_SOLVER_GROUP[lodl]=lodl
METHOD_SOLVER_GROUP[perturb]=pno
METHOD_SOLVER_GROUP[pg]=pno
METHOD_SOLVER_GROUP[qptl]=pno
METHOD_SOLVER_GROUP[cpLayer]=pno

# Default batch config (benchmark default per method group)
declare -A METHOD_DEFAULT_OPT
METHOD_DEFAULT_OPT[mse]=gd
METHOD_DEFAULT_OPT[dfl]=gd
METHOD_DEFAULT_OPT[identity]=gd
METHOD_DEFAULT_OPT[spo]=sgd      # benchmark used bs=1
METHOD_DEFAULT_OPT[nce]=sgd
METHOD_DEFAULT_OPT[blackbox]=gd
METHOD_DEFAULT_OPT[pointLTR]=sgd
METHOD_DEFAULT_OPT[pairLTR]=sgd
METHOD_DEFAULT_OPT[listLTR]=sgd
METHOD_DEFAULT_OPT[lodl]=gd
METHOD_DEFAULT_OPT[perturb]=gd
METHOD_DEFAULT_OPT[pg]=gd
METHOD_DEFAULT_OPT[qptl]=gd
METHOD_DEFAULT_OPT[cpLayer]=gd

declare -A METHOD_DEFAULT_BS     # only used when opt=sgd
METHOD_DEFAULT_BS[spo]=1
METHOD_DEFAULT_BS[nce]=1
METHOD_DEFAULT_BS[pointLTR]=1
METHOD_DEFAULT_BS[pairLTR]=1
METHOD_DEFAULT_BS[listLTR]=1

# Alt batch config (always sgd bs=32; bipartitematching uses bs=4)
METHOD_ALT_OPT=sgd   # all methods

# Method config YAML (--method_path)
declare -A METHOD_PATH
METHOD_PATH[mse]=openpto/config/models/default.yaml
METHOD_PATH[dfl]=openpto/config/models/default.yaml
METHOD_PATH[identity]=openpto/config/models/default.yaml
METHOD_PATH[spo]=openpto/config/models/default.yaml
METHOD_PATH[nce]=openpto/config/models/default.yaml
METHOD_PATH[blackbox]=openpto/config/models/default.yaml
METHOD_PATH[pointLTR]=openpto/config/models/default.yaml
METHOD_PATH[pairLTR]=openpto/config/models/default.yaml
METHOD_PATH[listLTR]=openpto/config/models/default.yaml
METHOD_PATH[lodl]=openpto/config/models/default.yaml
METHOD_PATH[perturb]=openpto/config/models/perturb_s1_n10.yaml   # sigma=1.0, n=10
METHOD_PATH[pg]=openpto/config/models/default.yaml
METHOD_PATH[qptl]=openpto/config/models/default.yaml
METHOD_PATH[cpLayer]=openpto/config/models/default.yaml

# Extra args per method (e.g. --skip_solver_eval for PtO on energy)
# These are evaluated per (method, prob) at submit time

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

# Build set of job names currently RUNNING or PENDING in SLURM.
# Use --Format=name (capital F) to get full untruncated names.
# --format="%j" (lowercase, no width) gives full untruncated job names.
_SLURM_ACTIVE_JOBS=$(squeue --user="$USER" --format="%j" --noheader 2>/dev/null || true)

is_completed() {
    local prob="$1" method="$2" prefix="$3"
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
    [[ -f "$out" ]]
}

is_running() {
    # Returns true if a job with this exact name is already active in SLURM.
    local job_name="$1"
    echo "$_SLURM_ACTIVE_JOBS" | grep -qxF "$job_name"
}

has_checkpoint() {
    local prob="$1" method="$2" prefix="$3"
    local ckpt="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/checkpoints/checkpoint_latest.pt"
    [[ -f "$ckpt" ]]
}

n_submitted=0
n_skipped=0
n_resuming=0

# Manifest: write one JSON-lines entry per call; merge to array at end.
MANIFEST_TMP="$SCRIPT_DIR/.manifest_p1_tmp.jsonl"
: > "$MANIFEST_TMP"   # truncate/create

add_manifest_entry() {
    local prob="$1" method="$2" prefix="$3"
    printf '{"prob":"%s","opt_model":"%s","prefix":"%s"}\n' \
        "$prob" "$method" "$prefix" >> "$MANIFEST_TMP"
}

submit_job() {
    local prob="$1" method="$2" batch_label="$3" lr="$4"

    if [[ -n "$PROB_FILTER"   && "$prob"   != "$PROB_FILTER"   ]]; then return; fi
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then return; fi

    # Check if this problem is in scope for this method
    local prob_list="${METHOD_PROBLEMS[$method]}"
    if [[ "$prob_list" != "all" ]]; then
        local found=false
        for p in $prob_list; do [[ "$p" == "$prob" ]] && found=true; done
        $found || return 0
    fi

    # Determine opt_name and batch_size
    local opt_name bs_flag=""
    if [[ "$batch_label" == "default" ]]; then
        opt_name="${METHOD_DEFAULT_OPT[$method]:-gd}"
        local default_bs="${METHOD_DEFAULT_BS[$method]:-}"
        [[ -n "$default_bs" ]] && bs_flag="--batch_size ${default_bs}"
    else
        # alt: sgd bs=32 (or bs=4 for bipartitematching)
        opt_name=sgd
        local alt_bs=32
        [[ "$prob" == "bipartitematching" ]] && alt_bs=4
        bs_flag="--batch_size ${alt_bs}"
    fi

    # Build prefix
    local lr_tag="${lr//./_}"
    local prefix="bench_p1_${method}_${batch_label}_lr${lr}"

    # Register in manifest (regardless of completion status)
    add_manifest_entry "$prob" "$method" "$prefix"

    local job_name="bp1_${prob}_${method}_${batch_label}_lr${lr}"

    # Skip if already complete
    if $SKIP_COMPLETED && is_completed "$prob" "$method" "$prefix"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip] ${prob} ${method} ${batch_label} lr=${lr}" || true
        return
    fi

    # Skip if already active in SLURM (running or pending) — avoids duplicate submission
    if is_running "$job_name"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip/active] ${job_name}" || true
        return
    fi

    local resuming=false
    has_checkpoint "$prob" "$method" "$prefix" && resuming=true

    # Determine solver
    local solver_group="${METHOD_SOLVER_GROUP[$method]}"
    local solver
    if [[ "$method" == "qptl" || "$method" == "cpLayer" ]]; then
        solver="${SOLVER_CVXPY[$prob]:-cvxpy}"
    elif [[ "$solver_group" == "pto" ]]; then
        solver="${SOLVER_PTO[$prob]}"
    else
        solver="${SOLVER_PNO[$prob]}"
    fi

    # Config path
    local prob_arg="${PROB_ARG[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    # Instance counts
    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"

    # Extra args: PtO methods on energy get --skip_solver_eval
    local extra_args=""
    if [[ "$solver_group" == "pto" && "$prob" == "energy" ]]; then
        extra_args="--skip_solver_eval"
    fi

    # Walltime
    local walltime
    walltime=$(get_walltime "$prob" "$solver_group")

    local method_path="${METHOD_PATH[$method]}"

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

    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    if $DRY_RUN; then
        local resume_tag=""
        $resuming && resume_tag=" [RESUME]"
        echo "[DRY-RUN]${resume_tag} ${job_name}  (time=${walltime})"
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
    $resuming && (( n_resuming++ )) || true
    echo "  [submit] ${job_name}$(${resuming} && echo ' [RESUME]' || echo '')"
}

# ====================================================================
# Main sweep
# ====================================================================

METHODS=(mse dfl identity spo nce blackbox pointLTR pairLTR listLTR lodl perturb pg qptl cpLayer)
BATCH_LABELS=(default alt)

echo "=== Benchmark Phase 1 sweep ==="
echo "Methods:  ${METHODS[*]}"
echo "Problems: ${PROBLEMS[*]}"
echo "LRs:      ${LRS[*]}"
echo "Batches:  ${BATCH_LABELS[*]}"
echo "Partition: $PARTITION"
echo "Filters:  problem='${PROB_FILTER}' method='${METHOD_FILTER}'"
echo ""

for prob in "${PROBLEMS[@]}"; do
    echo "--- ${prob} ---"
    for method in "${METHODS[@]}"; do
        for batch in "${BATCH_LABELS[@]}"; do
            for lr in "${LRS[@]}"; do
                submit_job "$prob" "$method" "$batch" "$lr"
            done
        done
    done
    echo ""
done

echo "Submitted: $n_submitted   Resuming: $n_resuming   Skipped (done): $n_skipped"

# Write manifest JSON (convert jsonl temp file → JSON array)
python3 -c "
import json, sys
entries = [json.loads(l) for l in open('$MANIFEST_TMP') if l.strip()]
with open('$SCRIPT_DIR/$MANIFEST_FILE', 'w') as f:
    json.dump(entries, f, indent=2)
print(f'Manifest written to $MANIFEST_FILE ({len(entries)} entries)')
"
rm -f "$MANIFEST_TMP"
