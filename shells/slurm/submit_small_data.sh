#!/usr/bin/env bash
# ====================================================================
# Small-data sweep (Committee Plan #4)
# ====================================================================
# Tests whether decision-aware methods degrade faster than MSE as train
# size shrinks. 3 problems × 5 methods × 4 sizes = 60 jobs.
#
# Problems:
#   - knapsack   (synthetic, linear CO, dense MLP)
#   - sp_synth   (synthetic, linear CO, linear head — deg-6 mis-spec)
#   - cook_county (real, TopK, 4 train timesteps)
#
# Methods: mse, dfl, perturb, lodl, dad
# HP: use Phase-1 best (LR, batch) per method × problem — read from
#     bench_p1_best.json. Method-specific HPs stay at defaults
#     (perturb σ=1.0 n=10, lodl num_samples=500, dad stein_weight=1.0).
#
# Sizes:
#   knapsack/sp_synth: 50, 100, 200, 320   (320 = full train at val_frac=0.2)
#   cook_county:       1, 2, 3, 4          (years)
#
# Usage:
#   bash shells/slurm/submit_small_data.sh --dry-run
#   bash shells/slurm/submit_small_data.sh
#   bash shells/slurm/submit_small_data.sh --problem knapsack --method mse
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
METHOD_FILTER=""
SIZE_FILTER=""
PARTITION="preempt"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true
BEST_JSON="bench_p1_best.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --problem)           PROB_FILTER="$2"; shift 2 ;;
        --method)            METHOD_FILTER="$2"; shift 2 ;;
        --size)              SIZE_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Problem + method config (reuse bench_p1 naming)
# ====================================================================

PROBLEMS=(knapsack sp_synth cook_county)

declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[sp_synth]=sp_synth
PROB_ARG[cook_county]=cook_county

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[sp_synth]=synth
PROB_VERSION[cook_county]=real

declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[sp_synth]=openpto/config/probs/sp_synth.yaml
PROB_CONFIG[cook_county]=openpto/config/probs/cook_county.yaml

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[sp_synth]=400
INSTANCES[cook_county]=400  # ignored

declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[sp_synth]=10000
TESTINSTANCES[cook_county]=200  # ignored

declare -A PRED_MODEL_ARGS
PRED_MODEL_ARGS[sp_synth]="--pred_model dense --n_layers 1"

# Train-size grid per problem
declare -A SIZES
SIZES[knapsack]="50 100 200 320"
SIZES[sp_synth]="50 100 200 320"
SIZES[cook_county]="1 2 3 4"

# Solver per method-group per problem
declare -A SOLVER
SOLVER[knapsack,pto]=heuristic
SOLVER[knapsack,pno]=heuristic
SOLVER[sp_synth,pto]=heuristic
SOLVER[sp_synth,pno]=heuristic
SOLVER[cook_county,pto]=heuristic
SOLVER[cook_county,pno]=heuristic

METHODS=(mse dfl perturb lodl dad)

declare -A METHOD_GROUP
METHOD_GROUP[mse]=pto
METHOD_GROUP[dfl]=pto
METHOD_GROUP[perturb]=pno
METHOD_GROUP[lodl]=lodl
METHOD_GROUP[dad]=pno

declare -A METHOD_PATH
METHOD_PATH[mse]=openpto/config/models/default.yaml
METHOD_PATH[dfl]=openpto/config/models/default.yaml
METHOD_PATH[perturb]=openpto/config/models/perturb_s1_n10.yaml
METHOD_PATH[lodl]=openpto/config/models/default.yaml
METHOD_PATH[dad]=openpto/config/models/default.yaml

# ====================================================================
# Read Phase-1 best (lr, batch) per (method, problem) from JSON
# ====================================================================
if [[ ! -f "$SCRIPT_DIR/$BEST_JSON" ]]; then
    echo "ERROR: $BEST_JSON not found. Run collect_bench_p1.py first."
    exit 1
fi

declare -A BEST_LR
declare -A BEST_BATCH
while IFS=$'\t' read -r m p lr batch; do
    BEST_LR[$m,$p]="$lr"
    BEST_BATCH[$m,$p]="$batch"
done < <(python3 -c "
import json
d = json.load(open('$SCRIPT_DIR/$BEST_JSON'))
for m in ['mse','dfl','perturb','lodl','dad']:
    for p in ['knapsack','sp_synth','cook_county']:
        cfg = d.get(m, {}).get(p)
        if cfg:
            print(f\"{m}\t{p}\t{cfg['lr']}\t{cfg['batch']}\")
")

# ====================================================================
# Walltime — small sizes finish quickly; use conservative caps
# ====================================================================
get_walltime() {
    local prob="$1" group="$2"
    case "$group" in
        pto)                   echo "1:00:00" ;;
        pno|lodl)
            case "$prob" in
                budgetalloc)   echo "3:00:00" ;;
                *)             echo "1:30:00" ;;
            esac ;;
    esac
}

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

_SLURM_ACTIVE_JOBS=$(squeue --user="$USER" --format="%j" --noheader 2>/dev/null || true)

is_completed() {
    local prob="$1" method="$2" prefix="$3"
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
    [[ -f "$out" ]]
}

is_running() {
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

submit_job() {
    local prob="$1" method="$2" size="$3"

    if [[ -n "$PROB_FILTER"   && "$prob"   != "$PROB_FILTER"   ]]; then return; fi
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then return; fi
    if [[ -n "$SIZE_FILTER"   && "$size"   != "$SIZE_FILTER"   ]]; then return; fi

    local lr="${BEST_LR[$method,$prob]}"
    local batch_label="${BEST_BATCH[$method,$prob]}"
    if [[ -z "$lr" || -z "$batch_label" ]]; then
        echo "  [skip] no Phase-1 best for ${method}/${prob}"
        return
    fi

    # Determine opt_name and batch_size (mirrors submit_bench_p1.sh)
    local opt_name bs_flag=""
    if [[ "$batch_label" == "default" ]]; then
        # For our 5 methods, default=gd for all (no bs flag)
        opt_name="gd"
    else
        opt_name="sgd"
        bs_flag="--batch_size 32"
    fi

    # Prefix encodes train size
    local prefix="small_n${size}_${method}_${batch_label}_lr${lr}"
    local job_name="sd_${prob}_${method}_n${size}"

    if $SKIP_COMPLETED && is_completed "$prob" "$method" "$prefix"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip] ${prob} ${method} n=${size}" || true
        return
    fi

    if is_running "$job_name"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip/active] ${job_name}" || true
        return
    fi

    local resuming=false
    has_checkpoint "$prob" "$method" "$prefix" && resuming=true

    local group="${METHOD_GROUP[$method]}"
    local solver="${SOLVER[$prob,$group]:-heuristic}"
    # lodl shares solver with pno here
    [[ "$group" == "lodl" ]] && solver="${SOLVER[$prob,pno]}"

    local prob_arg="${PROB_ARG[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local pred_model_args="${PRED_MODEL_ARGS[$prob]:-}"
    local walltime
    walltime=$(get_walltime "$prob" "$group")
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
        --train_subsample_n ${size} \
        ${bs_flag} \
        ${cfg_flag} \
        ${pred_model_args}"

    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    if $DRY_RUN; then
        local resume_tag=""
        $resuming && resume_tag=" [RESUME]"
        echo "[DRY-RUN]${resume_tag} ${job_name}  time=${walltime}  lr=${lr} batch=${batch_label}"
        echo "  output: saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
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

echo "=== Small-data sweep ==="
echo "Problems: ${PROBLEMS[*]}"
echo "Methods:  ${METHODS[*]}"
echo "Partition: $PARTITION"
echo "Filters:   problem='${PROB_FILTER}' method='${METHOD_FILTER}' size='${SIZE_FILTER}'"
echo ""

for prob in "${PROBLEMS[@]}"; do
    echo "--- ${prob}  (sizes: ${SIZES[$prob]}) ---"
    for method in "${METHODS[@]}"; do
        for size in ${SIZES[$prob]}; do
            submit_job "$prob" "$method" "$size"
        done
    done
    echo ""
done

echo "Submitted: $n_submitted   Resuming: $n_resuming   Skipped: $n_skipped"
