#!/usr/bin/env bash
# ====================================================================
# Knapsack wave 2 — fully parallel adaptive sigma sweep
# ====================================================================
# 1 SLURM job per (target × LR × model × controller).
#
# Grid:
#   targets    : 5 log-spaced values around sweet spot (0.031–0.141)
#   LRs        : 5e-2, 1e-2, 5e-3
#   models     : dense, poly (poly uses --poly_deg 4 --poly_offset 3)
#   controllers: global (proportional), per-instance
#   epochs     : 300
#
# Total: 5 × 3 × 2 × 2 = 60 jobs, each ~20 min.
#
# Prefixes (results land under perturb_adaptive_sweep/<prefix>/):
#   global      → kn_w2_global_lr{lr}
#   per-instance→ kn_w2_pi_lr{lr}
#
# Usage:
#   bash shells/slurm/submit_kn_w2_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_w2_sweep.sh
#   bash shells/slurm/submit_kn_w2_sweep.sh --lr 1e-2
#   bash shells/slurm/submit_kn_w2_sweep.sh --model dense
#   bash shells/slurm/submit_kn_w2_sweep.sh --controller global
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
MODEL_FILTER=""
CTRL_FILTER=""
PARTITION="batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --model)              MODEL_FILTER="$2"; shift 2 ;;
        --controller)         CTRL_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --time)               TIME="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# Exact float values that produce the desired 3-decimal filenames
# (must match logspace(log10(0.01), log10(2.0), 15) entries)
declare -A TARGET_FLOAT
TARGET_FLOAT[0.031]="0.031123"
TARGET_FLOAT[0.045]="0.045440"
TARGET_FLOAT[0.066]="0.066343"
TARGET_FLOAT[0.097]="0.096863"
TARGET_FLOAT[0.141]="0.141421"
TARGETS=(0.031 0.045 0.066 0.097 0.141)

LRS=("5e-2" "1e-2" "5e-3")
MODELS=("dense" "poly")
CONTROLLERS=("global" "per_instance")

is_completed() {
    local prefix="$1" stem="$2" target_label="$3" model="$4"
    local f="$SWEEP_BASE/${prefix}/${stem}_t${target_label}_s1_${model}.npz"
    [[ -f "$f" ]]
}

submit_job() {
    local job_name="$1" cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
        return
    fi

    local jid
    jid=$(sbatch --parsable \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --cpus-per-task=2 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV
echo 'Job: $job_name  Start:' \$(date)
$cmd
echo 'End:' \$(date)
")
    echo "  $job_name → job $jid"
}

# ====================================================================
echo "======================================================================"
echo "Knapsack wave 2 — fully parallel (1 job per run)"
echo "5 targets × 3 LRs × 2 models × 2 controllers = 60 jobs"
echo "Partition: $PARTITION  Time: $TIME  Epochs: 300"
if [[ -n "$LR_FILTER" ]];   then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$MODEL_FILTER" ]]; then echo "Model filter: $MODEL_FILTER"; fi
if [[ -n "$CTRL_FILTER" ]];  then echo "Controller filter: $CTRL_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for ctrl in "${CONTROLLERS[@]}"; do
    [[ -n "$CTRL_FILTER" && "$ctrl" != "$CTRL_FILTER" ]] && continue

    # stem used in .npz filename: prop_t... vs pi_t...
    if [[ "$ctrl" == "per_instance" ]]; then
        pi_flag="--per_instance"
        stem="pi"
    else
        pi_flag=""
        stem="prop"
    fi

    for lr in "${LRS[@]}"; do
        [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]] && continue

        # prefix directory for this (ctrl, lr) combination
        if [[ "$ctrl" == "per_instance" ]]; then
            prefix="kn_w2_pi_lr${lr}"
        else
            prefix="kn_w2_global_lr${lr}"
        fi

        for model in "${MODELS[@]}"; do
            [[ -n "$MODEL_FILTER" && "$model" != "$MODEL_FILTER" ]] && continue

            # poly model: use correctly-specified poly_deg=4 offset=3
            if [[ "$model" == "poly" ]]; then
                model_args="--poly_deg 4 --poly_offset 3.0"
            else
                model_args=""
            fi

            for target_label in "${TARGETS[@]}"; do
                target_float="${TARGET_FLOAT[$target_label]}"

                if $SKIP_COMPLETED && is_completed "$prefix" "$stem" "$target_label" "$model"; then
                    skipped=$((skipped + 1))
                    continue
                fi

                job_name="kn2_${ctrl:0:2}_${model}_lr${lr//-/}_t${target_label//.}"
                cmd="python -u rethink_exp/perturb_adaptive_sweep.py \
--problem knapsack --solver heuristic \
--config_path openpto/config/probs/knapsack_small.yaml \
--mode proportional \
--n_epochs 300 --n_samples 100 \
--pred_models ${model} ${model_args} \
--lr ${lr} \
--prop_targets ${target_float} \
--prefix ${prefix} \
--gpu -1 ${pi_flag}"

                submit_job "$job_name" "$cmd"
                submitted=$((submitted + 1))
            done
        done
    done
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep kn2"; fi
