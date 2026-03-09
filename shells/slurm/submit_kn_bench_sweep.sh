#!/usr/bin/env bash
# ====================================================================
# Knapsack benchmark-scale adaptive sigma sweep
# ====================================================================
# Full benchmark scale: 320 train + 80 val + 200 test (Knapsack_7.pkl)
# Dense model only. Both global and per-instance controllers.
# Both DP and Gurobi solvers (different n_samples/epoch budgets).
#
# Grid: 3 targets × 3 LRs × 2 controllers × 2 solvers = 36 jobs
#
# Timing per run:
#   DP     n=100 300ep → ~18 min  (time limit: 1h)
#   Gurobi n=10  100ep → ~156 min (time limit: 3h)
#
# Prefixes (under perturb_adaptive_sweep/):
#   kn_bench_{dp,grb}_{global,pi}_lr{lr}
#
# Targets: 3 log-spaced around the small-scale sweet spot (t=0.045–0.097)
# Exact floats from the 15-point logspace(0.01, 2.0, 15) grid.
#
# Usage:
#   bash shells/slurm/submit_kn_bench_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_bench_sweep.sh
#   bash shells/slurm/submit_kn_bench_sweep.sh --solver dp
#   bash shells/slurm/submit_kn_bench_sweep.sh --controller pi
#   bash shells/slurm/submit_kn_bench_sweep.sh --lr 1e-2
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
SOLVER_FILTER=""
CTRL_FILTER=""
PARTITION="batch"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --solver)             SOLVER_FILTER="$2"; shift 2 ;;
        --controller)         CTRL_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP_BASE="$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# Exact float values from logspace(log10(0.01), log10(2.0), 15)
declare -A TARGET_FLOAT
TARGET_FLOAT[0.045]="0.045440"
TARGET_FLOAT[0.066]="0.066343"
TARGET_FLOAT[0.097]="0.096863"
TARGETS=(0.045 0.066 0.097)

LRS=("5e-2" "1e-2" "5e-3")
CONTROLLERS=("global" "per_instance")

# Shared problem args — loads Knapsack_7.pkl (320 train + 80 val + 200 test)
PROB_ARGS="--problem knapsack \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--mode proportional --gpu -1"

# Solver-specific settings
declare -A SOLVER_ARGS
declare -A SOLVER_TIME
SOLVER_ARGS[dp]="--solver heuristic --n_samples 100 --n_epochs 300"
SOLVER_ARGS[grb]="--solver gurobi --n_samples 10 --n_epochs 100"
SOLVER_TIME[dp]="1:00:00"
SOLVER_TIME[grb]="3:00:00"

is_completed() {
    local prefix="$1" stem="$2" target_label="$3"
    local f="$SWEEP_BASE/${prefix}/${stem}_t${target_label}_s1_dense.npz"
    [[ -f "$f" ]]
}

submit_job() {
    local job_name="$1" cmd="$2" time_limit="$3"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name  (limit: $time_limit)"
        echo "  CMD: $cmd"
        return
    fi

    local jid
    jid=$(sbatch --parsable \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$time_limit" \
        --mem="8G" \
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
echo "Knapsack benchmark-scale sweep (Knapsack_7: 320+80 train/val, 200 test)"
echo "3 targets × 3 LRs × 2 controllers × 2 solvers = 36 jobs"
echo "  DP:     n=100, 300ep, ~18 min/run, 1h limit"
echo "  Gurobi: n=10,  100ep, ~156 min/run, 3h limit"
echo "Partition: $PARTITION"
if [[ -n "$LR_FILTER" ]];     then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$SOLVER_FILTER" ]]; then echo "Solver filter: $SOLVER_FILTER"; fi
if [[ -n "$CTRL_FILTER" ]];   then echo "Controller filter: $CTRL_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0; skipped=0

for solver in dp grb; do
    [[ -n "$SOLVER_FILTER" && "$solver" != "$SOLVER_FILTER" ]] && continue

    for ctrl in "${CONTROLLERS[@]}"; do
        [[ -n "$CTRL_FILTER" && "$ctrl" != "$CTRL_FILTER" ]] && continue

        if [[ "$ctrl" == "per_instance" ]]; then
            pi_flag="--per_instance"
            stem="pi"
        else
            pi_flag=""
            stem="prop"
        fi

        for lr in "${LRS[@]}"; do
            [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]] && continue

            prefix="kn_bench_${solver}_${ctrl}_lr${lr}"

            for target_label in "${TARGETS[@]}"; do
                target_float="${TARGET_FLOAT[$target_label]}"

                if $SKIP_COMPLETED && is_completed "$prefix" "$stem" "$target_label"; then
                    skipped=$((skipped + 1))
                    continue
                fi

                job_name="knb_${solver}_${ctrl:0:2}_lr${lr//-/}_t${target_label//.}"
                cmd="python -u rethink_exp/perturb_adaptive_sweep.py \
$PROB_ARGS ${SOLVER_ARGS[$solver]} \
--pred_models dense \
--lr ${lr} \
--prop_targets ${target_float} \
--prefix ${prefix} ${pi_flag}"

                submit_job "$job_name" "$cmd" "${SOLVER_TIME[$solver]}"
                submitted=$((submitted + 1))
            done
        done
    done
done

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; else echo "Monitor: squeue -u \$USER | grep knb"; fi
