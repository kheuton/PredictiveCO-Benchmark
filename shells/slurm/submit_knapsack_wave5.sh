#!/usr/bin/env bash
# ====================================================================
# Wave 5 — Knapsack: sigmoid/tanh activation + annealing combos
# ====================================================================
# Grid:
#   activations: sigmoid, tanh
#   σ ∈ {1.0, 0.5, 0.1, 0.05}
#   lr=5e-3 → always anneal (σ→0.01, warmup=90, 100ep decay)
#   lr=1e-2 → with anneal + without anneal
#   = 2 act × 4 σ × 3 combos = 24 experiments
#
# n_epochs=500, patience=150, n_samples=5
#
# Usage:
#   bash shells/slurm/submit_knapsack_wave5.sh --dry-run
#   bash shells/slurm/submit_knapsack_wave5.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="preempt"
BACKUP_PARTITION="hugheslab,gpu"
USE_BACKUP=true
TIME="24:00:00"
MEM="16G"
GRES="gpu:1"
GPU_ID="0"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                DRY_RUN=true; shift ;;
        --partition)              PARTITION="$2"; shift 2 ;;
        --backup-partition)       BACKUP_PARTITION="$2"; shift 2 ;;
        --no-backup)              USE_BACKUP=false; shift ;;
        --time)                   TIME="$2"; shift 2 ;;
        --mem)                    MEM="$2"; shift 2 ;;
        --gres)                   GRES="$2"; shift 2 ;;
        --gpu)                    GPU_ID="$2"; shift 2 ;;
        --conda-env)              CONDA_ENV="$2"; shift 2 ;;
        --no-skip-completed)      SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="./openpto/data/knapsack"
CFGDIR="openpto/config/models"

mkdir -p "$SCRIPT_DIR/logs/slurm"

is_completed() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb/${prefix}/results.npy" ]]
}

submit_experiment() {
    local job_name="$1"
    local cmd="$2"

    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
        echo ""
        return
    fi

    local primary_jid
    primary_jid=$(sbatch --parsable \
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
echo 'Job: $job_name  Partition: $PARTITION'
echo 'CMD: $cmd'
echo 'Start:' \$(date)
echo '=============================================='
$cmd
echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
")
    echo "  $job_name → $PARTITION (job $primary_jid)"

    if $USE_BACKUP && [[ -n "$BACKUP_PARTITION" ]]; then
        local backup_jid
        backup_jid=$(sbatch --parsable \
            --job-name="${job_name}" \
            --partition="$BACKUP_PARTITION" \
            --time="$TIME" \
            --mem="$MEM" \
            --gres="$GRES" \
            --cpus-per-task=1 \
            --output="$SCRIPT_DIR/logs/slurm/${job_name}_bk_%j.out" \
            --error="$SCRIPT_DIR/logs/slurm/${job_name}_bk_%j.err" \
            --dependency="afternotok:${primary_jid}" \
            --kill-on-invalid-dep=yes \
            --export=ALL \
            --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate $CONDA_ENV
echo '=============================================='
echo 'Job: $job_name  Partition: $BACKUP_PARTITION (backup)'
echo 'CMD: $cmd'
echo 'Start:' \$(date)
echo '=============================================='
$cmd
echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
")
        echo "    ↳ backup → $BACKUP_PARTITION (job $backup_jid, afternotok:$primary_jid)"
    fi
}

# ====================================================================
EXPERIMENTS=()
N_EPOCHS=500
PATIENCE=150

ACTS=("sig" "tanh")
SIGMAS=("s10" "s05" "s01" "s005")

for act in "${ACTS[@]}"; do
    for stag in "${SIGMAS[@]}"; do
        # lr=5e-3, anneal
        yaml="${CFGDIR}/perturb_kn_w5_${act}_${stag}_ann.yaml"
        prefix="sd1_kn_w5_${act}_${stag}_ann_lr5e3"
        cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr 0.005 --data_dir ${DIR} --gpu ${GPU_ID}"
        EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")

        # lr=1e-2, no anneal (constant)
        yaml="${CFGDIR}/perturb_kn_w5_${act}_${stag}_const.yaml"
        prefix="sd1_kn_w5_${act}_${stag}_const_lr1e2"
        cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr 0.01 --data_dir ${DIR} --gpu ${GPU_ID}"
        EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")

        # lr=1e-2, anneal
        yaml="${CFGDIR}/perturb_kn_w5_${act}_${stag}_ann.yaml"
        prefix="sd1_kn_w5_${act}_${stag}_ann_lr1e2"
        cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --patience ${PATIENCE} --lr 0.01 --data_dir ${DIR} --gpu ${GPU_ID}"
        EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")
    done
done

# ---- Main ----
echo "=============================================="
echo "Wave 5 — Knapsack: sigmoid/tanh + annealing"
echo "2 act × 4 σ × 3 combos = ${#EXPERIMENTS[@]} experiments"
echo "=============================================="
echo "Primary: $PARTITION"
if $USE_BACKUP; then echo "Backup: $BACKUP_PARTITION"; fi
echo "n_epochs=${N_EPOCHS} | patience=${PATIENCE}"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "=============================================="
echo ""

submitted=0; skipped=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r yaml_path prefix extra_args <<< "$entry"
    job_name="pco_knapsack_${prefix}"

    if $SKIP_COMPLETED && is_completed "$prefix"; then
        echo "[SKIP] $job_name — already completed"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -f "$SCRIPT_DIR/$yaml_path" ]]; then
        echo "[WARN] Missing config: $yaml_path — skipping"
        skipped=$((skipped + 1))
        continue
    fi

    cmd="python rethink_exp/main_results.py --problem=knapsack --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"
    submit_experiment "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "=============================================="
echo "Submitted: $submitted"
if $USE_BACKUP; then echo "Total SLURM jobs: $((submitted * 2))"; fi
if [[ $skipped -gt 0 ]]; then echo "Skipped: $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; fi
