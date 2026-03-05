#!/usr/bin/env bash
# ====================================================================
# Wave 1 — Energy: sigmoid activation + sigma sweep
# ====================================================================
# Grid:
#   activation: sigmoid (always)
#   σ ∈ {0.5, 0.1, 0.05}  (constant)
#   σ = 0.5 → 0.01 annealing over 200 epochs (no warmup)
#   lr ∈ {0.05, 0.01, 0.005}
#   = 3 lr × (3 constant + 1 anneal) = 12 experiments
#
# n_epochs=300, patience=40 (default), n_samples=1
#
# Usage:
#   bash shells/slurm/submit_energy_wave1.sh --dry-run
#   bash shells/slurm/submit_energy_wave1.sh
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
DIR="./openpto/data/"
CFGDIR="openpto/config/models"

mkdir -p "$SCRIPT_DIR/logs/slurm"

is_completed() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/energy-energy/perturb/${prefix}/results.npy" ]]
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
N_EPOCHS=300

LRS=("0.05" "0.01" "0.005")
LR_TAGS=("lr5e2" "lr1e2" "lr5e3")
CONST_SIGMAS=("s05" "s01" "s005")

# Constant-sigma experiments: 3 σ × 3 lr = 9
for i in "${!LRS[@]}"; do
    lr="${LRS[$i]}"
    ltag="${LR_TAGS[$i]}"
    for stag in "${CONST_SIGMAS[@]}"; do
        yaml="${CFGDIR}/perturb_en_w1_sig_${stag}.yaml"
        prefix="sd1_en_w1_sig_${stag}_${ltag}"
        cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --lr ${lr} --data_dir ${DIR} --gpu ${GPU_ID}"
        EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")
    done
done

# Annealing experiments: σ=0.5→0.01, 3 lr = 3
for i in "${!LRS[@]}"; do
    lr="${LRS[$i]}"
    ltag="${LR_TAGS[$i]}"
    yaml="${CFGDIR}/perturb_en_w1_sig_s05_ann.yaml"
    prefix="sd1_en_w1_sig_s05_ann_${ltag}"
    cmd_extra="--solver gurobi --n_epochs ${N_EPOCHS} --lr ${lr} --data_dir ${DIR} --gpu ${GPU_ID}"
    EXPERIMENTS+=("${yaml}|${prefix}|${cmd_extra}")
done

# ---- Main ----
echo "=============================================="
echo "Wave 1 — Energy: sigmoid + sigma sweep"
echo "3 lr × (3 constant + 1 anneal) = ${#EXPERIMENTS[@]} experiments"
echo "=============================================="
echo "Primary: $PARTITION"
if $USE_BACKUP; then echo "Backup: $BACKUP_PARTITION"; fi
echo "n_epochs=${N_EPOCHS} | patience=40 (default)"
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "=============================================="
echo ""

submitted=0; skipped=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r yaml_path prefix extra_args <<< "$entry"
    job_name="pco_energy_${prefix}"

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

    cmd="python rethink_exp/main_results.py --problem=energy --opt_model perturb --method_path ${yaml_path} --prefix ${prefix} ${extra_args}"
    submit_experiment "$job_name" "$cmd"
    submitted=$((submitted + 1))
done

echo ""
echo "=============================================="
echo "Submitted: $submitted"
if $USE_BACKUP; then echo "Total SLURM jobs: $((submitted * 2))"; fi
if [[ $skipped -gt 0 ]]; then echo "Skipped: $skipped"; fi
if $DRY_RUN; then echo "Dry run complete."; fi
