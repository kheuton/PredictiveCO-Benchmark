#!/usr/bin/env bash
# ====================================================================
# One-off: PG warm-started from SPO+ winner on pg_misspec
# ====================================================================
# The PG paper §4.1 (arxiv 2402.03256) warm-starts PG with SPO+ for this
# experiment — without it, a fresh random-init PG with a small sigma can
# predict all-positive costs everywhere → binary solver outputs z=0 for
# both c_hat and c_hat-sigma*c_true → loss = 0 → no gradient → stuck.
#
# Our cold-start P2 winners confirm this: PG/DAD got val=0.244, test=1.0
# (z=0 everywhere), while SPO+ correctly learned slope=-0.918 with
# zero-crossing near x*=0.51 (true threshold = 0.50).
#
# This script does NOT modify the rerun infrastructure. It uses a fresh
# `oneoff_pgws_*` prefix and writes to `saved_records/pg_misspec-v3/pg/`
# alongside the existing bench_p2_pg_* runs but does not get added to
# sweep_manifest_p2.json.
#
# Sweeps:
#   sigma ∈ {0.01, 0.05, 0.1, 0.5, 1.0}   (same as cold-start P2)
#   lr    ∈ {1e-2, 5e-3}                  (P2 winner LR + perturb's best)
#
# Usage:
#   bash shells/slurm/submit_pg_warmstart_pgmisspec.sh --dry-run
#   bash shells/slurm/submit_pg_warmstart_pgmisspec.sh
# ====================================================================

set -e

DRY_RUN=false
PARTITION="preempt"
MEM="4G"
CONDA_ENV="pco_bench_rhel7"
WALLTIME="00:30:00"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# SPO+ winner checkpoint (best by val on pg_misspec)
SPO_CKPT="$SCRIPT_DIR/saved_records/pg_misspec-v3/spo/bench_p1_spo_default_lr5e-2/checkpoints/tr_pred_best.pt"
if [[ ! -f "$SPO_CKPT" ]]; then
    echo "ERROR: SPO+ winner checkpoint not found at $SPO_CKPT"
    exit 1
fi

# Shared per-pg_misspec args
PROB="pg_misspec"
PROB_CFG="openpto/config/probs/pg_misspec.yaml"
PRED_ARGS="--pred_model dense --n_layers 1"
SOLVER="heuristic"
INSTANCES=400
TESTINSTANCES=10000

SIGMAS=(0.01 0.05 0.1 0.5 1.0)
LRS=(1e-2 5e-3)

n_submitted=0
n_skipped=0

for lr in "${LRS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        sigma_tag="s${sigma//./p}"
        prefix="oneoff_pgws_${sigma_tag}_lr${lr}"
        method_path="$SCRIPT_DIR/openpto/config/models/bench_p2_pg_sigma_${sigma}.yaml"
        log_dir="saved_records/pg_misspec-v3/pg/${prefix}"
        results_npy="$SCRIPT_DIR/${log_dir}/results.npy"
        job_name="pgws_${sigma_tag}_lr${lr}"

        if $SKIP_COMPLETED && [[ -f "$results_npy" ]]; then
            (( n_skipped++ )) || true
            $DRY_RUN && echo "  [skip] ${job_name} (results.npy exists)"
            continue
        fi

        cmd="python rethink_exp/main_results.py \
            --problem ${PROB} \
            --opt_model pg \
            --solver ${SOLVER} \
            --opt_name gd \
            --lr ${lr} \
            --n_epochs 300 \
            --patience 40 \
            --instances ${INSTANCES} \
            --testinstances ${TESTINSTANCES} \
            --seed 2023 \
            --n_ptr_epochs 0 \
            --prefix ${prefix} \
            --method_path ${method_path} \
            --config_path ${PROB_CFG} \
            --trained_path ${SPO_CKPT} \
            ${PRED_ARGS}"

        if $DRY_RUN; then
            echo "[DRY-RUN] ${job_name}"
            echo "  output: ${log_dir}/results.npy"
            echo "  cmd: ${cmd}"
            continue
        fi

        log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
        sbatch --job-name="$job_name" \
               --output="$log_file" \
               --partition="$PARTITION" \
               --cpus-per-task=2 \
               --mem="$MEM" \
               --time="$WALLTIME" \
               --requeue \
               --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
        (( n_submitted++ )) || true
        echo "  [submit] ${job_name}"
    done
done

echo ""
echo "Submitted: $n_submitted    Skipped: $n_skipped"
