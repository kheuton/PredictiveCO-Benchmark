#!/usr/bin/env bash
# ====================================================================
# Knapsack minibatch noise sweep
# ====================================================================
# Tests whether SGD minibatch noise helps MSE, plain perturb, MSE-reg
# perturb, and surgery — all on knapsack-gen at benchmark scale.
#
# Groups (144 total jobs):
#   mse    — MSE baseline          (9  jobs: 3bs × 3lr)
#   plain  — Plain perturb σ∈{0.1,0.5,1.0}  (27 jobs: 3σ × 3bs × 3lr)
#   reg    — MSE-reg perturb σ=0.5 λ∈{1,5,10} (27 jobs: 3λ × 3bs × 3lr)
#   surg   — Surgery σ∈{0.1,0.5,1.0} w∈{0.5,1.0,2.0} (81 jobs: 3σ×3w×3bs×3lr)
#
# All runs use --opt_name sgd (true minibatch) and --solver heuristic (DP n=100).
#
# Full-batch baselines for comparison (from prior experiments):
#   MSE (Gurobi)         : 0.064
#   Plain σ=0.5 lr=5e-3  : 0.074  (full-batch gd)
#   Reg  λ=10  lr=5e-3   : 0.061  (full-batch gd)
#   Surg σ=0.5 w=1 lr=1e-2: 0.081 (full-batch gd, constant σ)
#
# Usage:
#   bash shells/slurm/submit_kn_minibatch_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_minibatch_sweep.sh
#   bash shells/slurm/submit_kn_minibatch_sweep.sh --exp mse
#   bash shells/slurm/submit_kn_minibatch_sweep.sh --exp plain
#   bash shells/slurm/submit_kn_minibatch_sweep.sh --exp reg
#   bash shells/slurm/submit_kn_minibatch_sweep.sh --exp surg
# ====================================================================

set -e

DRY_RUN=false
EXP_FILTER=""
PARTITION="preempt"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
WALLTIME="2:00:00"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --exp)                EXP_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Sweep parameters
# ====================================================================

BATCH_SIZES=(32 64 128)
LRS=(5e-3 1e-2 5e-2)

# Plain / surgery sigmas
PLAIN_SIGMAS=(01 05 1)
declare -A SIGMA_YAML
SIGMA_YAML[01]=openpto/config/models/perturb_kn_s01_n100.yaml
SIGMA_YAML[05]=openpto/config/models/perturb_kn_s05_n100.yaml
SIGMA_YAML[1]=openpto/config/models/perturb_kn_s1_n100.yaml

# Reg lambdas (sigma=0.5 fixed)
REG_LAMBDAS=(1 5 10)
declare -A REG_YAML
REG_YAML[1]=openpto/config/models/perturb_kn_reg_w1_n100.yaml
REG_YAML[5]=openpto/config/models/perturb_kn_reg_w5_n100.yaml
REG_YAML[10]=openpto/config/models/perturb_kn_reg_w10_n100.yaml

# Surgery weights
SURG_WEIGHTS=(0.5 1.0 2.0)

# ====================================================================
# Helpers
# ====================================================================

is_completed_mse() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/mse/${prefix}/results.npy" ]]
}

is_completed_perturb() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb/${prefix}/results.npy" ]]
}

n_submitted=0
n_skipped=0

# lr label: replace 'e' with 'e' but dots are fine in prefix; replace . with p for safety
lr_label() {
    echo "$1" | sed 's/\./p/g'
}

# ====================================================================
# Submit helpers
# ====================================================================

submit_mse() {
    local bs="$1"
    local lr="$2"
    local lrl=$(lr_label "$lr")
    local prefix="kn_mb_mse_bs${bs}_lr${lrl}"

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "mse" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_mse "$prefix"; then
        echo "  [skip] mse ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local job_name="mb_mse_bs${bs}_lr${lrl}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/main_results.py \
        --problem knapsack \
        --opt_model mse \
        --solver gurobi \
        --opt_name sgd \
        --lr ${lr} \
        --batch_size ${bs} \
        --n_epochs 300 \
        --n_ptr_epochs 0 \
        --instances 400 \
        --testinstances 200 \
        --seed 2023 \
        --loadnew False \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${WALLTIME})"
        echo "  output: saved_records/knapsack-gen/mse/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${WALLTIME}" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

submit_plain() {
    local sigma="$1"
    local bs="$2"
    local lr="$3"
    local lrl=$(lr_label "$lr")
    local prefix="kn_mb_plain_s${sigma}_bs${bs}_lr${lrl}"
    local method="${SIGMA_YAML[$sigma]}"

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "plain" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_perturb "$prefix"; then
        echo "  [skip] plain ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local job_name="mb_plain_s${sigma}_bs${bs}_lr${lrl}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/main_results.py \
        --problem knapsack \
        --opt_model perturb \
        --solver heuristic \
        --opt_name sgd \
        --lr ${lr} \
        --batch_size ${bs} \
        --n_epochs 300 \
        --n_ptr_epochs 0 \
        --instances 400 \
        --testinstances 200 \
        --seed 2023 \
        --loadnew False \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --method_path ${method} \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${WALLTIME})"
        echo "  output: saved_records/knapsack-gen/perturb/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${WALLTIME}" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

submit_reg() {
    local lam="$1"
    local bs="$2"
    local lr="$3"
    local lrl=$(lr_label "$lr")
    local prefix="kn_mb_reg_s05_lam${lam}_bs${bs}_lr${lrl}"
    local method="${REG_YAML[$lam]}"

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "reg" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_perturb "$prefix"; then
        echo "  [skip] reg ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local job_name="mb_reg_lam${lam}_bs${bs}_lr${lrl}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/main_results.py \
        --problem knapsack \
        --opt_model perturb \
        --solver heuristic \
        --opt_name sgd \
        --lr ${lr} \
        --batch_size ${bs} \
        --n_epochs 300 \
        --n_ptr_epochs 0 \
        --instances 400 \
        --testinstances 200 \
        --seed 2023 \
        --loadnew False \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --method_path ${method} \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${WALLTIME})"
        echo "  output: saved_records/knapsack-gen/perturb/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${WALLTIME}" \
        --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

submit_surg() {
    local sigma="$1"
    local w="$2"
    local bs="$3"
    local lr="$4"
    local lrl=$(lr_label "$lr")
    # weight label: 0.5 -> 05, 1.0 -> 1, 2.0 -> 2
    local wl=$(echo "$w" | sed 's/\.0$//' | sed 's/\./p/g')
    local prefix="kn_mb_surg_s${sigma}_w${wl}_bs${bs}_lr${lrl}"
    local method="${SIGMA_YAML[$sigma]}"

    if [[ -n "$EXP_FILTER" && "$EXP_FILTER" != "surg" ]]; then return; fi
    if $SKIP_COMPLETED && is_completed_perturb "$prefix"; then
        echo "  [skip] surg ${prefix} — already completed"
        (( n_skipped++ )) || true; return
    fi

    local job_name="mb_surg_s${sigma}_w${wl}_bs${bs}_lr${lrl}"
    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"
    local cmd="python rethink_exp/main_results.py \
        --problem knapsack \
        --opt_model perturb \
        --solver heuristic \
        --opt_name sgd \
        --lr ${lr} \
        --batch_size ${bs} \
        --n_epochs 300 \
        --n_ptr_epochs 0 \
        --instances 400 \
        --testinstances 200 \
        --seed 2023 \
        --loadnew False \
        --config_path openpto/config/probs/knapsack_small.yaml \
        --method_path ${method} \
        --grad_surgery \
        --surgery_weight ${w} \
        --prefix ${prefix}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name}  (time=${WALLTIME})"
        echo "  output: saved_records/knapsack-gen/perturb/${prefix}/results.npy"
        echo "  cmd: ${cmd}"; return
    fi

    sbatch --job-name="$job_name" --output="$log_file" \
        --partition="$PARTITION" --cpus-per-task=2 --mem="$MEM" --time="${WALLTIME}" \
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
# Main sweep: 144 total jobs
# ====================================================================

echo "=== Knapsack minibatch noise sweep ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "LRs:         ${LRS[*]}"
echo "Partition:   $PARTITION"
echo "Walltime:    $WALLTIME"
echo ""

# --- MSE (9 jobs) ---
if [[ -z "$EXP_FILTER" || "$EXP_FILTER" == "mse" ]]; then
    echo "--- MSE group (9 jobs: 3bs × 3lr) ---"
    for bs in "${BATCH_SIZES[@]}"; do
        for lr in "${LRS[@]}"; do
            submit_mse "$bs" "$lr"
        done
    done
    echo ""
fi

# --- Plain perturb (27 jobs: 3σ × 3bs × 3lr) ---
if [[ -z "$EXP_FILTER" || "$EXP_FILTER" == "plain" ]]; then
    echo "--- Plain perturb group (27 jobs: 3σ × 3bs × 3lr) ---"
    for sigma in "${PLAIN_SIGMAS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for lr in "${LRS[@]}"; do
                submit_plain "$sigma" "$bs" "$lr"
            done
        done
    done
    echo ""
fi

# --- Reg perturb (27 jobs: 3λ × 3bs × 3lr) ---
if [[ -z "$EXP_FILTER" || "$EXP_FILTER" == "reg" ]]; then
    echo "--- MSE-reg perturb group (27 jobs: 3λ × 3bs × 3lr) ---"
    for lam in "${REG_LAMBDAS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for lr in "${LRS[@]}"; do
                submit_reg "$lam" "$bs" "$lr"
            done
        done
    done
    echo ""
fi

# --- Surgery (81 jobs: 3σ × 3w × 3bs × 3lr) ---
if [[ -z "$EXP_FILTER" || "$EXP_FILTER" == "surg" ]]; then
    echo "--- Surgery group (81 jobs: 3σ × 3w × 3bs × 3lr) ---"
    for sigma in "${PLAIN_SIGMAS[@]}"; do
        for w in "${SURG_WEIGHTS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                for lr in "${LRS[@]}"; do
                    submit_surg "$sigma" "$w" "$bs" "$lr"
                done
            done
        done
    done
    echo ""
fi

echo "Submitted: $n_submitted   Skipped (already done): $n_skipped"
