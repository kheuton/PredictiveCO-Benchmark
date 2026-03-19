#!/usr/bin/env bash
# ====================================================================
# Knapsack per-item sigma sweep
# ====================================================================
# Part 1 — Hamming-rate per-item controllers
#   4 variants × 3 targets × 3 LRs = 36 jobs
#   Variants:
#     A: --per_item               sigma_min=1e-4  stem=pitem_t
#     B: --per_item --no_sigma_min sigma_min=None  stem=pitem_t
#     C: --per_inst_item           sigma_min=1e-4  stem=piitem_t
#     D: --per_inst_item --no_sigma_min             stem=piitem_t
#   Hamming targets: 0.05, 0.10, 0.20
#
# Part 2 — Coefficient-relative sigma: sigma_{b,i} = alpha * |coeff_hat_{b,i}|
#   5 alpha values × 3 LRs = 15 jobs
#   Alpha values: 0.1, 0.5, 1.0, 2.0, 5.0
#
# Part 3 — Warm-start + MSE-reg (λ=10): 2 jobs
#
# Total: 53 jobs
# LRs: 5e-2, 1e-2, 5e-3
# Solver: DP (heuristic), n_samples=100, 300 epochs, ~45 min/run
#
# Usage:
#   bash shells/slurm/submit_kn_per_item_sweep.sh --dry-run
#   bash shells/slurm/submit_kn_per_item_sweep.sh
#   bash shells/slurm/submit_kn_per_item_sweep.sh --lr 1e-2
#   bash shells/slurm/submit_kn_per_item_sweep.sh --variant A
#   bash shells/slurm/submit_kn_per_item_sweep.sh --part coeff_rel
# ====================================================================

set -e

DRY_RUN=false
LR_FILTER=""
VARIANT_FILTER=""
PART_FILTER=""   # "hamming", "coeff_rel", "warmstart", or "" (all)
PARTITION="hugheslab,batch"
TIME="1:00:00"
MEM="8G"
CONDA_ENV="pco_bench_rhel7"
SKIP_COMPLETED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true; shift ;;
        --lr)                 LR_FILTER="$2"; shift 2 ;;
        --variant)            VARIANT_FILTER="$2"; shift 2 ;;
        --part)               PART_FILTER="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --no-skip-completed)  SKIP_COMPLETED=false; shift ;;
        --time)               TIME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Completion checks ----
is_completed_hamming() {
    local prefix="$1" stem="$2" target="$3"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}/${stem}${target}_s1_dense.npz" ]]
}

is_completed_cr() {
    local prefix="$1" alpha="$2"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb_adaptive_sweep/${prefix}/cr_a${alpha}_s1_dense.npz" ]]
}

is_completed_main() {
    local prefix="$1"
    [[ -f "$SCRIPT_DIR/saved_records/knapsack-gen/perturb/${prefix}/results.npy" ]]
}

# ---- Submit helper ----
submit_job() {
    local job_name="$1" cmd="$2"
    if $DRY_RUN; then
        echo "[DRY RUN] $job_name"
        echo "  CMD: $cmd"
        echo ""
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
# Grid
# ====================================================================
ALPHAS=("0.1" "0.5" "1.0" "2.0" "5.0")
TARGETS=("0.050" "0.100" "0.200")
LRS=("5e-2" "1e-2" "5e-3")

declare -A VAR_FLAGS
VAR_FLAGS[A]="--per_item"
VAR_FLAGS[B]="--per_item --no_sigma_min"
VAR_FLAGS[C]="--per_inst_item"
VAR_FLAGS[D]="--per_inst_item --no_sigma_min"

declare -A VAR_STEM
VAR_STEM[A]="pitem_t"
VAR_STEM[B]="pitem_t"
VAR_STEM[C]="piitem_t"
VAR_STEM[D]="piitem_t"

BASE_ARGS="--problem knapsack \
--config_path openpto/config/probs/knapsack_small.yaml \
--instances 400 --testinstances 200 \
--mode proportional --gpu -1 \
--solver heuristic --n_samples 100 --n_epochs 300 \
--pred_models dense"

MSE_CKPT="saved_records/knapsack-gen/mse/kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt"
REG_YAML="openpto/config/models/perturb_kn_reg_w10_n100.yaml"

echo "======================================================================"
echo "Knapsack per-item sigma sweep (53 total jobs)"
echo "  Part 1 — Hamming per-item: 4 variants × 3 targets × 3 LRs = 36"
echo "  Part 2 — Coeff-relative:   5 alpha × 3 LRs = 15"
echo "  Part 3 — Warm-start+reg:   2"
echo "Partition: $PARTITION  Time: $TIME"
if [[ -n "$LR_FILTER" ]]; then echo "LR filter: $LR_FILTER"; fi
if [[ -n "$VARIANT_FILTER" ]]; then echo "Variant filter: $VARIANT_FILTER"; fi
if [[ -n "$PART_FILTER" ]]; then echo "Part filter: $PART_FILTER"; fi
if $DRY_RUN; then echo "*** DRY RUN ***"; fi
if $SKIP_COMPLETED; then echo "Skip completed: ON"; fi
echo "======================================================================"
echo ""

submitted=0
skipped=0

# ---- Part 1: Hamming per-item sweep ----
if [[ -z "$PART_FILTER" || "$PART_FILTER" == "hamming" ]]; then
    echo "--- Part 1: Hamming per-item ---"
    for variant in A B C D; do
        if [[ -n "$VARIANT_FILTER" && "$variant" != "$VARIANT_FILTER" ]]; then continue; fi

        flags="${VAR_FLAGS[$variant]}"
        stem="${VAR_STEM[$variant]}"

        for lr in "${LRS[@]}"; do
            if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

            prefix="kn_per_item_${variant}_lr${lr}"

            for target in "${TARGETS[@]}"; do
                job_name="kn_pitem_${variant}_lr${lr//-/}_t${target//./_}"

                if $SKIP_COMPLETED && is_completed_hamming "$prefix" "$stem" "$target"; then
                    echo "[SKIP] $job_name — already completed"
                    skipped=$((skipped + 1))
                    continue
                fi

                cmd="python -u rethink_exp/perturb_adaptive_sweep.py $BASE_ARGS \
$flags --lr ${lr} --hamming_targets ${target} --prefix ${prefix}"
                submit_job "$job_name" "$cmd"
                submitted=$((submitted + 1))
            done
        done
    done
fi

# ---- Part 2: Coeff-relative sweep ----
if [[ -z "$PART_FILTER" || "$PART_FILTER" == "coeff_rel" ]]; then
    echo ""
    echo "--- Part 2: Coefficient-relative sigma ---"
    for lr in "${LRS[@]}"; do
        if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

        prefix="kn_coeff_rel_lr${lr}"

        for alpha in "${ALPHAS[@]}"; do
            job_name="kn_cr_lr${lr//-/}_a${alpha//./_}"

            if $SKIP_COMPLETED && is_completed_cr "$prefix" "$alpha"; then
                echo "[SKIP] $job_name — already completed"
                skipped=$((skipped + 1))
                continue
            fi

            cmd="python -u rethink_exp/perturb_adaptive_sweep.py $BASE_ARGS \
--coeff_relative --lr ${lr} --cr_alphas ${alpha} --prefix ${prefix}"
            submit_job "$job_name" "$cmd"
            submitted=$((submitted + 1))
        done
    done
fi

# ---- Part 3: Warm-start + MSE reg (λ=10) ----
if [[ -z "$PART_FILTER" || "$PART_FILTER" == "warmstart" ]]; then
    echo ""
    echo "--- Part 3: Warm-start + MSE reg (λ=10) ---"
    for lr in "5e-3" "1e-2"; do
        if [[ -n "$LR_FILTER" && "$lr" != "$LR_FILTER" ]]; then continue; fi

        prefix="kn_plain_mse_reg_w10_warmstart_lr${lr}"
        job_name="kn_ws_reg10_lr${lr//-/}"

        if $SKIP_COMPLETED && is_completed_main "$prefix"; then
            echo "[SKIP] $job_name — already completed"
            skipped=$((skipped + 1))
            continue
        fi

        cmd="python -u rethink_exp/main_results.py \
--problem knapsack --opt_model perturb \
--method_path ${REG_YAML} \
--config_path openpto/config/probs/knapsack_small.yaml \
--solver heuristic --n_epochs 300 --lr ${lr} \
--instances 400 --testinstances 200 \
--n_ptr_epochs 0 --opt_name gd --batch_size 400 \
--pred_model dense \
--prefix ${prefix} \
--warmstart_ckpt ${MSE_CKPT} \
--gpu -1"
        submit_job "$job_name" "$cmd"
        submitted=$((submitted + 1))
    done
fi

echo ""
echo "======================================================================"
echo "Submitted: $submitted jobs"
if [[ $skipped -gt 0 ]]; then echo "Skipped (completed): $skipped"; fi
if $DRY_RUN; then
    echo "Dry run complete."
else
    echo "Monitor: squeue -u \$USER | grep 'kn_pitem\|kn_cr\|kn_ws'"
fi
