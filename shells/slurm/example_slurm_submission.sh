#!/usr/bin/env bash
# Submit classification experiments to SLURM.
#
# Each method gets its own SLURM job that runs all trials serially.
# Or submit a single method with --method.
#
# Usage:
#   ./submit_classification.sh --variant 1                          # all methods
#   ./submit_classification.sh --variant 1 --method SPO+            # single method
#   ./submit_classification.sh --variant 1 --method SPO+ --start 50 --end 99
#   ./submit_classification.sh --variant 1 --all-variants           # run variants 1,2,3
#   ./submit_classification.sh --all-variants --dpo-sigma 0.1 --tag dpo_s0.1 --method DPO
#   ./submit_classification.sh --all-variants --sweep-init           # init at sweep optimal

set -e

# Defaults
VARIANT=""
METHOD=""
START=0
END=99
ALL_VARIANTS=false
PARTITION="batch,hugheslab"
TIME="12:00:00"
MEM="8G"
DPO_SIGMA=""
INIT_THRESHOLD=""
TAG=""
SWEEP_INIT=false
DPO_SIGMAS=""
DPO_N_SAMPLES=""
PG_H=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)    VARIANT="$2"; shift 2 ;;
        --method)     METHOD="$2"; shift 2 ;;
        --start)      START="$2"; shift 2 ;;
        --end)        END="$2"; shift 2 ;;
        --all-variants) ALL_VARIANTS=true; shift ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --dpo-sigma)  DPO_SIGMA="$2"; shift 2 ;;
        --init-threshold) INIT_THRESHOLD="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --sweep-init) SWEEP_INIT=true; shift ;;
        --dpo-sigmas) DPO_SIGMAS="$2"; shift 2 ;;
        --dpo-n-samples) DPO_N_SAMPLES="$2"; shift 2 ;;
        --pg-h)       PG_H="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $ALL_VARIANTS; then
    VARIANTS=(1 2 3)
elif [[ -n "$VARIANT" ]]; then
    VARIANTS=("$VARIANT")
else
    echo "Error: must specify --variant N or --all-variants"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT_SCRIPT="$SCRIPT_DIR/pg_paper_supplement/run_classification.py"
OUTPUT_DIR="$SCRIPT_DIR/results/classification"
SWEEP_JSON="$SCRIPT_DIR/results/optimal_thresholds_noisy.json"

ALL_METHODS=("SPO+" "FYL" "DBB" "MSE" "LTR_list" "LTR_pair" "LTR_point" "PGB" "PGF" "PGC" "DCA" "DPO")

# If sweep-init, load thresholds and require the JSON file
if $SWEEP_INIT; then
    if [[ ! -f "$SWEEP_JSON" ]]; then
        echo "Error: $SWEEP_JSON not found. Run sweep_thresholds.py first."
        exit 1
    fi
    if [[ -z "$TAG" ]]; then
        TAG="sweepinit"
    fi
    # Default DPO sigmas for sweep-init if not set
    if [[ -z "$DPO_SIGMAS" && -z "$DPO_SIGMA" ]]; then
        DPO_SIGMAS="0.05,0.1,0.25"
    fi
fi

# Create log directory
mkdir -p "$SCRIPT_DIR/logs/slurm"

submit_job() {
    local method="$1"
    local variant="$2"
    local sigma_override="$3"  # optional: override DPO sigma
    # Sanitize method name for SLURM job name (SPO+ -> SPOp)
    local safe_method="${method//+/p}"
    local tag_suffix=""
    [[ -n "$TAG" ]] && tag_suffix="_${TAG}"
    # Append sigma to job name for DPO variants
    local sigma_suffix=""
    if [[ -n "$sigma_override" ]]; then
        sigma_suffix="_s${sigma_override}"
    fi
    local job_name="cls_v${variant}_${safe_method}${tag_suffix}${sigma_suffix}"

    # Build extra args
    local extra_args=""
    if [[ -n "$sigma_override" ]]; then
        extra_args="$extra_args --dpo-sigma $sigma_override"
    elif [[ -n "$DPO_SIGMA" ]]; then
        extra_args="$extra_args --dpo-sigma $DPO_SIGMA"
    fi
    if [[ -n "$TAG" ]]; then
        extra_args="$extra_args --tag $TAG"
    fi
    if [[ -n "$DPO_N_SAMPLES" ]]; then
        extra_args="$extra_args --dpo-n-samples $DPO_N_SAMPLES"
    fi
    if [[ -n "$PG_H" ]]; then
        extra_args="$extra_args --pg-h $PG_H"
    fi

    # Determine init-threshold
    local init_thresh_arg=""
    if [[ -n "$INIT_THRESHOLD" ]]; then
        init_thresh_arg="--init-threshold $INIT_THRESHOLD"
    elif $SWEEP_INIT; then
        # Read threshold from JSON: jq '."variant"."method"'
        local thresh
        thresh=$(python -c "import json; d=json.load(open('$SWEEP_JSON')); print(d['$variant']['$method'])")
        init_thresh_arg="--init-threshold $thresh"
        echo "  -> init-threshold=$thresh (from sweep)"
    fi

    echo "Submitting: $job_name (method=$method, variant=$variant, trials=$START-$END)"

    sbatch \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --cpus-per-task=1 \
        --output="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out" \
        --error="$SCRIPT_DIR/logs/slurm/${job_name}_%j.err" \
        --export=ALL \
        --wrap="
cd $SCRIPT_DIR
eval \"\$(conda shell.bash hook)\"
conda activate ieo_rhel7
export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd):\$(pwd)/PyEPO/pkg\"

echo '=============================================='
echo 'Job: $job_name'
echo 'Method: $method, Variant: $variant'
echo 'Trials: $START to $END'
echo 'Extra args: $extra_args $init_thresh_arg'
echo 'Python:' \$(which python)
echo 'Start:' \$(date)
echo '=============================================='

for trial in \$(seq $START $END); do
    echo \"[\$(date +%H:%M:%S)] Starting trial \$trial\"
    python $EXPERIMENT_SCRIPT --method '$method' --trial \$trial --variant $variant --output-dir $OUTPUT_DIR $extra_args $init_thresh_arg
    echo \"[\$(date +%H:%M:%S)] Trial \$trial done\"
done

echo '=============================================='
echo 'End:' \$(date)
echo '=============================================='
"
}

# Submit jobs
for V in "${VARIANTS[@]}"; do
    if [[ -n "$METHOD" ]]; then
        METHODS_TO_RUN=("$METHOD")
    else
        METHODS_TO_RUN=("${ALL_METHODS[@]}")
    fi

    for M in "${METHODS_TO_RUN[@]}"; do
        if [[ "$M" == "DPO" && -n "$DPO_SIGMAS" ]]; then
            # Fan out DPO into one job per sigma
            IFS=',' read -ra SIGMAS <<< "$DPO_SIGMAS"
            for S in "${SIGMAS[@]}"; do
                submit_job "$M" "$V" "$S"
            done
        else
            submit_job "$M" "$V"
        fi
    done
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
