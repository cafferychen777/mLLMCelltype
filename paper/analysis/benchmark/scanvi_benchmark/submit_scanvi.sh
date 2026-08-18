#!/bin/bash
# Submit scANVI benchmark jobs on arseven
# Usage: bash submit_scanvi.sh [--dry-run]

set -euo pipefail

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SCRATCH="${MLLMCELLTYPE_SCANVI_ROOT:-${SCRATCH_ROOT}/scanvi_benchmark}"
VENV="$SCRATCH/.venv"
SCRIPT="$SCRATCH/scripts/run_all_scanvi.py"
DATA_DIR="$SCRATCH/data"
OUTPUT_DIR="$SCRATCH/results"
LOG_DIR="$SCRATCH/logs"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Datasets that use cross-dataset mode (need GPU, more time)
CROSS_DATASETS=("thymus" "lung")

# Datasets that use leave-one-donor-out (faster per fold but multiple runs)
LODO_DATASETS=("blood" "heart" "liver" "pancreas")

submit_job() {
    local dataset=$1
    local time=$2
    local mem=$3

    local job_script=$(cat <<SLURM
#!/bin/bash
#SBATCH --job-name=scanvi_${dataset}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${LOG_DIR}/scanvi_${dataset}_%j.out
#SBATCH --error=${LOG_DIR}/scanvi_${dataset}_%j.err

source ${VENV}/bin/activate
python ${SCRIPT} --dataset ${dataset} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
SLURM
)

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: scanvi_${dataset} (time=${time}, mem=${mem})"
    else
        echo "$job_script" | sbatch
        echo "Submitted: scanvi_${dataset}"
    fi
}

# Cross-dataset jobs (larger, need more time)
for ds in "${CROSS_DATASETS[@]}"; do
    submit_job "$ds" "04:00:00" "64G"
done

# Leave-one-donor-out jobs (multiple folds but smaller datasets)
for ds in "${LODO_DATASETS[@]}"; do
    submit_job "$ds" "04:00:00" "32G"
done

echo "Done. Check status: squeue -u \$USER"
