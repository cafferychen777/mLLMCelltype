#!/bin/bash
# =============================================================================
# scANVI Benchmark: Setup and Run on Arseven
# =============================================================================
# Usage: bash setup_and_run.sh [--dry-run]
#
# This script:
#   1. Creates project directory on /scratch
#   2. Uploads data from local machine (run from local)
#   3. Submits SLURM jobs for each dataset
# =============================================================================

set -euo pipefail

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SCRATCH="${MLLMCELLTYPE_SCANVI_ROOT:-${SCRATCH_ROOT}/scanvi_benchmark}"
VENV="${SCRATCH}/.venv"
SCRIPTS="${SCRATCH}/scripts"
DATA="${SCRATCH}/data"
RESULTS="${SCRATCH}/results"
LOGS="${SCRATCH}/logs"

# ---- Dataset configurations ----
# Format: DATASET_NAME|REFERENCE_PATH|QUERY_PATH|REF_LABEL_KEY|REF_BATCH_KEY|QUERY_BATCH_KEY
#
# Selected based on:
#   - Overlap with existing popV comparison (7 datasets from rebuttal Table)
#   - Diverse tissues
#   - Available Tabula Sapiens reference
#   - Range of difficulty levels (59%–92% mLLMCelltype accuracy)

declare -a DATASETS=(
    # Tier 1: Core popV comparison datasets
    "Thymus|${DATA}/ref/TS_Thymus_filtered.h5ad|${DATA}/query/Thymus.h5ad|cell_type|donor|donor"
    "LCA|${DATA}/ref/TS_Lung_filtered.h5ad|${DATA}/query/LCA_with_umap.h5ad|cell_type|donor|donor"
    "Pancreas|${DATA}/ref/TS_Pancreas_filtered.h5ad|${DATA}/query/TS_Pancreas_filtered.h5ad|cell_type|donor|donor"
    "Heart|${DATA}/ref/TS_Heart_filtered.h5ad|${DATA}/query/TS_Heart_filtered.h5ad|cell_type|donor|donor"
    "Liver|${DATA}/ref/TS_Liver_filtered.h5ad|${DATA}/query/TS_Liver_filtered.h5ad|cell_type|donor|donor"
    # Tier 2: Additional TS tissues for broader coverage
    "Blood|${DATA}/ref/TS_Blood_filtered.h5ad|${DATA}/query/TS_Blood_filtered.h5ad|cell_type|donor|donor"
    "Lung|${DATA}/ref/TS_Lung_filtered.h5ad|${DATA}/query/TS_Lung_filtered.h5ad|cell_type|donor|donor"
)

submit_job() {
    local dataset="$1"
    local ref="$2"
    local query="$3"
    local label_key="$4"
    local ref_batch="$5"
    local query_batch="$6"

    local job_script="${LOGS}/${dataset}_job.sh"

    cat > "${job_script}" << SLURM
#!/bin/bash
#SBATCH --job-name=scanvi_${dataset}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGS}/${dataset}_%j.out
#SBATCH --error=${LOGS}/${dataset}_%j.err

source ${VENV}/bin/activate

python ${SCRIPTS}/run_scanvi_benchmark.py \\
    --dataset ${dataset} \\
    --reference ${ref} \\
    --query ${query} \\
    --output_dir ${RESULTS} \\
    --ref_label_key ${label_key} \\
    --ref_batch_key ${ref_batch} \\
    --query_batch_key ${query_batch}
SLURM

    if [[ "${DRY_RUN:-}" == "true" ]]; then
        echo "[DRY RUN] Would submit: ${job_script}"
    else
        sbatch "${job_script}"
    fi
}

# ---- Parse args ----
DRY_RUN="false"
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="true"

# ---- Create directories ----
echo "Creating directories..."
mkdir -p "${DATA}/ref" "${DATA}/query" "${RESULTS}" "${LOGS}"

# ---- Submit jobs ----
echo "Submitting jobs..."
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ref query label_key ref_batch query_batch <<< "${entry}"

    if [[ ! -f "${ref}" ]]; then
        echo "WARNING: Reference not found: ${ref} — skipping ${dataset}"
        continue
    fi
    if [[ ! -f "${query}" ]]; then
        echo "WARNING: Query not found: ${query} — skipping ${dataset}"
        continue
    fi

    echo "  -> ${dataset}"
    submit_job "${dataset}" "${ref}" "${query}" "${label_key}" "${ref_batch}" "${query_batch}"
done

echo "Done. Check job status with: squeue -u \$USER"
