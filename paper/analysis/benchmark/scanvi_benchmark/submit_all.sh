#!/usr/bin/env bash
# =============================================================================
# submit_all.sh
# =============================================================================
# Submit scANVI benchmark jobs to the arseven SLURM cluster.
#
# Usage:
#   bash submit_all.sh              # submit all datasets
#   bash submit_all.sh --dry-run    # print commands without submitting
#
# Prerequisites:
#   - scvi-tools installed in the Python environment
#   - Reference and query h5ad files in the expected locations
#   - GPU partition available (gpu, gres=gpu:a30:1)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PAPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ROOT="${MLLMCELLTYPE_PAPER_ROOT:-${DEFAULT_PAPER_ROOT}}"
SCRIPT_DIR="${PROJECT_ROOT}/analysis/benchmark/scanvi_benchmark"
PYTHON="${MLLMCELLTYPE_PYTHON:-python3}"
DATA_DIR="${PROJECT_ROOT}/data/raw"
OUTPUT_BASE="${PROJECT_ROOT}/results/benchmark/scanvi_benchmark"
LOG_DIR="${OUTPUT_BASE}/logs"

# SLURM defaults
PARTITION="gpu"
GPUS="gpu:a30:1"
MEM="64G"
TIME="04:00:00"
CPUS=8

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Commands will be printed but not submitted."
fi

# Create output and log directories
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each entry: DATASET_NAME|REFERENCE_PATH|QUERY_PATH|REF_LABEL_KEY|REF_BATCH_KEY
#
# The reference files are Tabula Sapiens tissue-specific atlases.
# The query files are the benchmark datasets used in the mLLMCelltype paper.
# Adjust paths below to match the actual file locations on arseven.
# ---------------------------------------------------------------------------

declare -a JOBS=(
    "Thymus|${DATA_DIR}/TS_Thymus_filtered.h5ad|${DATA_DIR}/Thymus.h5ad|cell_type|donor"
    "Lung_LCA|${DATA_DIR}/TS_Lung_filtered.h5ad|${DATA_DIR}/LCA.h5ad|cell_type|donor"
    "PBMC|${DATA_DIR}/TS_Blood_filtered.h5ad|${DATA_DIR}/pbmc_lifespan.h5ad|cell_type|donor"
    "Pancreas|${DATA_DIR}/TS_Pancreas_filtered.h5ad|${DATA_DIR}/pancreas.h5ad|cell_type|donor"
    "Kidney|${DATA_DIR}/TS_Kidney_filtered.h5ad|${DATA_DIR}/kidney.h5ad|cell_type|donor"
    "Liver|${DATA_DIR}/TS_Liver_filtered.h5ad|${DATA_DIR}/liver.h5ad|cell_type|donor"
    "Heart|${DATA_DIR}/TS_Heart_filtered.h5ad|${DATA_DIR}/heart.h5ad|cell_type|donor"
)

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
echo "============================================================"
echo "scANVI Benchmark Job Submission"
echo "============================================================"
echo "Project root:  ${PROJECT_ROOT}"
echo "Output dir:    ${OUTPUT_BASE}"
echo "Python:        ${PYTHON}"
echo "Partition:     ${PARTITION}"
echo "GPU:           ${GPUS}"
echo "Memory:        ${MEM}"
echo "Time limit:    ${TIME}"
echo "CPUs:          ${CPUS}"
echo "============================================================"
echo ""

SUBMITTED=0
SKIPPED=0

for entry in "${JOBS[@]}"; do
    IFS='|' read -r DATASET REF_PATH QUERY_PATH LABEL_KEY BATCH_KEY <<< "${entry}"

    JOB_NAME="scanvi_${DATASET}"
    LOG_OUT="${LOG_DIR}/${DATASET}_%j.out"
    LOG_ERR="${LOG_DIR}/${DATASET}_%j.err"
    OUT_DIR="${OUTPUT_BASE}/${DATASET}"

    # Check if reference and query files exist
    if [[ ! -f "${REF_PATH}" ]]; then
        echo "[SKIP] ${DATASET}: reference not found at ${REF_PATH}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [[ ! -f "${QUERY_PATH}" ]]; then
        echo "[SKIP] ${DATASET}: query not found at ${QUERY_PATH}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Build the SLURM submission command
    SBATCH_CMD="sbatch \
        --job-name=${JOB_NAME} \
        --partition=${PARTITION} \
        --gres=${GPUS} \
        --mem=${MEM} \
        --time=${TIME} \
        --cpus-per-task=${CPUS} \
        --output=${LOG_OUT} \
        --error=${LOG_ERR} \
        --wrap=\"${PYTHON} ${SCRIPT_DIR}/run_scanvi_benchmark.py \
            --dataset ${DATASET} \
            --reference ${REF_PATH} \
            --query ${QUERY_PATH} \
            --output_dir ${OUT_DIR} \
            --ref_label_key ${LABEL_KEY} \
            --ref_batch_key ${BATCH_KEY} \
            --save_model\""

    if [[ "${DRY_RUN}" == true ]]; then
        echo "[DRY RUN] ${DATASET}:"
        echo "  ${SBATCH_CMD}"
        echo ""
    else
        echo "[SUBMIT] ${DATASET}"
        eval "${SBATCH_CMD}"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo "============================================================"
echo "Summary: submitted=${SUBMITTED}, skipped=${SKIPPED}"
echo "============================================================"
