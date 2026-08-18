#!/usr/bin/env bash
# =============================================================================
# scANVI Benchmark Pipeline: Download → Prepare → Run (6 datasets)
#
# Usage:
#   bash submit_pipeline.sh           # Submit all steps
#   bash submit_pipeline.sh download  # Download only
#   bash submit_pipeline.sh prepare   # Prepare only (assumes download done)
#   bash submit_pipeline.sh run       # Run only (assumes prepare done)
#   bash submit_pipeline.sh status    # Check status
# =============================================================================
set -euo pipefail

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
BASE="${MLLMCELLTYPE_SCANVI_ROOT:-${SCRATCH_ROOT}/scanvi_benchmark}"
VENV="${BASE}/.venv/bin/activate"
SCRIPT="${BASE}/scripts/prepare_and_run.py"
LOG_DIR="${BASE}/logs"
mkdir -p "${LOG_DIR}"

STEP="${1:-all}"

# --- Step 1: Download (CPU, medium partition, needs internet) ---
if [[ "${STEP}" == "all" || "${STEP}" == "download" ]]; then
    echo "[SUBMIT] Step 1: Download datasets"
    DOWNLOAD_JOB=$(sbatch --parsable \
        --job-name=scanvi_download \
        --partition=short \
        --nodes=1 \
        --mem=8G \
        --time=02:00:00 \
        --cpus-per-task=2 \
        --output="${LOG_DIR}/download_%j.out" \
        --error="${LOG_DIR}/download_%j.err" \
        --wrap="source ${VENV} && python ${SCRIPT} download")
    echo "  Job ID: ${DOWNLOAD_JOB}"
fi

# --- Step 2: Prepare data (CPU, needs R for Azimuth RDS conversion) ---
if [[ "${STEP}" == "all" || "${STEP}" == "prepare" ]]; then
    DEPEND=""
    if [[ -n "${DOWNLOAD_JOB:-}" ]]; then
        DEPEND="--dependency=afterok:${DOWNLOAD_JOB}"
    fi
    echo "[SUBMIT] Step 2: Prepare datasets (gene name fix + RDS conversion)"
    PREPARE_JOB=$(sbatch --parsable \
        ${DEPEND} \
        --job-name=scanvi_prepare \
        --partition=medium \
        --nodes=1 \
        --mem=64G \
        --time=06:00:00 \
        --cpus-per-task=8 \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        --wrap="module load R/4.4.1-gfbf-2023b && source ${VENV} && python ${SCRIPT} prepare")
    echo "  Job ID: ${PREPARE_JOB}"
fi

# --- Step 3: Run scANVI on each dataset (GPU jobs) ---
if [[ "${STEP}" == "all" || "${STEP}" == "run" ]]; then
    DEPEND=""
    if [[ -n "${PREPARE_JOB:-}" ]]; then
        DEPEND="--dependency=afterok:${PREPARE_JOB}"
    fi

    DATASETS=("thymus" "lca" "gtex_lung" "gtex_heart" "azimuth_pancreas" "azimuth_liver")

    for DS in "${DATASETS[@]}"; do
        echo "[SUBMIT] Step 3: Run scANVI on ${DS}"
        sbatch \
            ${DEPEND} \
            --job-name="scanvi_${DS}" \
            --partition=gpu \
            --gres=gpu:a30:1 \
            --nodes=1 \
            --mem=64G \
            --time=04:00:00 \
            --cpus-per-task=8 \
            --output="${LOG_DIR}/${DS}_%j.out" \
            --error="${LOG_DIR}/${DS}_%j.err" \
            --wrap="source ${VENV} && python ${SCRIPT} run ${DS}"
    done
fi

# --- Status check ---
if [[ "${STEP}" == "status" ]]; then
    source "${VENV}" && python "${SCRIPT}" status
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs:   ls -lt ${LOG_DIR}/"
