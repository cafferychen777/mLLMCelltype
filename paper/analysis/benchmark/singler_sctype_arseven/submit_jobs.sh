#!/bin/bash
# Submit SingleR/scType benchmark jobs on arseven
# Step 1: Preprocess h5ad → intermediate files (Python, CPU)
# Step 2: Run SingleR/scType per dataset (R, CPU)
#
# Usage: bash submit_jobs.sh [--dry-run]

set -euo pipefail

SCRATCH="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
DATA_DIR="${SCRATCH}/singler_sctype_data/raw"
PREPROC_DIR="${SCRATCH}/singler_sctype_data/preprocessed"
RESULTS_DIR="${SCRATCH}/singler_sctype_results"
SCRIPTS_DIR="${SCRATCH}/singler_sctype_scripts"
LOG_DIR="${SCRATCH}/singler_sctype_logs"
VENV="${SCRATCH}/scanvi_benchmark/.venv"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$PREPROC_DIR" "$RESULTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Step 1: Preprocessing job (Python - convert all h5ad to intermediate format)
# ---------------------------------------------------------------------------
PREPROC_SCRIPT="${LOG_DIR}/preprocess_job.sh"
cat > "$PREPROC_SCRIPT" << 'SLURM'
#!/bin/bash
#SBATCH --job-name=preprocess_h5ad
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=LOGDIR/preprocess_%j.out
#SBATCH --error=LOGDIR/preprocess_%j.err

source VENV/bin/activate
python SCRIPTS_DIR/preprocess_h5ad.py \
    --data_dir DATA_DIR \
    --output_dir PREPROC_DIR
SLURM

# Replace placeholders
sed -i "s|LOGDIR|${LOG_DIR}|g" "$PREPROC_SCRIPT"
sed -i "s|VENV|${VENV}|g" "$PREPROC_SCRIPT"
sed -i "s|SCRIPTS_DIR|${SCRIPTS_DIR}|g" "$PREPROC_SCRIPT"
sed -i "s|DATA_DIR|${DATA_DIR}|g" "$PREPROC_SCRIPT"
sed -i "s|PREPROC_DIR|${PREPROC_DIR}|g" "$PREPROC_SCRIPT"

if $DRY_RUN; then
    echo "[DRY RUN] Would submit preprocessing job"
else
    PREPROC_JOB=$(sbatch --parsable "$PREPROC_SCRIPT")
    echo "Submitted preprocessing job: $PREPROC_JOB"
fi

# ---------------------------------------------------------------------------
# Step 2: SingleR/scType jobs (R) - one per dataset for parallelism
# ---------------------------------------------------------------------------
# Large datasets get more memory
declare -A MEM_OVERRIDE=(
    ["HLCA_Core"]="128G"
    ["Thymus"]="128G"
    ["HGA"]="96G"
    ["TS_Lung_filtered"]="96G"
    ["TS_Lymph_Node_filtered"]="96G"
    ["TS_Blood_filtered"]="96G"
    ["TS_Muscle_filtered"]="96G"
)

DATASETS=(
    "TS_Bladder_filtered" "TS_Blood_filtered" "TS_Bone_Marrow_filtered"
    "TS_Eye_filtered" "TS_Fat_filtered" "TS_Heart_filtered"
    "TS_Large_Intestine_filtered" "TS_Liver_filtered" "TS_Lung_filtered"
    "TS_Lymph_Node_filtered" "TS_Mammary_filtered" "TS_Muscle_filtered"
    "TS_Pancreas_filtered" "TS_Prostate_filtered" "TS_Salivary_Gland_filtered"
    "TS_Skin_filtered" "TS_Small_Intestine_filtered" "TS_Spleen_filtered"
    "TS_Thymus_filtered" "TS_Tongue_filtered" "TS_Trachea_filtered"
    "TS_Uterus_filtered" "TS_Vasculature_filtered"
    "BCL" "HGA" "HGA_smartseq" "HLCA_Core" "InfantGut" "LCA"
    "M1C" "MTG" "MTG_sampled" "Schiller_2020" "Sun_2020" "Thymus"
)

for ds in "${DATASETS[@]}"; do
    MEM="${MEM_OVERRIDE[$ds]:-64G}"

    JOB_SCRIPT="${LOG_DIR}/singler_${ds}_job.sh"
    cat > "$JOB_SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=sr_${ds}
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/singler_${ds}_%j.out
#SBATCH --error=${LOG_DIR}/singler_${ds}_%j.err

module load R/4.3.2

Rscript ${SCRIPTS_DIR}/run_singler_sctype.R \\
    --data_dir ${PREPROC_DIR} \\
    --output_dir ${RESULTS_DIR} \\
    --dataset ${ds}
SLURM

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: sr_${ds} (mem=${MEM})"
    else
        DEP_FLAG=""
        [[ -n "${PREPROC_JOB:-}" ]] && DEP_FLAG="--dependency=afterok:${PREPROC_JOB}"
        JOB_ID=$(sbatch --parsable $DEP_FLAG "$JOB_SCRIPT")
        echo "Submitted: sr_${ds} (job $JOB_ID, mem=${MEM})"
    fi
done

echo ""
echo "Done. Check status: squeue -u \$USER"
echo "Results will be in: ${RESULTS_DIR}/"
