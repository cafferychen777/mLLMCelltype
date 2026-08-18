#!/bin/bash
# Master script to run the entire Azimuth + MCA SingleR/scType pipeline on arseven.
#
# Steps:
# 1. Download CELLxGENE data (SLURM job)
# 2. Preprocess all datasets (SLURM job, depends on step 1)
# 3. Run SingleR/scType (SLURM job, depends on step 2)
#
# Usage:
#   bash submit_azimuth_pipeline.sh

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
BASE_DIR="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}"
SCRIPT_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/data"
AZIMUTH_DIR="$DATA_DIR/downloads/azimuth"
PREPARED_DIR="$DATA_DIR/prepared_azimuth"
RESULTS_DIR="$BASE_DIR/results"
MARKER_DIR="$BASE_DIR/markers"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$AZIMUTH_DIR" "$PREPARED_DIR" "$RESULTS_DIR" "$MARKER_DIR" "$LOG_DIR"
cd "$BASE_DIR"

echo "=== Azimuth + MCA SingleR/scType Pipeline ==="
echo "Base: $BASE_DIR"
echo ""

# Step 1: Submit download job
cat > "$LOG_DIR/download_azimuth.sh" << 'DOWNLOAD_EOF'
#!/bin/bash
#SBATCH --job-name=dl_azimuth
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=__LOG_DIR__/download_%j.out
#SBATCH --error=__LOG_DIR__/download_%j.err

DATA_DIR="__BASE_DIR__/data/downloads/azimuth"
mkdir -p "$DATA_DIR"

echo "=== Download start: $(date) ==="

# Adipose (1.3 GB)
if [ ! -f "$DATA_DIR/azimuth_adipose.h5ad" ]; then
    echo "Downloading Adipose..."
    curl -L -o "$DATA_DIR/azimuth_adipose.h5ad" \
        "https://datasets.cellxgene.cziscience.com/7a5c4b59-efca-4198-9132-03c289db823a.h5ad"
fi

# Heart (2.95 GB)
if [ ! -f "$DATA_DIR/azimuth_heart.h5ad" ]; then
    echo "Downloading Heart..."
    curl -L -o "$DATA_DIR/azimuth_heart.h5ad" \
        "https://datasets.cellxgene.cziscience.com/a7f6822d-0e0e-451e-9858-af81392fcb9b.h5ad"
fi

# Kidney (2.96 GB)
if [ ! -f "$DATA_DIR/azimuth_kidney.h5ad" ]; then
    echo "Downloading Kidney..."
    curl -L -o "$DATA_DIR/azimuth_kidney.h5ad" \
        "https://datasets.cellxgene.cziscience.com/25b62707-5933-415b-8000-3369ec4af23f.h5ad"
fi

# Fetal Dev (4.9 GB)
if [ ! -f "$DATA_DIR/azimuth_fetal.h5ad" ]; then
    echo "Downloading Fetal Development..."
    curl -L -o "$DATA_DIR/azimuth_fetal.h5ad" \
        "https://datasets.cellxgene.cziscience.com/c3bf819d-423d-435f-b8d0-e36ad6088138.h5ad"
fi

# Bone Marrow (2.8 GB)
if [ ! -f "$DATA_DIR/azimuth_bone_marrow.h5ad" ]; then
    echo "Downloading Bone Marrow..."
    curl -L -o "$DATA_DIR/azimuth_bone_marrow.h5ad" \
        "https://datasets.cellxgene.cziscience.com/96c26450-ad18-4e43-8ec6-c84331bba832.h5ad"
fi

echo ""
echo "=== Download complete: $(date) ==="
ls -lh "$DATA_DIR"
du -sh "$DATA_DIR"
DOWNLOAD_EOF
sed -i "s|__BASE_DIR__|${BASE_DIR}|g; s|__LOG_DIR__|${LOG_DIR}|g" "$LOG_DIR/download_azimuth.sh"

echo "Step 1: Submitting download job..."
DL_JOB=$(sbatch "$LOG_DIR/download_azimuth.sh" | awk '{print $4}')
echo "  Download job: $DL_JOB"

# Step 2: Submit preprocessing job (depends on download)
cat > "$LOG_DIR/preprocess_azimuth.sh" << 'PREPROCESS_EOF'
#!/bin/bash
#SBATCH --job-name=prep_azimuth
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=__LOG_DIR__/preprocess_%j.out
#SBATCH --error=__LOG_DIR__/preprocess_%j.err

module load Python/3.11.5-GCCcore-13.2.0

BASE_DIR="__BASE_DIR__"
DATA_DIR="$BASE_DIR/data/downloads/azimuth"
MARKER_DIR="$BASE_DIR/markers"
PREPARED_DIR="$BASE_DIR/data/prepared_azimuth"

# Activate venv
source __VENV__/bin/activate

echo "=== Preprocessing start: $(date) ==="

# Preprocess CELLxGENE datasets
python "$BASE_DIR/scripts/prepare_azimuth_cellxgene.py" \
    --data_dir "$DATA_DIR" \
    --marker_dir "$MARKER_DIR" \
    --output_dir "$PREPARED_DIR"

# Preprocess MCA
if [ -f "$BASE_DIR/data/MCA_BatchRemoved_Merge_dge.h5ad" ]; then
    python "$BASE_DIR/scripts/prepare_mca_data.py" \
        --h5ad "$BASE_DIR/data/MCA_BatchRemoved_Merge_dge.h5ad" \
        --cellinfo "$BASE_DIR/data/MCA_BatchRemoved_Merge_dge_cellinfo.csv" \
        --marker_stats "$MARKER_DIR/MCA_markers_L1_stats.csv" \
        --output_dir "$PREPARED_DIR" \
        --max_cells 100000
fi

# Also prep HLCA (Lung) if not already done
if [ -f "$BASE_DIR/data/HLCA_Core.h5ad" ] && [ ! -d "$PREPARED_DIR/Azimuth_lung" ]; then
    echo "Preparing HLCA/Lung..."
    python "$BASE_DIR/scripts/prepare_azimuth_cellxgene.py" \
        --data_dir "$BASE_DIR/data" \
        --marker_dir "$MARKER_DIR" \
        --output_dir "$PREPARED_DIR" \
        --dataset lung
fi

echo ""
echo "=== Preprocessing complete: $(date) ==="
ls -la "$PREPARED_DIR"/
PREPROCESS_EOF
SCANVI_ROOT="${MLLMCELLTYPE_SCANVI_ROOT:-${SCRATCH_ROOT}/scanvi_benchmark}"
VENV="${MLLMCELLTYPE_VENV:-${SCANVI_ROOT}/.venv}"
sed -i "s|__BASE_DIR__|${BASE_DIR}|g; s|__LOG_DIR__|${LOG_DIR}|g; s|__VENV__|${VENV}|g" "$LOG_DIR/preprocess_azimuth.sh"

echo "Step 2: Submitting preprocess job (after download)..."
PREP_JOB=$(sbatch --dependency=afterok:$DL_JOB "$LOG_DIR/preprocess_azimuth.sh" | awk '{print $4}')
echo "  Preprocess job: $PREP_JOB (depends on $DL_JOB)"

# Step 3: Submit SingleR/scType job (depends on preprocessing)
cat > "$LOG_DIR/run_azimuth_singler.sh" << 'RUN_EOF'
#!/bin/bash
#SBATCH --job-name=singler_azimuth
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=__LOG_DIR__/singler_%j.out
#SBATCH --error=__LOG_DIR__/singler_%j.err

module load R/4.4.1-gfbf-2023b

export R_LIBS_USER="__R_LIB__"

BASE_DIR="__BASE_DIR__"
PREPARED_DIR="$BASE_DIR/data/prepared_azimuth"
RESULTS_DIR="$BASE_DIR/results"

echo "=== SingleR/scType start: $(date) ==="

Rscript "$BASE_DIR/scripts/run_azimuth_singler_sctype.R" \
    --prepared_dir "$PREPARED_DIR" \
    --results_dir "$RESULTS_DIR"

echo ""
echo "=== SingleR/scType complete: $(date) ==="
echo ""
echo "Results:"
cat "$RESULTS_DIR/azimuth_singler_sctype_results.csv"
RUN_EOF
R_LIB="${MLLMCELLTYPE_R_LIB:-${SCRATCH_ROOT}/R_libs}"
sed -i "s|__BASE_DIR__|${BASE_DIR}|g; s|__LOG_DIR__|${LOG_DIR}|g; s|__R_LIB__|${R_LIB}|g" "$LOG_DIR/run_azimuth_singler.sh"

echo "Step 3: Submitting SingleR/scType job (after preprocess)..."
RUN_JOB=$(sbatch --dependency=afterok:$PREP_JOB "$LOG_DIR/run_azimuth_singler.sh" | awk '{print $4}')
echo "  SingleR job: $RUN_JOB (depends on $PREP_JOB)"

echo ""
echo "=== Pipeline submitted ==="
echo "  Download:   $DL_JOB"
echo "  Preprocess: $PREP_JOB (after $DL_JOB)"
echo "  SingleR:    $RUN_JOB (after $PREP_JOB)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs:   ls $LOG_DIR/*.out"
