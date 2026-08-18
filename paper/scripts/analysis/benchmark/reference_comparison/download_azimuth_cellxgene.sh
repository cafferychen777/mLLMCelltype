#!/bin/bash
#SBATCH --job-name=download_azimuth
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=download_azimuth_%j.out
#SBATCH --error=download_azimuth_%j.err

# Download Azimuth reference expression data from CELLxGENE for SingleR/scType evaluation.
# These are the original publication datasets that Azimuth references were built from.

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SINGLER_ROOT="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}"
DATA_DIR="${SINGLER_ROOT}/data/downloads/azimuth"
mkdir -p "$DATA_DIR"

echo "=== Downloading Azimuth reference datasets from CELLxGENE ==="
echo "Target directory: $DATA_DIR"
echo "Start time: $(date)"

# CELLxGENE download URL format: https://datasets.cellxgene.cziscience.com/{dataset_id}.h5ad

# 1. Adipose - Emont et al. 2022 (Human White Adipose Tissue)
# CELLxGENE ID: 7a5c4b59-efca-4198-9132-03c289db823a, 166K cells, ~1.3 GB
echo ""
echo "--- Downloading Adipose (Emont 2022, ~1.3 GB) ---"
if [ ! -f "$DATA_DIR/azimuth_adipose.h5ad" ]; then
    curl -L -o "$DATA_DIR/azimuth_adipose.h5ad" \
        "https://datasets.cellxgene.cziscience.com/7a5c4b59-efca-4198-9132-03c289db823a.h5ad"
    echo "  Downloaded: $(ls -lh $DATA_DIR/azimuth_adipose.h5ad | awk '{print $5}')"
else
    echo "  Already exists, skipping"
fi

# 2. Heart - Litvinukova et al. 2020 (Cells of the adult human heart)
# CELLxGENE ID: a7f6822d-0e0e-451e-9858-af81392fcb9b, 486K cells, ~2.95 GB
echo ""
echo "--- Downloading Heart (Litvinukova 2020, ~2.95 GB) ---"
if [ ! -f "$DATA_DIR/azimuth_heart.h5ad" ]; then
    curl -L -o "$DATA_DIR/azimuth_heart.h5ad" \
        "https://datasets.cellxgene.cziscience.com/a7f6822d-0e0e-451e-9858-af81392fcb9b.h5ad"
    echo "  Downloaded: $(ls -lh $DATA_DIR/azimuth_heart.h5ad | awk '{print $5}')"
else
    echo "  Already exists, skipping"
fi

# 3. Kidney - Lake et al. 2023 (Integrated snRNA/scRNA-seq)
# CELLxGENE ID: 25b62707-5933-415b-8000-3369ec4af23f, 305K cells, ~2.96 GB
echo ""
echo "--- Downloading Kidney (Lake 2023, ~2.96 GB) ---"
if [ ! -f "$DATA_DIR/azimuth_kidney.h5ad" ]; then
    curl -L -o "$DATA_DIR/azimuth_kidney.h5ad" \
        "https://datasets.cellxgene.cziscience.com/25b62707-5933-415b-8000-3369ec4af23f.h5ad"
    echo "  Downloaded: $(ls -lh $DATA_DIR/azimuth_kidney.h5ad | awk '{print $5}')"
else
    echo "  Already exists, skipping"
fi

# 4. Fetal Development - Cao et al. 2020 (1M cells subset)
# CELLxGENE ID: c3bf819d-423d-435f-b8d0-e36ad6088138, 1M cells, ~4.9 GB
echo ""
echo "--- Downloading Fetal Development (Cao 2020, ~4.9 GB) ---"
if [ ! -f "$DATA_DIR/azimuth_fetal.h5ad" ]; then
    curl -L -o "$DATA_DIR/azimuth_fetal.h5ad" \
        "https://datasets.cellxgene.cziscience.com/c3bf819d-423d-435f-b8d0-e36ad6088138.h5ad"
    echo "  Downloaded: $(ls -lh $DATA_DIR/azimuth_fetal.h5ad | awk '{print $5}')"
else
    echo "  Already exists, skipping"
fi

# 5. Bone Marrow - Balanced BM Reference Map
# CELLxGENE ID: 96c26450-ad18-4e43-8ec6-c84331bba832, 263K cells, ~2.8 GB
echo ""
echo "--- Downloading Bone Marrow (Balanced BM Reference, ~2.8 GB) ---"
if [ ! -f "$DATA_DIR/azimuth_bone_marrow.h5ad" ]; then
    curl -L -o "$DATA_DIR/azimuth_bone_marrow.h5ad" \
        "https://datasets.cellxgene.cziscience.com/96c26450-ad18-4e43-8ec6-c84331bba832.h5ad"
    echo "  Downloaded: $(ls -lh $DATA_DIR/azimuth_bone_marrow.h5ad | awk '{print $5}')"
else
    echo "  Already exists, skipping"
fi

echo ""
echo "=== Download complete ==="
echo "End time: $(date)"
echo ""
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
echo ""
echo "Total size:"
du -sh "$DATA_DIR"
