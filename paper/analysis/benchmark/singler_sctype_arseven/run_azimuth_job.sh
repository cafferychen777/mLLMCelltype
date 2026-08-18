#!/bin/bash
#SBATCH --job-name=azimuth_singler_sctype
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_azimuth_%j.out
#SBATCH --error=logs/run_azimuth_%j.err

module load R/4.4.1-gfbf-2023b

# R module sets comprehensive LD_LIBRARY_PATH; only add R 4.4.3 src path
export LD_LIBRARY_PATH=/modules/lmod/software/R/4.4.3-src-2025b/lib64/R/lib:${LD_LIBRARY_PATH}

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SINGLER_ROOT="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}"
R_LIB="${MLLMCELLTYPE_R_LIB:-${SCRATCH_ROOT}/R_libs}"
export R_LIBS_USER="${HOME}/R/4.4:${HOME}/R/library:${R_LIB}"

cd "${SINGLER_ROOT}"

# Process each Azimuth dataset
for ds in Azimuth_adipose Azimuth_bone_marrow Azimuth_fetal Azimuth_heart Azimuth_kidney Azimuth_lung Azimuth_pancreas Azimuth_tonsil; do
  echo "===== Processing $ds ===== $(date)"
  Rscript scripts/run_singler_sctype_large.R \
    --prepared_dir prepared \
    --results_dir results \
    --dataset $ds
  echo "===== Done $ds ===== $(date)"
done

echo "ALL DONE: $(date)"
