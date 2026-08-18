#!/bin/bash
#SBATCH --job-name=prep_azimuth
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep_azimuth_%j.out
#SBATCH --error=logs/prep_azimuth_%j.err

module load R/4.4.1-gfbf-2023b

# R module sets LD_LIBRARY_PATH with all needed libs (libdeflate, libiconv, etc.)
# Only add R 4.4.3 src path for additional compatibility
export LD_LIBRARY_PATH=/modules/lmod/software/R/4.4.3-src-2025b/lib64/R/lib:${LD_LIBRARY_PATH}

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SINGLER_ROOT="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}"
R_LIB="${MLLMCELLTYPE_R_LIB:-${SCRATCH_ROOT}/R_libs}"
export R_LIBS_USER="${HOME}/R/4.4:${HOME}/R/library:${R_LIB}"

cd "${SINGLER_ROOT}"

Rscript scripts/prepare_azimuth_data_v2.R \
  --output_dir "${SINGLER_ROOT}/prepared"

echo "DONE: $(date)"
