#!/bin/bash
# Submit SingleR/scType benchmark pipeline on arseven
# Step 1: Python preprocessing (CPU, fast)
# Step 2: R analysis (CPU, memory-intensive)
#
# Usage: bash submit_all.sh

set -euo pipefail

SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
SCRATCH="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}"
SCANVI_ROOT="${MLLMCELLTYPE_SCANVI_ROOT:-${SCRATCH_ROOT}/scanvi_benchmark}"
VENV="${MLLMCELLTYPE_VENV:-${SCANVI_ROOT}/.venv}"
SCRIPTS="${SCRATCH}/scripts"
DATA_DIR="${SCRATCH}/data"
PREPARED_DIR="${SCRATCH}/prepared"
RESULTS_DIR="${SCRATCH}/results"
LOG_DIR="${SCRATCH}/logs"

mkdir -p "$PREPARED_DIR" "$RESULTS_DIR" "$LOG_DIR"

# --- Step 1: Python preprocessing (convert h5ad -> mtx+csv) ---
cat > "${LOG_DIR}/prep_data.sh" << SLURM
#!/bin/bash
#SBATCH --job-name=prep_singler
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/prep_data_%j.out
#SBATCH --error=${LOG_DIR}/prep_data_%j.err

source ${VENV}/bin/activate
python ${SCRIPTS}/prepare_data.py --data_dir ${DATA_DIR} --output_dir ${PREPARED_DIR}
SLURM

PREP_JOB=$(sbatch --parsable "${LOG_DIR}/prep_data.sh")
echo "Submitted preprocessing: job ${PREP_JOB}"

# --- Step 2: R analysis (depends on Step 1) ---
# Split into batches for parallel processing
# Batch 1: TS datasets A-L (13 datasets)
cat > "${LOG_DIR}/singler_batch1.sh" << SLURM
#!/bin/bash
#SBATCH --job-name=singler_b1
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG_DIR}/singler_batch1_%j.out
#SBATCH --error=${LOG_DIR}/singler_batch1_%j.err

module load R/4.3.2

for ds in BladderTS BloodTS Bone_MarrowTS EyeTS FatTS HeartTS Large_IntestineTS LiverTS LungTS Lymph_NodeTS MammaryTS MuscleTS PancreasTS; do
    if [ -d "${PREPARED_DIR}/\${ds}" ]; then
        echo "Processing \${ds}..."
        Rscript ${SCRIPTS}/run_singler_sctype.R --prepared_dir ${PREPARED_DIR} --results_dir ${RESULTS_DIR} --dataset \${ds}
    fi
done
SLURM

# Batch 2: TS datasets M-V (10 datasets)
cat > "${LOG_DIR}/singler_batch2.sh" << SLURM
#!/bin/bash
#SBATCH --job-name=singler_b2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG_DIR}/singler_batch2_%j.out
#SBATCH --error=${LOG_DIR}/singler_batch2_%j.err

module load R/4.3.2

for ds in ProstateTS Salivary_GlandTS SkinTS Small_IntestineTS SpleenTS ThymusTS TongueTS TracheaTS UterusTS VasculatureTS; do
    if [ -d "${PREPARED_DIR}/\${ds}" ]; then
        echo "Processing \${ds}..."
        Rscript ${SCRIPTS}/run_singler_sctype.R --prepared_dir ${PREPARED_DIR} --results_dir ${RESULTS_DIR} --dataset \${ds}
    fi
done
SLURM

# Batch 3: Non-TS datasets (11 datasets)
cat > "${LOG_DIR}/singler_batch3.sh" << SLURM
#!/bin/bash
#SBATCH --job-name=singler_b3
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG_DIR}/singler_batch3_%j.out
#SBATCH --error=${LOG_DIR}/singler_batch3_%j.err

module load R/4.3.2

for ds in BCL HGA HGA_smartseq HLCA InfantGut LCA M1C MTG MTG_sampled Schiller_2020 Sun_2020 Thymus_dev; do
    if [ -d "${PREPARED_DIR}/\${ds}" ]; then
        echo "Processing \${ds}..."
        Rscript ${SCRIPTS}/run_singler_sctype.R --prepared_dir ${PREPARED_DIR} --results_dir ${RESULTS_DIR} --dataset \${ds}
    fi
done
SLURM

# Submit R batches with dependency on preprocessing
B1_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} "${LOG_DIR}/singler_batch1.sh")
B2_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} "${LOG_DIR}/singler_batch2.sh")
B3_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} "${LOG_DIR}/singler_batch3.sh")

echo "Submitted R batch 1: job ${B1_JOB} (depends on ${PREP_JOB})"
echo "Submitted R batch 2: job ${B2_JOB} (depends on ${PREP_JOB})"
echo "Submitted R batch 3: job ${B3_JOB} (depends on ${PREP_JOB})"

# --- Step 3: Merge results (depends on all R batches) ---
cat > "${LOG_DIR}/merge_results.sh" << SLURM
#!/bin/bash
#SBATCH --job-name=merge_singler
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/merge_%j.out
#SBATCH --error=${LOG_DIR}/merge_%j.err

module load R/4.3.2

Rscript -e "
# Merge all per-dataset prediction files into one summary
results_dir <- '${RESULTS_DIR}'
csv_files <- list.files(results_dir, pattern='_predictions.csv', full.names=TRUE)
all_results <- data.frame()
for (f in csv_files) {
  ds_name <- gsub('_predictions.csv', '', basename(f))
  df <- read.csv(f, stringsAsFactors=FALSE)
  singler_acc <- mean(tolower(df\\\$singler_prediction) == tolower(df\\\$reference), na.rm=TRUE)
  sctype_acc <- mean(tolower(df\\\$sctype_prediction) == tolower(df\\\$reference), na.rm=TRUE)
  all_results <- rbind(all_results, data.frame(
    dataset=ds_name, singler_accuracy=singler_acc, sctype_accuracy=sctype_acc,
    n_clusters=nrow(df), stringsAsFactors=FALSE))
}
write.csv(all_results, file.path(results_dir, 'singler_sctype_per_dataset_results.csv'), row.names=FALSE)
cat('Merged', nrow(all_results), 'datasets\n')
cat('Mean SingleR:', round(mean(all_results\\\$singler_accuracy, na.rm=TRUE)*100, 1), '%\n')
cat('Mean scType:', round(mean(all_results\\\$sctype_accuracy, na.rm=TRUE)*100, 1), '%\n')
"
SLURM

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${B1_JOB}:${B2_JOB}:${B3_JOB} "${LOG_DIR}/merge_results.sh")
echo "Submitted merge: job ${MERGE_JOB} (depends on ${B1_JOB},${B2_JOB},${B3_JOB})"

echo ""
echo "=== Pipeline Summary ==="
echo "  Prep:    ${PREP_JOB}"
echo "  Batch 1: ${B1_JOB} (TS A-L)"
echo "  Batch 2: ${B2_JOB} (TS M-V)"
echo "  Batch 3: ${B3_JOB} (non-TS)"
echo "  Merge:   ${MERGE_JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${RESULTS_DIR}/singler_sctype_per_dataset_results.csv"
