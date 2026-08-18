#!/bin/bash
# Download SingleR/scType prediction files from arseven and run local evaluation.
# Usage: bash download_and_evaluate_singler_sctype.sh

set -euo pipefail

LOCAL_DIR="results/benchmark/singler_sctype_batch"
SCRATCH_ROOT="${MLLMCELLTYPE_SCRATCH_ROOT:-/scratch/user/${USER:-$(id -un)}}"
REMOTE_DIR="${MLLMCELLTYPE_SINGLER_ROOT:-${SCRATCH_ROOT}/singler_sctype_batch}/results"
PYTHON="${MLLMCELLTYPE_PYTHON:-python3}"

echo "=== Downloading prediction files from arseven ==="
rsync -avz --include='*_predictions.csv' --exclude='*' \
  arseven:${REMOTE_DIR}/ ${LOCAL_DIR}/arseven_raw/

echo ""
echo "=== Renaming files to benchmark naming convention ==="
mkdir -p "${LOCAL_DIR}/all_predictions"

# Mapping from arseven names to benchmark names
declare -A NAME_MAP
NAME_MAP["BladderTS"]="Bladder(TS)"
NAME_MAP["BloodTS"]="Blood(TS)"
NAME_MAP["Bone_MarrowTS"]="Bone Marrow(TS)"
NAME_MAP["EyeTS"]="Eye(TS)"
NAME_MAP["FatTS"]="Fat(TS)"
NAME_MAP["HeartTS"]="Heart(TS)"
NAME_MAP["Large_IntestineTS"]="Large Intestine(TS)"
NAME_MAP["LiverTS"]="Liver(TS)"
NAME_MAP["LungTS"]="Lung(TS)"
NAME_MAP["Lymph_NodeTS"]="Lymph Node(TS)"
NAME_MAP["MammaryTS"]="Mammary(TS)"
NAME_MAP["MuscleTS"]="Muscle(TS)"
NAME_MAP["PancreasTS"]="Pancreas(TS)"
NAME_MAP["ProstateTS"]="Prostate(TS)"
NAME_MAP["Salivary_GlandTS"]="Salivary Gland(TS)"
NAME_MAP["SkinTS"]="Skin(TS)"
NAME_MAP["Small_IntestineTS"]="Small Intestine(TS)"
NAME_MAP["SpleenTS"]="Spleen(TS)"
NAME_MAP["ThymusTS"]="Thymus(TS)"
NAME_MAP["TongueTS"]="Tongue(TS)"
NAME_MAP["TracheaTS"]="Trachea(TS)"
NAME_MAP["UterusTS"]="Uterus(TS)"
NAME_MAP["VasculatureTS"]="Vasculature(TS)"
NAME_MAP["BCL"]="BCL(Cancer)"

for file in "${LOCAL_DIR}"/arseven_raw/*_predictions.csv; do
  basename=$(basename "$file" _predictions.csv)
  if [[ -n "${NAME_MAP[$basename]+_}" ]]; then
    newname="${NAME_MAP[$basename]}"
    cp "$file" "${LOCAL_DIR}/all_predictions/${newname}_predictions.csv"
    echo "  $basename -> $newname"
  else
    echo "  WARNING: No mapping for $basename, copying as-is"
    cp "$file" "${LOCAL_DIR}/all_predictions/${basename}_predictions.csv"
  fi
done

echo ""
echo "=== Running local ontology-aware evaluation ==="
"${PYTHON}" -c "
import sys
sys.path.insert(0, 'analysis/benchmark')
from evaluate_singler_sctype_local import process_predictions
from pathlib import Path

results = process_predictions(Path('results/benchmark/singler_sctype_batch/all_predictions'))
if results is not None:
    print()
    print('=== SUMMARY ===')
    print(results.to_string(index=False))
"

echo ""
echo "=== Done ==="
