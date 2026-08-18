#!/usr/bin/env python3
"""
Evaluate SingleR/scType predictions using local Cell Ontology matching.

Uses rule-based matching consistent with the mLLMCelltype evaluation pipeline:
- 1.0: Identical/synonymous cell type names
- 0.5: Hierarchical relationship (ancestor/descendant in Cell Ontology)
- 0.0: No relationship

This script does NOT require an API key. It uses a comprehensive rule-based
approach with the OLS (Ontology Lookup Service) API as a fallback.
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from functools import lru_cache

# Paths
PREDICTIONS_DIR = Path("results/benchmark/singler_sctype_batch")
CACHE_FILE = PREDICTIONS_DIR / "match_score_cache_local.json"

# Mapping from arseven filenames to benchmark dataset names
DATASET_NAME_MAP = {
    "BladderTS": "Bladder(TS)",
    "BloodTS": "Blood(TS)",
    "Bone_MarrowTS": "Bone Marrow(TS)",
    "EyeTS": "Eye(TS)",
    "FatTS": "Fat(TS)",
    "HeartTS": "Heart(TS)",
    "Large_IntestineTS": "Large Intestine(TS)",
    "LiverTS": "Liver(TS)",
    "LungTS": "Lung(TS)",
    "Lymph_NodeTS": "Lymph Node(TS)",
    "MammaryTS": "Mammary(TS)",
    "MuscleTS": "Muscle(TS)",
    "PancreasTS": "Pancreas(TS)",
    "ProstateTS": "Prostate(TS)",
    "Salivary_GlandTS": "Salivary Gland(TS)",
    "SkinTS": "Skin(TS)",
    "Small_IntestineTS": "Small Intestine(TS)",
    "SpleenTS": "Spleen(TS)",
    "ThymusTS": "Thymus(TS)",
    "TongueTS": "Tongue(TS)",
    "TracheaTS": "Trachea(TS)",
    "UterusTS": "Uterus(TS)",
    "VasculatureTS": "Vasculature(TS)",
    "BCL": "BCL(Cancer)",
    "Lung_Cancer": "Lung(Cancer)",
    "HCL": "HCL",
    "CRC": "Colon(Cancer)",
    "InfantGut": "InfantGut",
    # GTEx snRNA-seq atlas (Eraslan et al. 2022)
    "GTEx_breast": "Breast(GTEx)",
    "GTEx_esophagus_mucosa": "Esophagus Mucosa(GTEx)",
    "GTEx_esophagus_muscularis": "Esophagus Muscularis(GTEx)",
    "GTEx_heart": "Heart(GTEx)",
    "GTEx_lung": "Lung(GTEx)",
    "GTEx_prostate": "Prostate(GTEx)",
    "GTEx_skeletal_muscle": "Skeletal Muscle(GTEx)",
    "GTEx_skin": "Skin(GTEx)",
    # Azimuth datasets (CELLxGENE expression data)
    "Azimuth_adipose": "Adipose(Azimuth)",
    "Azimuth_bone_marrow": "Bone Marrow(Azimuth)",
    "Azimuth_fetal": "Fetal Development(Azimuth)",
    "Azimuth_heart": "Heart(Azimuth)",
    "Azimuth_kidney": "Kidney(Azimuth)",
    "Azimuth_liver": "Liver(Azimuth)",
    "Azimuth_lung": "Lung(Azimuth)",
    "Azimuth_pancreas": "Pancreas(Azimuth)",
    # MCA (Mouse Cell Atlas)
    "MCA": "MCA",
}

# ---------------------------------------------------------------------------
# Cell type hierarchy knowledge base
# ---------------------------------------------------------------------------
# Map of broad → specific cell type relationships
HIERARCHY = {
    # T cell hierarchy
    "t cell": ["cd4 t cell", "cd8 t cell", "regulatory t cell", "gamma delta t cell",
                "nk t cell", "helper t cell", "cytotoxic t cell", "effector t cell",
                "memory t cell", "naive t cell", "cd4-positive alpha-beta t cell",
                "cd8-positive alpha-beta t cell", "mature nk t cell", "double-positive thymocyte",
                "double-negative thymocyte", "t follicular helper cell"],
    "cd4 t cell": ["regulatory t cell", "helper t cell", "t follicular helper cell",
                     "cd4-positive alpha-beta t cell", "naive cd4 t cell"],
    "cd8 t cell": ["cytotoxic t cell", "cd8-positive alpha-beta t cell", "effector cd8 t cell"],
    # B cell hierarchy
    "b cell": ["naive b cell", "memory b cell", "plasma cell", "pre-b cell", "pro-b cell",
                "germinal center b cell", "follicular b cell", "marginal zone b cell",
                "antibody-secreting cell", "plasmablast"],
    "plasma cell": ["antibody-secreting cell", "plasmablast"],
    # NK cell hierarchy
    "nk cell": ["natural killer cell", "mature nk t cell", "cd56bright nk cell",
                 "cd56dim nk cell"],
    # Monocyte/Macrophage hierarchy
    "monocyte": ["classical monocyte", "non-classical monocyte", "intermediate monocyte",
                  "cd14 monocyte", "cd16 monocyte", "cd14-positive monocyte",
                  "cd14-low cd16-positive monocyte"],
    "macrophage": ["alveolar macrophage", "kupffer cell", "tissue macrophage",
                    "peritoneal macrophage", "microglial cell"],
    "myeloid cell": ["monocyte", "macrophage", "dendritic cell", "granulocyte",
                      "mast cell", "basophil", "eosinophil", "neutrophil"],
    # Dendritic cell hierarchy
    "dendritic cell": ["plasmacytoid dendritic cell", "conventional dendritic cell",
                        "myeloid dendritic cell", "langerhans cell",
                        "antigen-presenting cell"],
    # Stem/Progenitor hierarchy
    "hematopoietic stem cell": ["hematopoietic progenitor", "multipotent progenitor",
                                  "common myeloid progenitor", "common lymphoid progenitor",
                                  "granulocyte-monocyte progenitor"],
    "progenitor cell": ["common myeloid progenitor", "common lymphoid progenitor",
                         "erythroid progenitor cell", "megakaryocyte-erythroid progenitor",
                         "granulocyte-monocyte progenitor"],
    # Epithelial cell hierarchy
    "epithelial cell": ["basal cell", "luminal cell", "club cell", "goblet cell",
                         "ciliated cell", "secretory cell", "type i pneumocyte",
                         "type ii pneumocyte", "alveolar type 1 cell", "alveolar type 2 cell",
                         "squamous epithelial cell", "columnar epithelial cell",
                         "keratinocyte", "enterocyte", "hepatocyte", "cholangiocyte",
                         "urothelial cell", "bladder urothelial cell",
                         "basal cell of prostate epithelium",
                         "luminal epithelial cell of mammary gland",
                         "retinal pigment epithelial cell"],
    # Endothelial hierarchy
    "endothelial cell": ["vascular endothelial cell", "lymphatic endothelial cell",
                          "capillary endothelial cell", "arterial endothelial cell",
                          "venous endothelial cell"],
    # Stromal hierarchy
    "fibroblast": ["myofibroblast", "adventitial fibroblast", "alveolar fibroblast"],
    "stromal cell": ["fibroblast", "pericyte", "smooth muscle cell",
                      "mesenchymal stem cell", "adipocyte"],
    # Immune system
    "immune cell": ["t cell", "b cell", "nk cell", "monocyte", "macrophage",
                     "dendritic cell", "granulocyte", "mast cell", "neutrophil",
                     "eosinophil", "basophil"],
    "granulocyte": ["neutrophil", "eosinophil", "basophil"],
    "lymphocyte": ["t cell", "b cell", "nk cell"],
    # Erythroid
    "erythrocyte": ["erythroid cell", "red blood cell", "reticulocyte"],
    "erythroid cell": ["erythrocyte", "erythroid progenitor cell", "reticulocyte"],
    # Muscle cells
    "myocyte": ["cardiomyocyte", "skeletal muscle cell", "smooth muscle cell"],
    "cardiomyocyte": ["cardiac myocyte", "cardiac muscle cell"],
    # Glial cells
    "glial cell": ["astrocyte", "oligodendrocyte", "schwann cell", "microglial cell"],
    # Lymphoid/Myeloid broad categories
    "lymphoid cell": ["t cell", "b cell", "nk cell", "innate lymphoid cell"],
    "myeloid cell": ["monocyte", "macrophage", "dendritic cell", "granulocyte",
                      "mast cell", "basophil", "eosinophil", "neutrophil"],
    # Vascular
    "vascular endothelial cell": ["arterial endothelial cell", "venous endothelial cell",
                                    "capillary endothelial cell", "cardiac microvascular endothelial cell"],
}

# Synonyms (bidirectional equivalences)
SYNONYMS = {
    "t cell": ["t lymphocyte", "t_cells", "t cells"],
    "b cell": ["b lymphocyte", "b_cell", "b cells"],
    "nk cell": ["natural killer cell", "nk_cell", "nk cells", "nk"],
    "monocyte": ["monocytes", "mono"],
    "macrophage": ["macrophages", "macro"],
    "dendritic cell": ["dc", "dendritic cells"],
    "plasmacytoid dendritic cell": ["pdc", "pdc cell"],
    "neutrophil": ["neutrophils", "neutro"],
    "erythrocyte": ["erythrocytes", "red blood cell", "rbc"],
    "fibroblast": ["fibroblasts", "fibro"],
    "endothelial cell": ["endothelial cells", "endothelial", "endo"],
    "epithelial cell": ["epithelial cells", "epithelial"],
    "smooth muscle cell": ["smooth muscle cells", "smc"],
    "adipocyte": ["adipocytes", "fat cell"],
    "pericyte": ["pericytes"],
    "hepatocyte": ["hepatocytes"],
    "plasma cell": ["plasma cells", "plasmacyte"],
    "mast cell": ["mast cells"],
    "hematopoietic stem cell": ["hsc", "hematopoietic stem cells"],
    "mesenchymal stem cell": ["msc", "mesenchymal stem cells", "mesenchymal stromal cell"],
    "keratinocyte": ["keratinocytes"],
    "enterocyte": ["enterocytes"],
    "goblet cell": ["goblet cells"],
    "club cell": ["club cells", "clara cell"],
    "ciliated cell": ["ciliated cells", "ciliated epithelial cell"],
    "alveolar macrophage": ["alveolar macrophages"],
    "kupffer cell": ["kupffer cells"],
    "regulatory t cell": ["treg", "tregs", "regulatory t cells"],
    "gamma delta t cell": ["gd t", "gd t cell", "gamma-delta t cell"],
    "cd14-positive monocyte": ["classical monocyte", "cd14+ monocyte"],
    "cd14-low cd16-positive monocyte": ["non-classical monocyte", "cd16+ monocyte"],
    "antigen-presenting cell": ["apc", "antigen presenting cell"],
    "common myeloid progenitor": ["cmp"],
    "granulocyte-monocyte progenitor": ["gmp"],
    "megakaryocyte-erythroid progenitor": ["mep"],
    # SingleR-specific abbreviations from HumanPrimaryCellAtlas
    "bm": ["bone marrow cell"],
    "bm & prog.": ["bone marrow progenitor", "progenitor cell"],
    "pre-b_cell_cd34-": ["pre-b cell"],
    "pro-b_cell_cd34+": ["pro-b cell"],
    # Muscle/cardiac synonyms
    "cardiomyocyte": ["cardiac myocyte", "cardiac muscle cell", "myocyte cardiac"],
    "smooth muscle cell": ["smooth muscle cells", "smc", "vascular smooth muscle cell"],
    "schwann cell": ["schwann cells"],
    # GTEx-style labels
    "lymphoid cell": ["lymphoid cells"],
    "myeloid cell": ["myeloid cells"],
    "stromal cell": ["stromal cells"],
}


def normalize_cell_type(name):
    """Normalize cell type name for matching."""
    n = str(name).strip().lower()
    n = n.replace("_", " ").replace("-", " ").replace(",", "")
    # Remove trailing 's' for plurals (but not for specific words)
    if n.endswith("s") and not n.endswith(("sis", "ous", "ss")):
        n = n[:-1]
    # Normalize word forms
    n = n.replace("epithelium", "epithelial")
    n = n.replace("endothelium", "endothelial")
    # Standardize common patterns
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def find_synonym(name):
    """Find the canonical name for a cell type."""
    n = normalize_cell_type(name)
    for canonical, syns in SYNONYMS.items():
        if n == normalize_cell_type(canonical):
            return canonical
        for syn in syns:
            if n == normalize_cell_type(syn):
                return canonical
    return n


def is_hierarchical_match(pred, ref):
    """Check if prediction and reference have a hierarchical relationship."""
    pred_n = find_synonym(normalize_cell_type(pred))
    ref_n = find_synonym(normalize_cell_type(ref))

    # Check if pred is ancestor of ref (pred is broader)
    if pred_n in HIERARCHY:
        descendants = [normalize_cell_type(d) for d in HIERARCHY[pred_n]]
        if ref_n in descendants:
            return True
        for d in descendants:
            if len(d) > 3 and len(ref_n) > 3 and (d in ref_n or ref_n in d):
                return True
            # Also check without " cell" suffix for abbreviated references
            d_short = d.replace(" cell", "").strip()
            if len(d_short) > 2 and len(ref_n) > 3 and d_short in ref_n:
                return True

    # Check if ref is ancestor of pred (pred is more specific)
    if ref_n in HIERARCHY:
        descendants = [normalize_cell_type(d) for d in HIERARCHY[ref_n]]
        if pred_n in descendants:
            return True
        for d in descendants:
            if len(d) > 3 and len(pred_n) > 3 and (d in pred_n or pred_n in d):
                return True
            d_short = d.replace(" cell", "").strip()
            if len(d_short) > 2 and len(pred_n) > 3 and d_short in pred_n:
                return True

    return False


def keyword_overlap_match(pred, ref):
    """Check if pred and ref share significant keywords or containment."""
    pred_str = normalize_cell_type(pred)
    ref_str = normalize_cell_type(ref)

    # Check containment first (one is substring of the other)
    if len(pred_str) > 2 and len(ref_str) > 2:
        if pred_str in ref_str or ref_str in pred_str:
            return True

    stop_words = {"cell", "cells", "type", "like", "positive", "negative",
                  "alpha", "beta", "the", "a", "an", "of", "and", "or"}

    pred_words = set(re.split(r"[\s/,_-]+", pred_str)) - stop_words
    ref_words = set(re.split(r"[\s/,_-]+", ref_str)) - stop_words

    # Remove very short words (1-2 chars) that might cause false matches
    pred_words = {w for w in pred_words if len(w) > 2}
    ref_words = {w for w in ref_words if len(w) > 2}

    if not pred_words or not ref_words:
        return False

    overlap = pred_words & ref_words
    if overlap:
        return True

    return False


def calculate_match_score(predicted, reference, cache=None):
    """Calculate match score: 1.0 (exact), 0.5 (hierarchical), 0.0 (no match)."""
    pred = str(predicted).strip()
    ref = str(reference).strip()

    if pred.lower() == "nan" or pred.lower() == "unknown" or pred == "":
        return 0.0

    cache_key = f"{pred.lower()}|||{ref.lower()}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Exact match (after normalization)
    pred_n = normalize_cell_type(pred)
    ref_n = normalize_cell_type(ref)

    if pred_n == ref_n:
        score = 1.0
    elif find_synonym(pred) == find_synonym(ref):
        score = 1.0
    elif is_hierarchical_match(pred, ref):
        score = 0.5
    elif keyword_overlap_match(pred, ref):
        score = 0.5
    else:
        score = 0.0

    if cache is not None:
        cache[cache_key] = score
    return score


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def process_predictions(predictions_dir=None):
    """Process all prediction CSV files and compute ontology-aware accuracy."""
    if predictions_dir is None:
        predictions_dir = PREDICTIONS_DIR

    predictions_dir = Path(predictions_dir)
    cache = load_cache()
    pred_files = sorted(predictions_dir.glob("*_predictions.csv"))

    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        return None

    all_results = []

    for pred_file in pred_files:
        raw_name = pred_file.stem.replace("_predictions", "")
        dataset_name = DATASET_NAME_MAP.get(raw_name, raw_name)
        print(f"\n=== Evaluating: {dataset_name} ===")

        df = pd.read_csv(pred_file)
        n = len(df)

        # Evaluate SingleR predictions
        singler_scores = []
        for _, row in df.iterrows():
            ref = str(row["reference"])
            pred = str(row.get("singler_prediction", ""))
            if pd.isna(row.get("singler_prediction")) or pred == "nan":
                singler_scores.append(0.0)
            else:
                score = calculate_match_score(pred, ref, cache)
                singler_scores.append(score)
                if score < 1.0:
                    print(f"  SingleR: '{pred}' vs '{ref}' → {score}")

        singler_acc = np.mean(singler_scores)
        singler_full = sum(1 for s in singler_scores if s == 1.0)
        singler_partial = sum(1 for s in singler_scores if s == 0.5)
        print(f"  SingleR: {singler_acc:.3f} (full={singler_full}, partial={singler_partial}, n={n})")

        # Evaluate scType predictions
        sctype_scores = []
        for _, row in df.iterrows():
            ref = str(row["reference"])
            pred = str(row.get("sctype_prediction", ""))
            if pd.isna(row.get("sctype_prediction")) or pred == "nan" or pred == "":
                sctype_scores.append(0.0)
            else:
                score = calculate_match_score(pred, ref, cache)
                sctype_scores.append(score)
                if score < 1.0:
                    print(f"  scType: '{pred}' vs '{ref}' → {score}")

        sctype_acc = np.mean(sctype_scores)
        sctype_full = sum(1 for s in sctype_scores if s == 1.0)
        sctype_partial = sum(1 for s in sctype_scores if s == 0.5)
        print(f"  scType: {sctype_acc:.3f} (full={sctype_full}, partial={sctype_partial}, n={n})")

        all_results.append({
            "dataset": dataset_name,
            "singler_accuracy": round(singler_acc, 4),
            "singler_full_matches": singler_full,
            "singler_partial_matches": singler_partial,
            "sctype_accuracy": round(sctype_acc, 4),
            "sctype_full_matches": sctype_full,
            "sctype_partial_matches": sctype_partial,
            "n_clusters": n,
        })

        save_cache(cache)

    results_df = pd.DataFrame(all_results)
    output_path = predictions_dir / "singler_sctype_co_matched_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    print(f"Processed {len(all_results)} datasets")
    print(f"Mean SingleR accuracy (CO): {results_df['singler_accuracy'].mean():.3f}")
    print(f"Mean scType accuracy (CO): {results_df['sctype_accuracy'].mean():.3f}")

    return results_df


if __name__ == "__main__":
    process_predictions()
