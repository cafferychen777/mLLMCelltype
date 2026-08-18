#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate scANVI Benchmark Results
==================================
Compare scANVI cell-type predictions against ground-truth annotations using
the same cell-ontology-aware matching logic employed for other methods in the
LLMCelltype benchmark.

Scoring scheme (consistent with popV comparison evaluation):
  - 1.0  exact or equivalent match
  - 0.5  hierarchical / partial match
  - 0.0  no match

Usage
-----
    python evaluate_scanvi_results.py \
        --dataset Thymus \
        --predictions /path/to/Thymus_scanvi_predictions.csv \
        --query /path/to/query.h5ad \
        --query_label_key cell_type \
        --output_dir /path/to/output

The script can also compare against LLM and popV results if paths are given.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================================================
# Cell-type ontology hierarchy (shared across all evaluation scripts)
# ==========================================================================
CELL_TYPE_HIERARCHY = {
    # Highest level
    "Hematopoietic cells": [
        "T cells", "B cells", "NK cells", "Monocytes",
        "Dendritic cells", "Granulocytes", "Megakaryocytes", "Erythroid cells",
    ],
    "Myeloid cells": ["Monocytes", "Granulocytes", "Dendritic cells", "Megakaryocytes"],
    "Lymphoid cells": ["T cells", "B cells", "NK cells"],
    # T cells
    "CD4+ T cells": [
        "CD4+ effector memory T cells", "CD4+ central memory T cells",
        "Naive CD4+ T cells", "Regulatory T cells",
        "Activated CD4+ T cells", "CD4+ effector T cells",
    ],
    "CD8+ T cells": [
        "CD8+ effector memory T cells", "CD8+ central memory T cells",
        "Naive CD8+ T cells", "MAIT cells",
        "Activated CD8+ T cells", "CD8+ effector T cells",
    ],
    "T cells": [
        "CD4+ T cells", "CD8+ T cells", "Gamma delta T cells",
        "NKT cells", "T helper cells", "Thymocytes",
    ],
    # B cells
    "B cells": [
        "Memory B cells", "Naive B cells", "Atypical memory B cells",
        "Plasma cells", "Transitional B cells",
    ],
    # NK cells
    "NK cells": ["CD56 bright NK cells", "CD56 dim NK cells", "Proliferating NK cells"],
    # Monocytes / macrophages
    "Monocytes": ["Classical monocytes", "Non-classical monocytes", "Intermediate monocytes"],
    "Macrophages": ["Tissue resident macrophages", "Inflammatory macrophages"],
    # Dendritic cells
    "Dendritic cells": [
        "Myeloid dendritic cells", "Plasmacytoid dendritic cells",
        "Conventional Dendritic Cells 1 (cDC1)",
        "Conventional Dendritic Cells 2 (cDC2)", "Langerhans cells",
    ],
    "Myeloid dendritic cells": [
        "Conventional Dendritic Cells 1 (cDC1)",
        "Conventional Dendritic Cells 2 (cDC2)",
    ],
    # Granulocytes
    "Granulocytes": ["Neutrophils", "Eosinophils", "Basophils", "Mast cells"],
    # Epithelial (lung / general)
    "Epithelial cells": [
        "AT1 cells", "AT2 cells", "Basal epithelial cells", "Club cells",
        "Ciliated epithelial cells", "Goblet cells", "Secretory cells",
        "Ionocytes", "Neuroendocrine cells",
    ],
    # Endothelial
    "Endothelial cells": [
        "Capillary endothelial cells", "Arterial endothelial cells",
        "Venous endothelial cells", "Lymphatic endothelial cells",
    ],
    # Stromal
    "Stromal cells": ["Fibroblasts", "Myofibroblasts", "Pericytes", "Smooth muscle cells"],
    # Thymic
    "Thymic epithelial cells": [
        "Cortical thymic epithelial cells", "Medullary thymic epithelial cells",
    ],
}

EQUIVALENT_NAMES = {
    "NK cells": ["Natural Killer Cells", "Natural killer cells", "natural killer cell"],
    "CD8+ effector memory T cells": ["Effector Memory CD8+ T cell (TEM)", "CD8+ TEM"],
    "CD4+ effector memory T cells": ["CD4+ TEM"],
    "Classical monocytes": ["Classical Monocytes", "CD14+ Monocytes", "classical monocyte"],
    "Non-classical monocytes": ["CD16+ Monocytes", "non-classical monocyte"],
    "Intermediate monocytes": ["intermediate monocyte"],
    "Regulatory T cells": ["Regulatory T cells (Tregs)", "Tregs", "T regulatory cells", "regulatory T cell"],
    "Gamma delta T cells": ["gdT", "Gamma Delta T cells"],
    "CD56 bright NK cells": ["NK_CD56_Bright", "CD56bright NK cells"],
    "CD56 dim NK cells": ["NK_CD56_Dim", "CD56dim NK cells"],
    "Proliferating NK cells": ["Proliferating cells"],
    "CD4+ central memory T cells": ["CD4+ TCM", "Central Memory CD4+ T cells"],
    "CD8+ central memory T cells": ["CD8+ TCM", "Central Memory CD8+ T cells"],
    "Conventional Dendritic Cells 1 (cDC1)": ["cDC1", "Type 1 conventional dendritic cells"],
    "Conventional Dendritic Cells 2 (cDC2)": ["cDC2", "Type 2 conventional dendritic cells"],
    "Macrophages": ["Monocytes/Macrophages", "macrophage"],
    "Fibroblasts": ["Stromal cells", "Connective tissue cells", "fibroblast"],
    # Lung cell types (TS ↔ Cell Ontology ↔ common names)
    "type II pneumocyte": [
        "Epithelial cell (alveolar type II)", "AT2 cell", "AT2",
        "alveolar type 2 cell", "alveolar type II cell",
        "pulmonary alveolar type 2 cell", "pulmonary alveolar type II cell",
        "type 2 pneumocyte",
    ],
    "type I pneumocyte": [
        "Epithelial cell (alveolar type I)", "AT1 cell", "AT1",
        "alveolar type 1 cell", "alveolar type I cell",
        "pulmonary alveolar type 1 cell", "pulmonary alveolar type I cell",
        "type 1 pneumocyte",
    ],
    "lung ciliated cell": [
        "Epithelial cell (ciliated)", "ciliated epithelial cell",
        "ciliated cell", "airway ciliated cell",
    ],
    "club cell": ["Epithelial cell (club)", "Clara cell", "secretory cell"],
    "basal cell": ["Epithelial cell (basal)", "basal epithelial cell"],
    # Endothelial subtypes
    "endothelial cell of lymphatic vessel": [
        "Endothelial cell (lymphatic)", "lymphatic endothelial cell",
    ],
    "endothelial cell of artery": [
        "arterial endothelial cell", "Endothelial cell (arterial)",
    ],
    "capillary endothelial cell": [
        "Endothelial cell (capillary)", "microvascular endothelial cell",
    ],
    "blood vessel endothelial cell": [
        "vascular endothelial cell", "Endothelial cell (vascular)",
    ],
    # Cardiac cell types
    "cardiac muscle cell": [
        "Myocyte (cardiac)", "Myocyte (cardiac, cytoplasmic)",
        "cardiomyocyte", "cardiac myocyte",
    ],
    "fibroblast of cardiac tissue": [
        "cardiac fibroblast", "Fibroblast I", "Fibroblast II", "Fibroblast III",
    ],
    "cardiac endothelial cell": [
        "Endothelial cell (cardiac microvascular)",
        "Endothelial cell (vascular) I", "Endothelial cell (vascular) II",
        "Endothelial cell (vascular) III",
    ],
    "smooth muscle cell": [
        "Pericyte/SMC I", "Pericyte/SMC II", "vascular smooth muscle cell",
    ],
    # Immune cell mappings
    "dendritic cell": [
        "Immune (DC/macrophage)", "myeloid dendritic cell",
    ],
    "mast cell": ["Immune (mast cell)"],
    "B cell": ["Immune (B cell)", "B lymphocyte"],
    "T cell": ["Immune (T cell)", "T lymphocyte"],
    "NK cell": ["Immune (NK cell)", "natural killer cell"],
    # Thymus / T cell subtypes
    "CD4-positive, alpha-beta T cell": [
        "CD4-positive helper T cell", "CD4+ T cell", "helper T cell",
    ],
    "CD8-positive, alpha-beta T cell": [
        "CD8+ T cell", "cytotoxic T cell",
    ],
    "regulatory T cell": [
        "naive regulatory T cell", "Treg", "CD4+ regulatory T cell",
    ],
    "alpha-beta T cell": [
        "mature alpha-beta T cell",
    ],
    # Pericytes
    "pericyte": ["pericyte cell", "mural cell"],
    # Pancreas cell types (CELLxGENE ↔ TS naming)
    "pancreatic acinar cell": ["acinar cell", "acinar cell of pancreas"],
    "pancreatic ductal cell": ["ductal cell", "duct epithelial cell"],
    "type B pancreatic cell": ["beta cell", "pancreatic beta cell", "pancreatic B cell"],
    "pancreatic A cell": ["alpha cell", "pancreatic alpha cell"],
    "pancreatic D cell": ["delta cell", "pancreatic delta cell"],
    "pancreatic PP cell": ["PP cell", "pancreatic polypeptide cell"],
    "pancreatic stellate cell": ["stellate cell", "hepatic stellate cell"],
    # Liver cell types (CELLxGENE ↔ TS naming)
    "hepatocyte": [
        "hepatoblast", "periportal region hepatocyte",
        "centrilobular region hepatocyte", "midzonal region hepatocyte",
    ],
    "Kupffer cell": ["kupffer cell", "liver macrophage"],
    "endothelial cell of hepatic sinusoid": [
        "liver sinusoidal endothelial cell", "LSEC",
    ],
    "intrahepatic cholangiocyte": ["cholangiocyte", "bile duct epithelial cell"],
    # Cross-tissue equivalents
    "erythrocyte": ["erythroblast", "erythroid cell", "red blood cell"],
    "endothelial cell": [
        "endothelial cell of vascular tree", "vascular endothelial cell",
    ],
    "stromal cell": ["mesenchymal cell", "mesenchymal stem cell"],
    "mononuclear phagocyte": ["monocyte-derived cell"],
    # Kidney cell types (CELLxGENE ↔ TS naming)
    "kidney epithelial cell": [
        "epithelial cell of proximal tubule",
        "kidney proximal convoluted tubule epithelial cell",
        "kidney loop of Henle thick ascending limb epithelial cell",
        "kidney distal convoluted tubule epithelial cell",
        "kidney collecting duct principal cell",
        "kidney collecting duct intercalated cell",
        "kidney connecting tubule epithelial cell",
    ],
}

# Build reverse lookup
_REVERSE_EQUIVALENTS: dict[str, str] = {}
for _canonical, _aliases in EQUIVALENT_NAMES.items():
    for _alias in _aliases:
        _REVERSE_EQUIVALENTS[_alias] = _canonical


def is_equivalent_name(ref_type: str, pred_type: str) -> bool:
    """Check if two cell-type names are semantically equivalent."""
    if ref_type == pred_type:
        return True
    if ref_type in EQUIVALENT_NAMES and pred_type in EQUIVALENT_NAMES[ref_type]:
        return True
    if pred_type in _REVERSE_EQUIVALENTS and ref_type == _REVERSE_EQUIVALENTS[pred_type]:
        return True
    # Symmetric check
    if pred_type in EQUIVALENT_NAMES and ref_type in EQUIVALENT_NAMES[pred_type]:
        return True
    if ref_type in _REVERSE_EQUIVALENTS and pred_type == _REVERSE_EQUIVALENTS[ref_type]:
        return True
    return False


def calculate_match_score(
    reference: str | float, prediction: str | float
) -> tuple[float, str]:
    """Calculate the match score between a reference and predicted cell type.

    Returns
    -------
    score : float
        1.0 for full match, 0.5 for partial/hierarchical, 0.0 for no match.
    match_type : str
        Human-readable description of the match category.
    """
    if pd.isna(reference) or pd.isna(prediction):
        return 0.0, "NaN value"

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # --- Normalised forms for substring checks ---
    ref_norm = reference.lower().replace("-", " ").replace("(", "").replace(")", "")
    pred_norm = prediction.lower().replace("-", " ").replace("(", "").replace(")", "")

    # 1. Exact match (case-insensitive, ignoring punctuation)
    if ref_norm == pred_norm:
        return 1.0, "Full match (exact)"

    # 2. Singular / plural variation
    if ref_norm.rstrip("s") == pred_norm.rstrip("s"):
        return 1.0, "Full match (singular/plural)"

    # 3. Equivalent / alias names
    if is_equivalent_name(reference, prediction):
        return 1.0, "Full match (equivalent name)"

    # 4. Known abbreviation pairs
    abbrev_pairs = [
        ("natural killer cells", "nk cells"),
        ("gamma delta t cells", "gd t cells"),
        ("alpha beta t cells", "ab t cells"),
    ]
    for full, abbr in abbrev_pairs:
        if (full in ref_norm and abbr in pred_norm) or (
            abbr in ref_norm and full in pred_norm
        ):
            return 1.0, "Full match (abbreviation)"

    # 5. Developmental / state relationships (partial = 0.5)
    state_relations = [
        ("cd4+ memory t cell", "activated t cell", "Cell state"),
        ("cd8+ memory t cell", "cytotoxic t cell", "Cell state"),
        ("memory b cell", "b cell", "Cell state"),
        ("naive b cell", "b cell", "Cell state"),
        ("immature b cell", "precursor b cell", "Cell state"),
        ("monocyte", "macrophage", "Cell state"),
        ("erythroblast", "erythrocyte", "Cell state"),
        ("hepatoblast", "hepatocyte", "Cell state"),
        ("kupffer cell", "macrophage", "Cell state"),
        ("mononuclear phagocyte", "monocyte", "Cell state"),
        ("innate lymphoid cell", "nk cell", "Cell state"),
        ("stromal cell", "fibroblast", "Cell state"),
        ("hepatic stellate cell", "fibroblast", "Cell state"),
    ]
    for s1, s2, desc in state_relations:
        if (s1 in ref_norm and s2 in pred_norm) or (s2 in ref_norm and s1 in pred_norm):
            return 0.5, desc

    developmental_stages = [
        ("cd4+ thymocyte", "cd4+ t cell", "Development"),
        ("cd8+ thymocyte", "cd8+ t cell", "Development"),
        ("double negative thymocyte", "early t cell", "Development"),
        ("double positive thymocyte", "cortical thymocyte", "Development"),
        ("thymocyte", "early t cell", "Development"),
        ("double positive thymocyte", "thymocyte", "Development"),
        ("double negative thymocyte", "thymocyte", "Development"),
        ("double positive thymocyte", "thymocyte", "Development"),
        ("double negative thymocyte", "dn1 thymic pro t cell", "Development"),
        ("double negative thymocyte", "dn3 thymocyte", "Development"),
        ("double negative thymocyte", "dn4 thymocyte", "Development"),
        ("double positive, alpha beta thymocyte", "dn4 thymocyte", "Development"),
        ("double positive, alpha beta thymocyte", "dn3 thymocyte", "Development"),
        ("double positive, alpha beta thymocyte", "thymocyte", "Development"),
        ("erythroid precursor", "erythrocyte", "Development"),
        ("b cell precursor", "b cell", "Development"),
    ]
    for s1, s2, desc in developmental_stages:
        if (s1 in ref_norm and s2 in pred_norm) or (s2 in ref_norm and s1 in pred_norm):
            return 0.5, desc

    # 6. Hierarchy-based partial matching
    for parent, children in CELL_TYPE_HIERARCHY.items():
        p_norm = parent.lower()
        c_norms = [c.lower() for c in children]
        # prediction is parent, reference is child (or vice versa)
        if pred_norm == p_norm and ref_norm in c_norms:
            return 0.5, f"Hierarchy (specific -> general: {parent})"
        if ref_norm == p_norm and pred_norm in c_norms:
            return 0.5, f"Hierarchy (general -> specific: {parent})"
        # Siblings
        if ref_norm in c_norms and pred_norm in c_norms:
            return 0.5, f"Hierarchy (sibling under {parent})"

    # 7. Core cell type extraction: handle "Immune (macrophage I)" ↔ "macrophage"
    #    and "Fibroblast II" ↔ "fibroblast" patterns
    def extract_core_type(name: str) -> str:
        """Extract the core cell type from qualified names."""
        import re
        # Remove "Immune (...)" wrapper
        m = re.match(r"immune\s*\((.+?)\)", name, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        # Remove "Epithelial cell (...)" wrapper
        m = re.match(r"epithelial cell\s*\((.+?)\)", name, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        # Remove "Endothelial cell (...)" wrapper
        m = re.match(r"endothelial cell\s*\((.+?)\)", name, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        # Remove "Myocyte (...)" wrapper
        m = re.match(r"myocyte\s*\((.+?)\)", name, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        # Remove trailing roman numerals / numbers for subtypes
        cleaned = re.sub(r"\s+(I{1,3}|IV|V|1|2|3|4|5)\s*$", "", name).strip()
        return cleaned.lower()

    ref_core = extract_core_type(reference)
    pred_core = extract_core_type(prediction)

    # If core types match after extraction, it's at least a partial match
    if ref_core and pred_core:
        if ref_core == pred_core:
            return 0.5, "Core type match (subtype resolution)"
        # Check if one core contains the other
        if ref_core in pred_core or pred_core in ref_core:
            return 0.5, "Core type containment"

    # 8. Flexible substring matching for key cell type terms
    key_terms = [
        "macrophage", "fibroblast", "endothelial", "epithelial",
        "monocyte", "neutrophil", "lymphocyte", "pericyte",
        "smooth muscle", "t cell", "b cell", "nk cell",
        "dendritic", "mast cell", "basophil", "eosinophil",
        "adipocyte", "myocyte", "pneumocyte", "ciliated",
        "hepatocyte", "stellate", "acinar", "ductal",
        "erythro", "thymocyte",
    ]
    for term in key_terms:
        if term in ref_norm and term in pred_norm:
            return 0.5, f"Shared key term ({term})"

    # 9. String similarity fallback (very high similarity = partial)
    sim = SequenceMatcher(None, ref_norm, pred_norm).ratio()
    if sim > 0.85:
        return 0.5, f"High string similarity ({sim:.2f})"

    return 0.0, "No match"


# ==========================================================================
# Main evaluation logic
# ==========================================================================
def evaluate_predictions(
    ground_truth: pd.Series,
    predictions: pd.Series,
    method_name: str = "scANVI",
) -> pd.DataFrame:
    """Score every cell and return a per-cell DataFrame.

    Parameters
    ----------
    ground_truth : pd.Series
        Reference cell-type labels indexed by barcode.
    predictions : pd.Series
        Predicted cell-type labels indexed by barcode.
    method_name : str
        Name for logging.

    Returns
    -------
    pd.DataFrame with columns [barcode, reference, prediction, score, match_type].
    """
    common_idx = ground_truth.index.intersection(predictions.index)
    logger.info(
        "%s: evaluating %d cells (ground_truth=%d, predictions=%d)",
        method_name, len(common_idx), len(ground_truth), len(predictions),
    )

    records = []
    for bc in common_idx:
        ref = ground_truth.loc[bc]
        pred = predictions.loc[bc]
        score, mtype = calculate_match_score(ref, pred)
        records.append(
            {"barcode": bc, "reference": ref, "prediction": pred, "score": score, "match_type": mtype}
        )

    df = pd.DataFrame(records)
    return df


def summarise(eval_df: pd.DataFrame, method_name: str) -> dict:
    """Produce summary statistics from an evaluation DataFrame."""
    n = len(eval_df)
    if n == 0:
        return {"method": method_name, "n_cells": 0}

    weighted_acc = eval_df["score"].mean() * 100
    exact_matches = (eval_df["score"] == 1.0).sum()
    partial_matches = (eval_df["score"] == 0.5).sum()
    no_matches = (eval_df["score"] == 0.0).sum()

    summary = {
        "method": method_name,
        "n_cells": n,
        "weighted_accuracy": round(weighted_acc, 2),
        "exact_match_pct": round(exact_matches / n * 100, 2),
        "partial_match_pct": round(partial_matches / n * 100, 2),
        "no_match_pct": round(no_matches / n * 100, 2),
        "exact_matches": int(exact_matches),
        "partial_matches": int(partial_matches),
        "no_matches": int(no_matches),
    }
    return summary


def print_summary(summary: dict) -> None:
    logger.info(
        "  %-20s | cells=%d | weighted_acc=%.2f%% | exact=%.2f%% | partial=%.2f%% | no_match=%.2f%%",
        summary["method"],
        summary["n_cells"],
        summary.get("weighted_accuracy", 0),
        summary.get("exact_match_pct", 0),
        summary.get("partial_match_pct", 0),
        summary.get("no_match_pct", 0),
    )


# ==========================================================================
# CLI
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scANVI predictions against ground-truth annotations.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to scANVI predictions CSV (columns: barcode, scanvi_prediction).",
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="Path to query .h5ad with ground-truth labels.",
    )
    parser.add_argument(
        "--query_label_key", type=str, default="cell_type",
        help="Column in query .obs for ground-truth labels.",
    )
    parser.add_argument(
        "--llm_results", type=str, default=None,
        help="(Optional) Path to LLM results CSV for side-by-side comparison.",
    )
    parser.add_argument(
        "--popv_results", type=str, default=None,
        help="(Optional) Path to popV results CSV for side-by-side comparison.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write evaluation outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluating scANVI results for %s", args.dataset)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    logger.info("Loading query data: %s", args.query)
    adata = sc.read_h5ad(args.query)
    if args.query_label_key not in adata.obs.columns:
        raise KeyError(
            f"Label key '{args.query_label_key}' not found in query .obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    ground_truth = adata.obs[args.query_label_key].astype(str)
    ground_truth.name = "reference"

    # ------------------------------------------------------------------
    # Load scANVI predictions
    # ------------------------------------------------------------------
    logger.info("Loading scANVI predictions: %s", args.predictions)
    scanvi_df = pd.read_csv(args.predictions)
    scanvi_preds = pd.Series(
        scanvi_df["scanvi_prediction"].values, index=scanvi_df["barcode"].values
    )

    # ------------------------------------------------------------------
    # Evaluate scANVI
    # ------------------------------------------------------------------
    scanvi_eval = evaluate_predictions(ground_truth, scanvi_preds, "scANVI")
    scanvi_summary = summarise(scanvi_eval, "scANVI")
    print_summary(scanvi_summary)

    all_summaries = [scanvi_summary]

    # ------------------------------------------------------------------
    # (Optional) Evaluate LLM results
    # ------------------------------------------------------------------
    if args.llm_results and os.path.exists(args.llm_results):
        logger.info("Loading LLM results: %s", args.llm_results)
        llm_df = pd.read_csv(args.llm_results)
        if "barcode" in llm_df.columns and "prediction" in llm_df.columns:
            llm_preds = pd.Series(
                llm_df["prediction"].values, index=llm_df["barcode"].values
            )
        elif "reference_name" in llm_df.columns and "final_consensus" in llm_df.columns:
            # Cluster-level results: map through query ground-truth
            cluster_map = dict(
                zip(llm_df["reference_name"], llm_df["final_consensus"])
            )
            llm_preds = ground_truth.map(cluster_map)
            llm_preds.index = ground_truth.index
        else:
            logger.warning("LLM results format not recognised, skipping.")
            llm_preds = None

        if llm_preds is not None:
            llm_eval = evaluate_predictions(ground_truth, llm_preds, "LLMCelltype")
            llm_summary = summarise(llm_eval, "LLMCelltype")
            print_summary(llm_summary)
            all_summaries.append(llm_summary)

    # ------------------------------------------------------------------
    # (Optional) Evaluate popV results
    # ------------------------------------------------------------------
    if args.popv_results and os.path.exists(args.popv_results):
        logger.info("Loading popV results: %s", args.popv_results)
        popv_df = pd.read_csv(args.popv_results)
        if "barcodes" in popv_df.columns and "popv_prediction" in popv_df.columns:
            popv_preds = pd.Series(
                popv_df["popv_prediction"].values,
                index=popv_df["barcodes"].values,
            )
        elif "barcode" in popv_df.columns and "prediction" in popv_df.columns:
            popv_preds = pd.Series(
                popv_df["prediction"].values, index=popv_df["barcode"].values
            )
        else:
            logger.warning("popV results format not recognised, skipping.")
            popv_preds = None

        if popv_preds is not None:
            popv_eval = evaluate_predictions(ground_truth, popv_preds, "popV")
            popv_summary = summarise(popv_eval, "popV")
            print_summary(popv_summary)
            all_summaries.append(popv_summary)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Per-cell evaluation
    eval_csv = os.path.join(args.output_dir, f"{args.dataset}_scanvi_evaluation.csv")
    scanvi_eval.to_csv(eval_csv, index=False)
    logger.info("Per-cell evaluation saved to %s", eval_csv)

    # Summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(args.output_dir, f"{args.dataset}_evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Evaluation summary saved to %s", summary_csv)

    # Match-type distribution
    match_dist = scanvi_eval["match_type"].value_counts()
    logger.info("scANVI match-type distribution:\n%s", match_dist.to_string())

    # Top no-match pairs
    no_match = scanvi_eval[scanvi_eval["score"] == 0.0]
    if len(no_match) > 0:
        top_mismatches = (
            no_match.groupby(["reference", "prediction"])
            .size()
            .sort_values(ascending=False)
            .head(20)
        )
        logger.info("Top 20 no-match pairs:\n%s", top_mismatches.to_string())

    logger.info("Evaluation complete for %s.", args.dataset)


if __name__ == "__main__":
    main()
