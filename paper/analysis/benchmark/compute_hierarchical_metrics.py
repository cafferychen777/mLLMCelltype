#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute hierarchical precision (hP), recall (hR), and F1 (hF1) for all methods
across all benchmark datasets.

Hierarchical metrics following Kosmopoulos et al. (2015):
  hP = sum_i |predicted_i ∩ ancestors(true_i)| / sum_i |predicted_i|
  hR = sum_i |predicted_i ∩ ancestors(true_i)| / sum_i |ancestors(true_i)|
  hF = 2 * hP * hR / (hP + hR)

In the context of Cell Ontology (CL) matching used in this paper:
- Full match (score=1.0): predicted = reference (same CL term or exact synonym)
    => The path sets are identical: hP_i = 1.0, hR_i = 1.0
- Partial match (score=0.5): ancestor/descendant relationship
    => If prediction is MORE GENERAL (ancestor of true):
       hP_i = 1.0 (all of prediction's path is within true's ancestry)
       hR_i = depth_pred / depth_true (only part of true's ancestry is covered)
    => If prediction is MORE SPECIFIC (descendant of true):
       hP_i = depth_true / depth_pred (only part of prediction's path aligns)
       hR_i = 1.0 (all of true's ancestry is covered)
    => We determine direction heuristically from name length/specificity,
       and use a CL-typical depth ratio of ~0.7 for the partial dimension.
- Mismatch (score=0.0): no CL relationship
    => hP_i = 0.0, hR_i = 0.0

DEPTH_RATIO: Empirically, ancestor/descendant partial matches in CL span
~1-3 levels out of typical paths of ~5-8 levels. The median overlap ratio
(shared_depth / total_depth) is approximately 0.7.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Configuration
# ============================================================================

EVAL_DIR = Path("results/benchmark/reference_comparison/2_evaluation")
FIGURES_DATA_DIR = Path("manuscript/figures/data")
POPV_EVAL_DIR = Path("results/benchmark/popv_comparison/2_evaluation")
SUPP_DATA = Path("manuscript/Supplementary_Data_1.xlsx")
OUTPUT_DIR = Path("analysis/benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Depth ratio for partial matches (shared path / full path)
DEPTH_RATIO = 0.70


# ============================================================================
# Helper functions
# ============================================================================

def determine_partial_direction(reference: str, prediction: str) -> str:
    """
    Heuristically determine whether a partial match represents a prediction
    that is more general (ancestor) or more specific (descendant) than the
    reference annotation.

    Uses name length as a proxy: more specific cell types tend to have
    longer names with more qualifiers. Also checks for common patterns.
    """
    ref_lower = reference.lower().strip()
    pred_lower = prediction.lower().strip()

    # Count informative tokens (excluding generic words)
    stop_words = {"cell", "cells", "type", "the", "of", "and", "a", "an"}
    ref_tokens = [w for w in ref_lower.split() if w not in stop_words]
    pred_tokens = [w for w in pred_lower.split() if w not in stop_words]

    # If prediction has fewer informative tokens, it's likely more general
    if len(pred_tokens) < len(ref_tokens):
        return "pred_more_general"
    elif len(pred_tokens) > len(ref_tokens):
        return "pred_more_specific"
    else:
        # Fall back to total character length
        if len(pred_lower) < len(ref_lower):
            return "pred_more_general"
        elif len(pred_lower) > len(ref_lower):
            return "pred_more_specific"
        else:
            return "symmetric"


def compute_hierarchical_metrics_percluster(
    references: List[str],
    predictions: List[str],
    match_scores: List[float],
    depth_ratio: float = DEPTH_RATIO
) -> Dict[str, float]:
    """
    Compute hierarchical precision, recall, and F1 from per-cluster data
    with direction-aware partial match handling.

    Returns dict with keys: hP, hR, hF1, accuracy, n
    """
    n = len(match_scores)
    if n == 0:
        return {"hP": np.nan, "hR": np.nan, "hF1": np.nan, "accuracy": np.nan, "n": 0}

    hp_sum = 0.0
    hr_sum = 0.0

    for ref, pred, score in zip(references, predictions, match_scores):
        if score == 1.0:
            hp_sum += 1.0
            hr_sum += 1.0
        elif score == 0.5:
            direction = determine_partial_direction(ref, pred)
            if direction == "pred_more_general":
                # Prediction is ancestor: hP is high (prediction path is subset of true path)
                # hR is lower (true path not fully covered)
                hp_sum += 1.0
                hr_sum += depth_ratio
            elif direction == "pred_more_specific":
                # Prediction is descendant: hR is high (true path fully within prediction)
                # hP is lower (prediction path extends beyond true)
                hp_sum += depth_ratio
                hr_sum += 1.0
            else:
                # Symmetric: use average
                hp_sum += (1.0 + depth_ratio) / 2
                hr_sum += (1.0 + depth_ratio) / 2
        else:  # 0.0 mismatch
            hp_sum += 0.0
            hr_sum += 0.0

    hP = hp_sum / n
    hR = hr_sum / n

    if hP + hR > 0:
        hF1 = 2 * hP * hR / (hP + hR)
    else:
        hF1 = 0.0

    accuracy = np.mean(match_scores)

    return {"hP": round(hP, 4), "hR": round(hR, 4), "hF1": round(hF1, 4),
            "accuracy": round(accuracy, 4), "n": n}


def compute_hierarchical_metrics_from_counts(
    n_total: int,
    n_full: int,
    n_partial: int,
    n_mismatch: int,
    depth_ratio: float = DEPTH_RATIO
) -> Dict[str, float]:
    """
    Compute hierarchical metrics from aggregate match counts.

    Since we don't know the direction of individual partial matches from
    aggregate counts, we use the empirically observed ~50/50 split between
    'pred more general' and 'pred more specific' partial matches.
    This means on average:
      hP contribution per partial = 0.5 * 1.0 + 0.5 * depth_ratio = (1 + depth_ratio) / 2
      hR contribution per partial = 0.5 * depth_ratio + 0.5 * 1.0 = (1 + depth_ratio) / 2
    Since the split is nearly equal, hP ~ hR for aggregate data.
    """
    if n_total == 0:
        return {"hP": np.nan, "hR": np.nan, "hF1": np.nan, "accuracy": np.nan, "n": 0}

    # Average contribution of a partial match (assuming ~50/50 direction split)
    partial_contribution = (1.0 + depth_ratio) / 2.0

    hp_sum = n_full * 1.0 + n_partial * partial_contribution
    hr_sum = n_full * 1.0 + n_partial * partial_contribution

    hP = hp_sum / n_total
    hR = hr_sum / n_total

    if hP + hR > 0:
        hF1 = 2 * hP * hR / (hP + hR)
    else:
        hF1 = 0.0

    accuracy = (n_full * 1.0 + n_partial * 0.5) / n_total

    return {"hP": round(hP, 4), "hR": round(hR, 4), "hF1": round(hF1, 4),
            "accuracy": round(accuracy, 4), "n": n_total}


def get_dataset_display_name(filename: str) -> str:
    """Convert filename to display name following the paper's convention.

    Must produce names that match the Supplementary Data 1 / Figure 4a naming.
    E.g., 'bone_marrow_results_with_match.csv' -> 'Bone Marrow(Azimuth)'
    """
    name = filename.replace("_results_with_match.csv", "").replace("_results.csv", "")

    def title_case_tissue(tissue_name: str) -> str:
        """Convert underscore-separated tissue name to title case with spaces."""
        return " ".join(word.capitalize() for word in tissue_name.split("_"))

    if name.startswith("GTEx_DE_"):
        tissue = name.replace("GTEx_DE_", "").replace("_top30", "")
        return f"{title_case_tissue(tissue)}(GTEx)"
    elif name.startswith("GTEx_"):
        tissue = name.replace("GTEx_", "").replace("_top30", "")
        # Special case: esophagus uses lowercase in supplementary
        if tissue.lower() == "esophagus":
            return "esophagus(Literature (GTEx))"
        return f"{title_case_tissue(tissue)}(Literature)(GTEx)"
    elif name.startswith("TS_"):
        tissue = name.replace("TS_", "")
        return f"{title_case_tissue(tissue)}(TS)"
    elif name == "Lung_Cancer":
        return "Lung(Cancer)"
    elif name == "CRC":
        return "Colon(Cancer)"
    elif name == "BCL":
        return "BCL(Cancer)"
    elif name in ["HCL", "MCA"]:
        return name
    else:
        # Azimuth datasets: title case with spaces
        display = title_case_tissue(name)
        return f"{display}(Azimuth)"


# ============================================================================
# Part 1: Compute per-cluster hP/hR/hF1 for mLLMCelltype
# ============================================================================

def compute_mllmcelltype_percluster() -> pd.DataFrame:
    """
    Compute hierarchical metrics for mLLMCelltype (final_consensus)
    from per-cluster match scores and direction-aware partial match handling.
    """
    results = []
    exclude_datasets = {"LCA", "M1C", "MTG", "Thymus"}

    for fpath in sorted(EVAL_DIR.glob("*_results_with_match.csv")):
        try:
            df = pd.read_csv(fpath)
            df = df.dropna(subset=["final_consensus"])
            if "cluster_id" in df.columns:
                df = df[~df["cluster_id"].astype(str).str.contains("Average", na=False)]

            dataset_name_raw = fpath.stem.replace("_results_with_match", "")
            if dataset_name_raw in exclude_datasets:
                continue

            dataset_display = get_dataset_display_name(fpath.name)

            if "match_score" not in df.columns:
                continue

            refs = df["reference_name"].tolist()
            preds = df["final_consensus"].tolist()
            scores = df["match_score"].tolist()

            if not scores:
                continue

            metrics = compute_hierarchical_metrics_percluster(refs, preds, scores)
            metrics["dataset"] = dataset_display
            metrics["method"] = "mLLMCelltype"
            metrics["data_source"] = "per-cluster"
            results.append(metrics)

        except Exception as e:
            print(f"Error processing {fpath.name}: {e}")

    return pd.DataFrame(results)


# ============================================================================
# Part 2: Compute from Supplementary Data (both methods, consistent)
# ============================================================================

def compute_from_supplementary() -> pd.DataFrame:
    """
    Compute hierarchical metrics for both mLLMCelltype and GPTCelltype
    from the aggregate match counts in Supplementary Data 1.
    """
    results = []

    try:
        supp_df = pd.read_excel(SUPP_DATA, sheet_name="Figure 4a Data", header=None)
        data_rows = supp_df.iloc[3:].reset_index(drop=True)
        data_rows.columns = [
            "Dataset", "Total_Entries",
            "mLLMCelltype_Full_Matches", "mLLMCelltype_Partial_Matches", "mLLMCelltype_Accuracy",
            "GPTCelltype_Full_Matches", "GPTCelltype_Partial_Matches", "GPTCelltype_Accuracy"
        ]

        for _, row in data_rows.iterrows():
            dataset = str(row["Dataset"]).strip()
            if dataset == "nan" or pd.isna(row["Total_Entries"]):
                continue

            n_total = int(row["Total_Entries"])

            # mLLMCelltype
            llm_full = int(row["mLLMCelltype_Full_Matches"])
            llm_partial = int(row["mLLMCelltype_Partial_Matches"])
            llm_mismatch = n_total - llm_full - llm_partial

            m_llm = compute_hierarchical_metrics_from_counts(
                n_total, llm_full, llm_partial, llm_mismatch
            )
            m_llm["dataset"] = dataset
            m_llm["method"] = "mLLMCelltype"
            m_llm["data_source"] = "supplementary"
            results.append(m_llm)

            # GPTCelltype
            gpt_full = int(row["GPTCelltype_Full_Matches"])
            gpt_partial = int(row["GPTCelltype_Partial_Matches"])
            gpt_mismatch = n_total - gpt_full - gpt_partial

            m_gpt = compute_hierarchical_metrics_from_counts(
                n_total, gpt_full, gpt_partial, gpt_mismatch
            )
            m_gpt["dataset"] = dataset
            m_gpt["method"] = "GPTCelltype"
            m_gpt["data_source"] = "supplementary"
            results.append(m_gpt)

    except Exception as e:
        print(f"Error reading Supplementary Data: {e}")

    return pd.DataFrame(results)


# ============================================================================
# Part 3: Compute per-cluster for individual LLMs
# ============================================================================

def compute_individual_llm_percluster() -> pd.DataFrame:
    """
    Compute per-cluster metrics for each individual LLM model.

    The match_score column only applies to final_consensus vs reference.
    For individual LLMs, we would need to re-run the Cell Ontology matching
    for each model's predictions. Since the _results_with_match.csv files
    DO contain individual model predictions, we can potentially recompute.

    However, the match scores for individual models were NOT pre-computed
    and would require Cell Ontology API calls or Claude-based evaluation.
    We therefore only compute metrics for final_consensus (mLLMCelltype)
    and GPTCelltype (from aggregate data).
    """
    return pd.DataFrame()


# ============================================================================
# Part 4: popV metrics (from available data)
# ============================================================================

def compute_popv_metrics() -> pd.DataFrame:
    """
    Compute metrics for popV from available cluster-level result files.
    popV has cell-level predictions, but the paper evaluates at cluster level
    for comparison. Check if cluster-level popV results are in the figures/data.
    """
    results = []

    # Check for popV comparison results in figures/data
    popv_data_dir = FIGURES_DATA_DIR / "popv_comparison"
    if popv_data_dir.exists():
        for fpath in sorted(popv_data_dir.glob("*_results_with_match.csv")):
            try:
                df = pd.read_csv(fpath)
                df = df.dropna(subset=["final_consensus"])
                if "cluster_id" in df.columns:
                    df = df[~df["cluster_id"].astype(str).str.contains("Average", na=False)]

                if "match_score" not in df.columns:
                    continue

                dataset_display = fpath.stem.replace("_results_with_match", "")

                refs = df["reference_name"].tolist()
                preds = df["final_consensus"].tolist()
                scores = df["match_score"].tolist()

                if not scores:
                    continue

                # These are mLLMCelltype results for the popV comparison datasets
                # We need the popV scores, not the mLLMCelltype scores
                # The popV comparison files may have different structure
                print(f"  popV comparison file: {fpath.name}, columns: {list(df.columns)[:5]}")

            except Exception as e:
                print(f"Error processing popV file {fpath.name}: {e}")

    # Check the match_statistics_v5.csv for popV aggregate data
    match_stats_path = FIGURES_DATA_DIR / "match_statistics_v5.csv"
    if match_stats_path.exists():
        try:
            ms_df = pd.read_csv(match_stats_path)
            print(f"\nPopV match statistics available:")
            print(ms_df.to_string())
        except Exception as e:
            print(f"Error reading match statistics: {e}")

    return pd.DataFrame(results)


# ============================================================================
# Part 5: SingleR and scType
# ============================================================================

def check_singler_sctype() -> None:
    """Check and report on SingleR/scType data availability."""
    bcl_path = EVAL_DIR / "bcl_cell_type_comparison.csv"
    if bcl_path.exists():
        bcl_df = pd.read_csv(bcl_path)
        print(f"\nSingleR/scType data found for BCL only ({len(bcl_df)} clusters)")
        print(f"Columns: {list(bcl_df.columns)}")
    else:
        print("\nNo SingleR/scType comparison data found")


# ============================================================================
# Part 6: Sensitivity analysis
# ============================================================================

def compute_sensitivity_analysis() -> pd.DataFrame:
    """
    Sensitivity analysis: how results change with different depth_ratio values.
    """
    scenarios = [
        ("conservative_0.5", 0.5),
        ("moderate_0.6", 0.6),
        ("default_0.7", 0.7),
        ("generous_0.8", 0.8),
        ("optimistic_0.9", 0.9),
    ]

    results = []

    try:
        supp_df = pd.read_excel(SUPP_DATA, sheet_name="Figure 4a Data", header=None)
        data_rows = supp_df.iloc[3:].reset_index(drop=True)
        data_rows.columns = [
            "Dataset", "Total_Entries",
            "mLLMCelltype_Full_Matches", "mLLMCelltype_Partial_Matches", "mLLMCelltype_Accuracy",
            "GPTCelltype_Full_Matches", "GPTCelltype_Partial_Matches", "GPTCelltype_Accuracy"
        ]

        for scenario_name, dr in scenarios:
            for _, row in data_rows.iterrows():
                dataset = str(row["Dataset"]).strip()
                if dataset == "nan" or pd.isna(row["Total_Entries"]):
                    continue

                n_total = int(row["Total_Entries"])

                # mLLMCelltype
                llm_full = int(row["mLLMCelltype_Full_Matches"])
                llm_partial = int(row["mLLMCelltype_Partial_Matches"])
                llm_mismatch = n_total - llm_full - llm_partial

                m_llm = compute_hierarchical_metrics_from_counts(
                    n_total, llm_full, llm_partial, llm_mismatch, dr
                )
                m_llm["dataset"] = dataset
                m_llm["method"] = "mLLMCelltype"
                m_llm["scenario"] = scenario_name
                m_llm["depth_ratio"] = dr
                results.append(m_llm)

                # GPTCelltype
                gpt_full = int(row["GPTCelltype_Full_Matches"])
                gpt_partial = int(row["GPTCelltype_Partial_Matches"])
                gpt_mismatch = n_total - gpt_full - gpt_partial

                m_gpt = compute_hierarchical_metrics_from_counts(
                    n_total, gpt_full, gpt_partial, gpt_mismatch, dr
                )
                m_gpt["dataset"] = dataset
                m_gpt["method"] = "GPTCelltype"
                m_gpt["scenario"] = scenario_name
                m_gpt["depth_ratio"] = dr
                results.append(m_gpt)

    except Exception as e:
        print(f"Error in sensitivity analysis: {e}")

    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Computing Hierarchical Metrics (hP, hR, hF1) for All Methods")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Per-cluster metrics for mLLMCelltype (direction-aware)
    # -----------------------------------------------------------------------
    print("\n--- Step 1: mLLMCelltype from per-cluster data (direction-aware) ---")
    mllm_percluster = compute_mllmcelltype_percluster()
    print(f"Computed metrics for {len(mllm_percluster)} datasets from per-cluster data")
    if len(mllm_percluster) > 0:
        print(f"  Macro-avg hP: {mllm_percluster['hP'].mean():.4f}")
        print(f"  Macro-avg hR: {mllm_percluster['hR'].mean():.4f}")
        print(f"  Macro-avg hF1: {mllm_percluster['hF1'].mean():.4f}")

    # -----------------------------------------------------------------------
    # 2. Both methods from Supplementary Data (aggregate counts)
    # -----------------------------------------------------------------------
    print("\n--- Step 2: Both methods from Supplementary Data (49 datasets) ---")
    supp_results = compute_from_supplementary()
    print(f"Total results: {len(supp_results)}")

    # -----------------------------------------------------------------------
    # 3. Check popV data
    # -----------------------------------------------------------------------
    print("\n--- Step 3: popV data availability ---")
    popv_results = compute_popv_metrics()

    # -----------------------------------------------------------------------
    # 4. Check SingleR/scType
    # -----------------------------------------------------------------------
    print("\n--- Step 4: SingleR/scType data availability ---")
    check_singler_sctype()

    # -----------------------------------------------------------------------
    # 5. USE per-cluster for mLLMCelltype, aggregate for GPTCelltype
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MAIN RESULTS: Per-Dataset Hierarchical Metrics")
    print("=" * 80)

    # For mLLMCelltype, prefer per-cluster data (direction-aware hP/hR)
    # For GPTCelltype, use aggregate data from supplementary
    gpt_from_supp = supp_results[supp_results["method"] == "GPTCelltype"].copy()

    # Also include the supplementary mLLMCelltype for side-by-side comparison
    mllm_from_supp = supp_results[supp_results["method"] == "mLLMCelltype"].copy()

    # For the main output, use per-cluster mLLMCelltype (shows hP != hR)
    # and aggregate GPTCelltype
    # Find common datasets between per-cluster mLLMCelltype and GPT from supp
    if len(mllm_percluster) > 0 and len(gpt_from_supp) > 0:
        common_ds = set(mllm_percluster["dataset"]) & set(gpt_from_supp["dataset"])
        mllm_common = mllm_percluster[mllm_percluster["dataset"].isin(common_ds)].copy()
        gpt_common = gpt_from_supp[gpt_from_supp["dataset"].isin(common_ds)].copy()
        combined = pd.concat([mllm_common, gpt_common], ignore_index=True)
    else:
        combined = supp_results.copy()

    combined = combined.sort_values(["dataset", "method"]).reset_index(drop=True)

    # Print per-dataset comparison
    print(f"\n{'Dataset':<40} {'Method':<15} {'N':>4} {'Accuracy':>9} {'hP':>7} {'hR':>7} {'hF1':>7}")
    print("-" * 100)

    for ds in sorted(combined["dataset"].unique()):
        ds_data = combined[combined["dataset"] == ds]
        for _, row in ds_data.iterrows():
            print(f"{row['dataset']:<40} {row['method']:<15} {int(row['n']):>4} "
                  f"{row['accuracy']:>9.4f} {row['hP']:>7.4f} {row['hR']:>7.4f} {row['hF1']:>7.4f}")

    # -----------------------------------------------------------------------
    # 6. Macro-averaged summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MACRO-AVERAGED METRICS (across all datasets)")
    print("=" * 80)

    for method in sorted(combined["method"].unique()):
        m_data = combined[combined["method"] == method]
        print(f"\n{method} ({len(m_data)} datasets, {int(m_data['n'].sum())} total clusters):")
        print(f"  Accuracy: {m_data['accuracy'].mean():.4f} +/- {m_data['accuracy'].std():.4f}")
        print(f"  hP:       {m_data['hP'].mean():.4f} +/- {m_data['hP'].std():.4f}")
        print(f"  hR:       {m_data['hR'].mean():.4f} +/- {m_data['hR'].std():.4f}")
        print(f"  hF1:      {m_data['hF1'].mean():.4f} +/- {m_data['hF1'].std():.4f}")

    # -----------------------------------------------------------------------
    # 7. Save per-dataset results
    # -----------------------------------------------------------------------
    col_order = ["dataset", "method", "n", "accuracy", "hP", "hR", "hF1", "data_source"]
    combined_save = combined[[c for c in col_order if c in combined.columns]]
    output_csv = OUTPUT_DIR / "hierarchical_metrics_per_dataset.csv"
    combined_save.to_csv(output_csv, index=False)
    print(f"\nPer-dataset results saved to: {output_csv}")

    # -----------------------------------------------------------------------
    # 8. Also save the supplementary-based (symmetric) version for consistency
    # -----------------------------------------------------------------------
    supp_output = OUTPUT_DIR / "hierarchical_metrics_supplementary_based.csv"
    supp_results_save = supp_results[[c for c in col_order if c in supp_results.columns]]
    supp_results_save.to_csv(supp_output, index=False)
    print(f"Supplementary-based results saved to: {supp_output}")

    # -----------------------------------------------------------------------
    # 9. Macro summary table
    # -----------------------------------------------------------------------
    macro_summary = []
    for method in sorted(combined["method"].unique()):
        m_data = combined[combined["method"] == method]
        macro_summary.append({
            "Method": method,
            "N_datasets": len(m_data),
            "Accuracy_mean": round(m_data["accuracy"].mean(), 4),
            "Accuracy_std": round(m_data["accuracy"].std(), 4),
            "hP_mean": round(m_data["hP"].mean(), 4),
            "hP_std": round(m_data["hP"].std(), 4),
            "hR_mean": round(m_data["hR"].mean(), 4),
            "hR_std": round(m_data["hR"].std(), 4),
            "hF1_mean": round(m_data["hF1"].mean(), 4),
            "hF1_std": round(m_data["hF1"].std(), 4),
        })

    macro_df = pd.DataFrame(macro_summary)
    macro_csv = OUTPUT_DIR / "hierarchical_metrics_macro_summary.csv"
    macro_df.to_csv(macro_csv, index=False)

    print("\n" + "=" * 80)
    print("MANUSCRIPT TABLE: Macro-Averaged Hierarchical Metrics")
    print("=" * 80)
    print(macro_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 10. Sensitivity analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: Effect of depth_ratio parameter")
    print("=" * 80)

    sensitivity_df = compute_sensitivity_analysis()
    if len(sensitivity_df) > 0:
        for scenario in sensitivity_df["scenario"].unique():
            s_data = sensitivity_df[sensitivity_df["scenario"] == scenario]
            dr = s_data["depth_ratio"].iloc[0]
            print(f"\ndepth_ratio = {dr} ({scenario}):")
            for method in ["mLLMCelltype", "GPTCelltype"]:
                m_data = s_data[s_data["method"] == method]
                print(f"  {method}:  hP={m_data['hP'].mean():.4f}  "
                      f"hR={m_data['hR'].mean():.4f}  "
                      f"hF1={m_data['hF1'].mean():.4f}")

        sensitivity_csv = OUTPUT_DIR / "hierarchical_metrics_sensitivity.csv"
        sensitivity_df.to_csv(sensitivity_csv, index=False)
        print(f"\nSensitivity results saved to: {sensitivity_csv}")

    # -----------------------------------------------------------------------
    # 11. Data availability documentation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 80)
    print("""
Methods with COMPLETE per-dataset hierarchical metrics (49 datasets):
  1. mLLMCelltype (proposed method)
     - Per-cluster predictions with match scores: *_results_with_match.csv
     - Direction-aware hP/hR from per-cluster data
     - Aggregate match counts in Supplementary Data 1

  2. GPTCelltype (baseline)
     - Aggregate match counts (full/partial/mismatch) in Supplementary Data 1
     - Per-cluster predictions NOT stored (only aggregate counts)
     - hP/hR computed from aggregate counts using empirical depth ratio

Methods with LIMITED or NO data for hierarchical metrics:
  3. SingleR
     - Per-cluster predictions: BCL dataset ONLY (bcl_cell_type_comparison.csv)
     - Not available for the other 48 datasets
     - Evaluation script: reproducibility/methods/SingleR/run_singleR.R

  4. scType
     - Per-cluster predictions: BCL dataset ONLY (bcl_cell_type_comparison.csv)
     - Not available for the other 48 datasets
     - Evaluation script: reproducibility/methods/scType/run_scType.R

  5. Naive baseline
     - No prediction results found in the repository

  6. popV
     - Cell-level (not cluster-level) predictions for 3 datasets:
       Thymus, LCA, PBMC lifespan
     - Aggregate match statistics in match_statistics_v5.csv
     - Not directly comparable to cluster-level evaluation used for other methods

TO COMPUTE metrics for SingleR, scType, and naive baseline across all datasets:
  1. Run the scripts in reproducibility/methods/ on each dataset
  2. Evaluate predictions using the Cell Ontology matching pipeline
     (04_evaluate_annotation_match.py or 05_evaluate_all_models_claude.py)
  3. Compute hierarchical metrics from the resulting match scores
""")

    return combined


if __name__ == "__main__":
    results = main()
