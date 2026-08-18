#!/usr/bin/env python3
"""
Collect all per-dataset SingleR/scType prediction CSVs into a summary.
Computes Cell Ontology-based match scores (same as mLLMCelltype evaluation).

Usage:
    python collect_results.py --results_dir /path/to/results --output summary.csv
"""

import argparse
import os
import re

import pandas as pd


def calculate_match_score(predicted, reference):
    """Cell Ontology-based 3-tier scoring: 1.0 (exact), 0.5 (partial), 0.0 (mismatch)."""
    pred = predicted.lower().strip()
    ref = reference.lower().strip()

    if pred == ref:
        return 1.0

    # Check containment (one is substring of the other)
    if pred in ref or ref in pred:
        return 0.5

    # Check shared key terms
    pred_words = set(re.split(r"[\s/,_-]+", pred))
    ref_words = set(re.split(r"[\s/,_-]+", ref))

    # Remove common stop words
    stop_words = {"cell", "cells", "type", "like", "positive", "negative", "alpha", "beta"}
    pred_key = pred_words - stop_words
    ref_key = ref_words - stop_words

    if pred_key and ref_key and pred_key & ref_key:
        return 0.5

    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_results = []

    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.endswith("_predictions.csv"):
            continue

        dataset_name = fname.replace("_predictions.csv", "")
        fpath = os.path.join(args.results_dir, fname)
        df = pd.read_csv(fpath)

        # Compute Cell Ontology match scores
        singler_scores = []
        sctype_scores = []

        for _, row in df.iterrows():
            ref = str(row["reference"])
            singler_pred = str(row["singler_prediction"])
            singler_scores.append(calculate_match_score(singler_pred, ref))

            if pd.notna(row.get("sctype_prediction")):
                sctype_pred = str(row["sctype_prediction"])
                sctype_scores.append(calculate_match_score(sctype_pred, ref))

        singler_acc = sum(singler_scores) / len(singler_scores) if singler_scores else None
        sctype_acc = sum(sctype_scores) / len(sctype_scores) if sctype_scores else None

        all_results.append({
            "dataset": dataset_name,
            "n_clusters": len(df),
            "singler_accuracy": round(singler_acc, 4) if singler_acc is not None else None,
            "sctype_accuracy": round(sctype_acc, 4) if sctype_acc is not None else None,
            "singler_exact_match": round(df["singler_accuracy"].iloc[0], 4) if "singler_accuracy" in df else None,
            "sctype_exact_match": round(df["sctype_accuracy"].iloc[0], 4) if "sctype_accuracy" in df else None,
        })

        print(f"  {dataset_name}: SingleR={singler_acc:.3f}, scType={sctype_acc:.3f}" if sctype_acc else f"  {dataset_name}: SingleR={singler_acc:.3f}, scType=N/A")

    summary = pd.DataFrame(all_results)
    summary.to_csv(args.output, index=False)
    print(f"\nSummary saved to {args.output}")
    print(f"Processed {len(all_results)} datasets")
    if len(all_results) > 0:
        print(f"Mean SingleR accuracy: {summary['singler_accuracy'].mean():.3f}")
        print(f"Mean scType accuracy: {summary['sctype_accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()
