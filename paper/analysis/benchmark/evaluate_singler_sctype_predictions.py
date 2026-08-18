#!/usr/bin/env python3
"""
Evaluate SingleR/scType predictions using Claude API-based Cell Ontology matching.

Uses the same matching method as the main mLLMCelltype evaluation pipeline:
- 1.0: Identical terminology / same Cell Ontology term
- 0.5: Shared hierarchical relationship (ancestor/descendant)
- 0.0: No CL relationship

Usage:
    python evaluate_singler_sctype_predictions.py
"""

import anthropic
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

# Paths
PREDICTIONS_DIR = Path("results/benchmark/singler_sctype_batch")
OUTPUT_DIR = PREDICTIONS_DIR
CACHE_FILE = PREDICTIONS_DIR / "match_score_cache.json"

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a cell type annotation expert. You will evaluate the match between reference cell type names and predicted annotations based on the following criteria:

Full match (1.0): Identical terminology and Cell Ontology (CL) terms
Partial match (0.5): Shared hierarchical relationships or aligned broad categories
Mismatch (0.0): Annotations with no shared CL ancestry or divergent classifications

Please output only the score (1.0, 0.5, or 0.0) without any explanation."""


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def evaluate_match(reference, prediction, cache):
    """Evaluate match score using Claude API with caching."""
    key = f"{reference.lower().strip()}|||{prediction.lower().strip()}"
    if key in cache:
        return cache[key]

    # Quick exact match check
    if reference.lower().strip() == prediction.lower().strip():
        cache[key] = 1.0
        return 1.0

    # Normalize common patterns for quick matching
    ref_norm = reference.lower().strip().replace("_", " ").replace("-", " ")
    pred_norm = prediction.lower().strip().replace("_", " ").replace("-", " ")
    if ref_norm == pred_norm:
        cache[key] = 1.0
        return 1.0

    # Use Claude API
    prompt = (
        f"Please evaluate the match between these cell type annotations:\n\n"
        f"Reference: {reference}\n"
        f"Prediction: {prediction}\n\n"
        f"Output only a single number (1.0, 0.5, or 0.0) based on these criteria:\n"
        f"- 1.0 for identical terminology and Cell Ontology (CL) terms\n"
        f"- 0.5 for shared hierarchical relationships or aligned broad categories\n"
        f"- 0.0 for annotations with no shared CL ancestry or divergent classifications"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.content[0].text.strip()
        score = float(score_text)
        if score not in [0.0, 0.5, 1.0]:
            print(f"  Invalid score {score} for '{reference}' vs '{prediction}', defaulting to 0.0")
            score = 0.0
    except Exception as e:
        print(f"  Error evaluating '{reference}' vs '{prediction}': {e}")
        score = 0.0

    cache[key] = score
    return score


def process_predictions():
    """Process all prediction CSV files from SingleR/scType batch."""
    cache = load_cache()
    pred_files = sorted(PREDICTIONS_DIR.glob("*_predictions.csv"))

    if not pred_files:
        print("No prediction files found. Run run_singler_sctype_batch.R first.")
        return

    all_results = []

    for pred_file in pred_files:
        dataset_name = pred_file.stem.replace("_predictions", "")
        print(f"\n=== Evaluating: {dataset_name} ===")

        df = pd.read_csv(pred_file)
        n = len(df)

        # Evaluate SingleR predictions
        singler_scores = []
        for _, row in df.iterrows():
            ref = str(row["reference"])
            pred = str(row["singler_prediction"])
            if pd.isna(row["singler_prediction"]) or pred == "nan":
                singler_scores.append(0.0)
            else:
                score = evaluate_match(ref, pred, cache)
                singler_scores.append(score)

        singler_acc = np.mean(singler_scores)
        print(f"  SingleR accuracy (CO-matched): {singler_acc:.3f}")

        # Evaluate scType predictions
        sctype_scores = []
        for _, row in df.iterrows():
            ref = str(row["reference"])
            pred = str(row.get("sctype_prediction", ""))
            if pd.isna(row.get("sctype_prediction")) or pred == "nan" or pred == "":
                sctype_scores.append(0.0)
            else:
                score = evaluate_match(ref, pred, cache)
                sctype_scores.append(score)

        sctype_acc = np.mean(sctype_scores) if any(s > 0 for s in sctype_scores) else np.nan
        print(f"  scType accuracy (CO-matched): {sctype_acc:.3f}" if not np.isnan(sctype_acc) else "  scType: N/A")

        # Save match scores back to file
        df["singler_match_score"] = singler_scores
        df["sctype_match_score"] = sctype_scores
        df.to_csv(pred_file, index=False)

        all_results.append({
            "dataset": dataset_name,
            "singler_accuracy": round(singler_acc, 4),
            "sctype_accuracy": round(sctype_acc, 4) if not np.isnan(sctype_acc) else None,
            "n_clusters": n,
        })

        # Save cache after each dataset
        save_cache(cache)

    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "singler_sctype_co_matched_results.csv", index=False)
    print(f"\n\nResults saved to: {OUTPUT_DIR / 'singler_sctype_co_matched_results.csv'}")
    print(f"Processed {len(all_results)} datasets")
    print(f"Mean SingleR accuracy (CO): {results_df['singler_accuracy'].mean():.3f}")
    if results_df['sctype_accuracy'].notna().any():
        print(f"Mean scType accuracy (CO): {results_df['sctype_accuracy'].mean():.3f}")


if __name__ == "__main__":
    process_predictions()
