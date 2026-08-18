#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script evaluates cell type annotation matches between reference and predicted annotations
based on the criteria of fully match, partially match, and mismatch using Cell Ontology.
"""

import pandas as pd
import os
import numpy as np
import requests
import json
import re
from typing import Tuple, List, Dict, Optional
from functools import lru_cache

def normalize_cell_type(annotation: str) -> str:
    """Normalize cell type annotations to a standard format.

    Args:
        annotation: The cell type annotation to normalize

    Returns:
        The normalized cell type annotation
    """
    import re
    """
    Normalize cell type annotation by converting to lowercase and standardizing common variations

    Args:
        annotation (str): Cell type annotation string

    Returns:
        str: Normalized cell type annotation
    """
    # Convert to lowercase and remove extra spaces
    text = annotation.lower().strip()

    # Replace underscores with spaces
    text = text.replace('_', ' ')

    # First remove plurals and common suffixes
    # Only apply suffix removal if it hasn't been applied before
    if not any(suffix in text for suffix in ['cyte', 'blast', 'phil', 'phage', 'drite']):
        text = re.sub(r'(cyte|blast|phil|phage|drite)s?\b', '', text)
    text = re.sub(r'cells?\b', '', text)

    # Standardize common variations
    replacements = {
        'b-': 'b ',      # Standardize hyphenation
        't-': 't ',
        't cell helper': 't helper',  # Standardize T helper cell variations
        't helper cell': 't helper',
        'helper t': 't helper',
        'b cell memory': 'memory b',  # Standardize memory B cell variations
        'memory b cell': 'memory b',
        'b memory cell': 'memory b',
        'b cell antibody secreting': 'antibody secreting b',  # Standardize antibody-secreting B cell
        'antibody secreting b cell': 'antibody secreting b',
        '-like': ' like',
        'pre-': 'pre',   # Standardize prefixes
        'pro-': 'pro',
        ' and ': ' ',    # Remove conjunctions
        '&': ' ',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove trailing 's' if it exists and the word isn't in exceptions
    exceptions = {'cells', 'series', 'mass', 'stress', 'lass', 'cross', 'adipose'}
    words = text.split()
    normalized_words = []

    for word in words:
        if word.endswith('s') and word not in exceptions:
            word = word[:-1]
        normalized_words.append(word)

    # Join words back together
    text = ' '.join(normalized_words)

    # Define common synonyms (only for well-established equivalences)
    synonyms = {
        # Antigen presenting cells
        'antigen presenting': 'antigen-presenting',
        'apc': 'antigen-presenting',

        # T cells
        't-': 't ',
        't lymphocyte': 't cell',
        't-lymphocyte': 't cell',
        't lymphocytes': 't cell',
        't-lymphocytes': 't cell',
        't helper': 't helper cell',
        'helper t': 't helper cell',

        # B cells
        'b-': 'b ',
        'b lymphocyte': 'b cell',
        'b-lymphocyte': 'b cell',
        'b lymphocytes': 'b cell',
        'b-lymphocytes': 'b cell',

        # Common cell type abbreviations
        'oligo': 'oligodendrocyte',
        'oligos': 'oligodendrocyte',
        'astro': 'astrocyte',
        'astros': 'astrocyte',
        'rbc': 'red blood cell',
        'dc': 'dendritic cell',
        'nk': 'natural killer',
        'mono': 'monocyte',
        'monos': 'monocyte',
        'neutro': 'neutrophil',
        'neutros': 'neutrophil',
        'macro': 'macrophage',
        'macros': 'macrophage',
        'fibro': 'fibroblast',
        'fibros': 'fibroblast',
        'endo': 'endothelial',
        'endos': 'endothelial'
    }

    # Apply synonyms
    for old, new in synonyms.items():
        if old in text:
            text = text.replace(old, new)

    return text.strip()

@lru_cache(maxsize=1000)
def query_cell_ontology(term: str) -> Optional[Dict]:
    """
    Query the OLS API for cell ontology information

    Args:
        term (str): Cell type term to query

    Returns:
        Optional[Dict]: Cell ontology information if found
    """
    base_url = "https://www.ebi.ac.uk/ols/api/search"

    # Preprocess the term
    term = term.lower()

    # Handle special cases
    term_variants = [term]

    # Handle B cell numbering
    # Match patterns like 'b3', 'B5', 'b cell 6', 'B-3 cell', etc.
    if re.search(r'[bB][-\s]*\d+|\d+[-\s]*[bB]', term):
        term_variants.append('b cell')
        # Also try with the number removed
        cleaned_term = re.sub(r'\d+', '', term).strip()
        if cleaned_term:
            term_variants.append(cleaned_term)
            if not cleaned_term.endswith(' cell'):
                term_variants.append(f"{cleaned_term} cell")

    # Add 'cell' suffix if not present
    if not term.endswith(' cell') and not term.endswith(' cells'):
        term_variants.append(f"{term} cell")
        term_variants.append(f"{term} cells")

    # Try each variant
    for variant in term_variants:
        # First try exact match
        params = {
            "q": f'"{variant}"',  # Quote the term for exact phrase matching
            "ontology": "cl",
            "exact": "true",
            "queryFields": "label,synonym"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["response"]["numFound"] > 0:
                return data["response"]["docs"][0]
        except Exception as e:
            print(f"Warning: Failed to query OLS API for term '{variant}': {str(e)}")
            continue

    # If no exact match found, try fuzzy search with the original term
    params = {
        "q": term,
        "ontology": "cl",
        "exact": "false",
        "queryFields": "label,synonym"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["response"]["numFound"] > 0:
            # Return the first result that has the highest score
            return data["response"]["docs"][0]
    except Exception as e:
        print(f"Warning: Failed to query OLS API for fuzzy search of term '{term}': {str(e)}")

    return None

@lru_cache(maxsize=1000)
def get_ancestors(term_id: str) -> List[str]:
    """
    Get ancestors of a term from OLS API

    Args:
        term_id (str): Ontology term ID (CL, GO, or UBERON)

    Returns:
        List[str]: List of ancestor terms
    """
    # Determine the ontology and format the ID
    if ':' in term_id:
        ontology, id_num = term_id.split(':', 1)
        if ontology == 'CL':
            ontology = 'cl'
            id_prefix = 'CL_'
        elif ontology == 'GO':
            ontology = 'go'
            id_prefix = 'GO_'
        elif ontology == 'UBERON':
            ontology = 'uberon'
            id_prefix = 'UBERON_'
        else:
            print(f"Warning: Unsupported ontology in ID '{term_id}'")
            return []
    else:
        # Assume it's a CL ID if no prefix
        ontology = 'cl'
        id_prefix = 'CL_'
        id_num = term_id

    # Format the URL correctly
    base_url = f"https://www.ebi.ac.uk/ols/api/ontologies/{ontology}/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{id_prefix}{id_num}/hierarchicalAncestors"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()

        # Check if we have the expected structure
        if "_embedded" in data and "terms" in data["_embedded"]:
            return [term["label"] for term in data["_embedded"]["terms"]]
        else:
            # Try alternative URL format
            alt_url = f"https://www.ebi.ac.uk/ols/api/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{id_prefix}{id_num}/hierarchicalAncestors"
            response = requests.get(alt_url)
            response.raise_for_status()
            data = response.json()
            if "_embedded" in data and "terms" in data["_embedded"]:
                return [term["label"] for term in data["_embedded"]["terms"]]

        print(f"Warning: Unexpected response structure for '{term_id}'")
        return []
    except Exception as e:
        print(f"Warning: Failed to get ancestors for '{term_id}': {str(e)}")
        return []

def get_broad_cell_type(annotation: str) -> str:
    """
    Extract broad cell type from detailed annotation using Cell Ontology

    Args:
        annotation (str): Cell type annotation string

    Returns:
        str: Broad cell type category
    """
    # Query cell ontology
    cl_info = query_cell_ontology(annotation)

    if cl_info:
        # Get ontology ID
        obo_id = cl_info.get("obo_id")
        if not obo_id:
            return normalize_cell_type(annotation)

        # Get ancestors
        ancestors = get_ancestors(obo_id)

        # Find the most specific broad cell type
        broad_types = [
            "cell", "animal cell", "native cell",  # Too general, skip these
            "professional antigen presenting cell",
            "lymphocyte", "myeloid cell", "stromal cell",
            "epithelial cell", "endothelial cell", "muscle cell",
            "connective tissue cell", "neural cell", "blood cell",
            "immune cell", "leukocyte", "granulocyte",
            "fibroblast", "adipocyte", "stem cell",
            "progenitor cell"
        ]

        # Sort ancestors by length to prefer more specific terms
        ancestors.sort(key=len, reverse=True)

        for ancestor in ancestors:
            ancestor_lower = ancestor.lower()
            if any(broad_type in ancestor_lower for broad_type in broad_types):
                return ancestor_lower

    # Fallback to normalized annotation if no ontology match found
    return normalize_cell_type(annotation)

def is_more_specific_b_cell(term: str) -> bool:
    """Check if a term is a more specific type of B cell"""
    specific_terms = ['memory', 'naive', 'plasma', 'antibody', 'secreting', 'germinal', 'follicular']
    return any(spec in term.lower() for spec in specific_terms)

def evaluate_match(reference: str, prediction: str, debug: bool = False) -> Tuple[str, float]:
    """
    Evaluate the match between reference and predicted cell type annotations

    Args:
        reference (str): Reference cell type annotation
        prediction (str): Predicted cell type annotation
        debug (bool): Whether to print debug information

    Returns:
        Tuple[str, float]: Match category and score
    """
    if debug:
        print(f"\nEvaluating: '{reference}' vs '{prediction}'")

    # Step 1: Handle B cell numbering before normalization
    ref_is_b_cell = bool(re.search(r'^[bB][-\s]*\d+', reference))
    if ref_is_b_cell:
        ref_for_norm = 'B cell'
        if debug:
            print(f"Detected B cell with number in reference, using '{ref_for_norm}'")
    else:
        ref_for_norm = reference

    # Step 2: Direct text match after normalization
    ref_norm = normalize_cell_type(ref_for_norm)
    pred_norm = normalize_cell_type(prediction)

    if debug:
        print(f"Normalized: '{ref_norm}' vs '{pred_norm}'")

    if ref_norm == pred_norm:
        if debug:
            print("✓ Exact match after normalization")
        return 'fully match', 1.0

    # Special handling for B cells
    if 'b' in ref_norm and 'b' in pred_norm:
        ref_specific = is_more_specific_b_cell(ref_norm)
        pred_specific = is_more_specific_b_cell(pred_norm)

        # If both are memory B cells or same specific type
        if ('memory b' in ref_norm and 'memory b' in pred_norm) or \
           (ref_norm == pred_norm and ref_specific):
            if debug:
                print("✓ Specific B cell match found")
            return 'fully match', 1.0

        # If one is more specific than the other
        if ref_specific and not pred_specific:
            if debug:
                print("✓ Reference is more specific B cell")
            return 'partially match', 0.5
        elif not ref_specific and pred_specific:
            if debug:
                print("✓ Prediction is more specific B cell")
            return 'partially match', 0.5

        # If both are just B cells
        if debug:
            print("✓ Basic B cell match found")
        return 'fully match', 1.0

    # Step 2: Try cell ontology lookup and broad type matching
    if debug:
        print("Querying cell ontology...")

    # Query cell ontology for both terms
    ref_info = query_cell_ontology(reference)
    pred_info = query_cell_ontology(prediction)

    if debug:
        if ref_info:
            print(f"Reference ontology ID: {ref_info.get('obo_id')}")
            print(f"Reference definition: {ref_info.get('description', [''])[0]}")
        if pred_info:
            print(f"Prediction ontology ID: {pred_info.get('obo_id')}")
            pred_desc = pred_info.get('description', [])
    if pred_desc:
        print(f"Prediction definition: {pred_desc[0]}")
    else:
        print("Prediction definition: Not available")

    # If both terms have ontology IDs, check their relationship
    if ref_info and pred_info:
        ref_id = ref_info.get('obo_id')
        pred_id = pred_info.get('obo_id')

        if ref_id == pred_id:
            if debug:
                print("✓ Same ontology ID")
            return 'fully match', 1.0

        # Get ancestors for both terms
        ref_ancestors = get_ancestors(ref_id)
        pred_ancestors = get_ancestors(pred_id)

        # Check if one is ancestor of the other
        if ref_id in pred_ancestors or pred_id in ref_ancestors:
            if debug:
                print("✓ One term is ancestor of the other")
            return 'partially match', 0.5

        # Check for shared ancestors (excluding very general terms)
        common_ancestors = set(ref_ancestors) & set(pred_ancestors)
        specific_ancestors = {a for a in common_ancestors
                            if not any(term in a.lower()
                                      for term in ['cell', 'native', 'animal'])}

        # Special handling for T helper cells
        if ('t helper' in ref_norm and 't helper' in pred_norm) or \
           (ref_id.startswith('CL:') and pred_id.startswith('CL:') and \
            any('T helper cell' in a for a in ref_ancestors) and \
            any('T helper cell' in a for a in pred_ancestors)):
            if debug:
                print("✓ T helper cell match found")
            return 'fully match', 1.0

        if specific_ancestors:
            if debug:
                print(f"✓ Shared specific ancestors: {specific_ancestors}")
            return 'partially match', 0.5

    # Step 3: Check for well-established relationships
    broad_type_relationships = {
        'dendritic': ['antigen-presenting'],
        't': ['lymphocyte'],
        'b': ['lymphocyte'],
        'nk': ['lymphocyte'],
        'endothelial': ['vascular']
    }

    ref_broad = normalize_cell_type(reference)
    pred_broad = normalize_cell_type(prediction)

    for main_type, related_types in broad_type_relationships.items():
        if (main_type in ref_broad and any(rel in pred_broad for rel in related_types)) or \
           (main_type in pred_broad and any(rel in ref_broad for rel in related_types)):
            if debug:
                print(f"✓ Found well-established relationship: {main_type} related to {related_types}")
            return 'partially match', 0.5

    if debug:
        print("✗ No match found")
    return 'mismatch', 0.0

def process_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Process dataset and add match evaluation

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file

    Returns:
        pd.DataFrame: Processed dataframe with match results
    """
    # Read input file
    df = pd.read_csv(input_file)

    # Evaluate matches with debug info
    matches = []
    for ref, pred in zip(df['reference_name'], df['final_consensus']):
        match_result = evaluate_match(ref, pred, debug=True)
        matches.append(match_result)

    # Add match results
    df['match'] = [m[0] for m in matches]
    df['match_score'] = [m[1] for m in matches]

    # Calculate average score
    avg_score = df['match_score'].mean()

    # Save results
    df.to_csv(output_file, index=False)

    # Print detailed results
    print(f"\nDetailed Results for {os.path.basename(input_file)}:")
    for i, row in df.iterrows():
        print(f"\n{i+1}. Reference: '{row['reference_name']}'")
        print(f"   Prediction: '{row['final_consensus']}'")
        print(f"   Result: {row['match']} (score: {row['match_score']})")

    # Print summary
    print(f"\nMatch Summary:")
    print(df['match'].value_counts())
    print(f"\nAverage Score: {avg_score:.3f}")

    return df

def main():
    """Main function to process all datasets"""
    # Get all processed result files
    base_dir = "results/benchmark/reference_comparison/2_evaluation"
    result_files = [f for f in os.listdir(base_dir) if f.endswith('_results_processed.csv')]

    # Initialize lists to store overall statistics
    all_summaries = []

    # Process each dataset
    for result_file in sorted(result_files):
        input_file = os.path.join(base_dir, result_file)
        output_file = input_file.replace('_processed.csv', '_processed_with_match.csv')

        print(f"\n=== Processing {result_file} ===")
        df = process_dataset(input_file, output_file)

        # Collect summary statistics
        summary = {
            'dataset': result_file,
            'total_pairs': len(df),
            'fully_match': (df['match'] == 'fully match').sum(),
            'partially_match': (df['match'] == 'partially match').sum(),
            'mismatch': (df['match'] == 'mismatch').sum(),
            'average_score': df['match_score'].mean(),
            'std_score': df['match_score'].std()
        }
        all_summaries.append(summary)

    # Convert summaries to DataFrame
    summary_df = pd.DataFrame(all_summaries)

    # Print overall summary
    print("\n=== Overall Summary ===")
    print(f"\nTotal datasets processed: {len(summary_df)}")
    print(f"Average score across all datasets: {summary_df['average_score'].mean():.3f} ± {summary_df['average_score'].std():.3f}")

    # Calculate total matches by type
    total_fully = summary_df['fully_match'].sum()
    total_partially = summary_df['partially_match'].sum()
    total_mismatch = summary_df['mismatch'].sum()
    total_pairs = total_fully + total_partially + total_mismatch

    print(f"\nTotal matches across all datasets ({total_pairs} pairs):")
    print(f"  Fully match: {total_fully} ({total_fully/total_pairs*100:.1f}%)")
    print(f"  Partially match: {total_partially} ({total_partially/total_pairs*100:.1f}%)")
    print(f"  Mismatch: {total_mismatch} ({total_mismatch/total_pairs*100:.1f}%)")

    # Save summary to CSV
    summary_file = os.path.join(base_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()
