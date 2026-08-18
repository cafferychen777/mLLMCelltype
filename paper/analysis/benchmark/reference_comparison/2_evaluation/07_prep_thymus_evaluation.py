#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cell_ontology_evaluation import CellOntologyEvaluator
import pandas as pd
import obonet
import networkx as nx
from typing import Dict, Set
import re

def get_cell_type_synonyms(graph: nx.DiGraph) -> Dict[str, str]:
    """
    Extract cell type names and their synonyms from the Cell Ontology graph
    """
    name_to_cl = {}

    # Special cases for composite cell types
    composite_types = {
        'monocyte and macrophage': 'CL:0000091',  # myeloid cell
        'nk and nkt cell': 'CL:0000814',  # mature NK T cell
        'monocytes and macrophages': 'CL:0000091',  # myeloid cell
        'erythroid precursor': 'CL:0000038',  # erythroid progenitor cell
        'erythroid precursors': 'CL:0000038',  # erythroid progenitor cell
        'erythroid cells': 'CL:0000038',      # erythroid progenitor cell
    }
    name_to_cl.update(composite_types)

    # Add developmental relationships
    developmental_stages = {
        'precursor': ['progenitor', 'immature', 'developing'],
        'progenitor': ['precursor', 'immature', 'developing'],
        'immature': ['precursor', 'progenitor', 'developing'],
        'mature': ['adult', 'differentiated'],
        'developing': ['precursor', 'progenitor', 'immature']
    }

    for node, data in graph.nodes(data=True):
        if not node.startswith('CL:'):
            continue

        # Add the main name
        if 'name' in data:
            name = data['name'].lower()
            name_to_cl[name] = node

            # Add variations of the main name
            variations = []

            # Handle common variations
            if 'positive' in name:
                variations.append(name.replace('positive', '+'))
            if 'negative' in name:
                variations.append(name.replace('negative', '-'))
            if ', alpha-beta' in name:
                variations.append(name.replace(', alpha-beta', ''))
                variations.append(name.replace(', alpha-beta', ' ab'))
            if 'thymus-derived' in name:
                variations.append(name.replace('thymus-derived ', ''))

            # Handle developmental stage variations
            for stage, variants in developmental_stages.items():
                if stage in name:
                    for variant in variants:
                        variations.append(name.replace(stage, variant))

            # Handle tissue-specific variations
            if 'fibroblast' in name:
                base_name = 'fibroblast'
                if 'of' in name or 'from' in name:
                    base_name = name
                variations.append(base_name)
                if 'thymic' in name:
                    variations.append('thymus fibroblast')

            # Handle erythroid variations
            if any(term in name for term in ['erythroid', 'erythrocyte']):
                if 'progenitor' in name:
                    variations.extend(['erythroid precursor', 'erythroid cell'])
                variations.extend(['rbc', 'red blood cell'])

            # Handle thymocyte variations
            if 'thymocyte' in name:
                if 'double-positive' in name:
                    variations.append('dp thymocyte')
                    variations.append('double positive thymocyte')
                if 'double negative' in name:
                    variations.append('dn thymocyte')
                    variations.append('double negative thymocyte')

            # Handle T cell variations
            if 'alpha-beta t cell' in name:
                variations.append(name.replace('alpha-beta t cell', 'ab t cell'))
                variations.append(name.replace('alpha-beta t cell', 't cell'))
            if 'gamma-delta t cell' in name:
                variations.append(name.replace('gamma-delta', 'gd'))
                variations.append('gamma delta t cell')

            for var in variations:
                name_to_cl[var] = node

        # Add synonyms
        if 'synonym' in data:
            synonyms = data['synonym']
            if isinstance(synonyms, str):
                synonyms = [synonyms]

            for syn in synonyms:
                # Extract the actual synonym text from the OBO format
                match = re.search(r'"([^"]+)"', syn)
                if match:
                    syn_text = match.group(1).lower()
                    name_to_cl[syn_text] = node

                    # Add variations of synonyms
                    if 'positive' in syn_text:
                        name_to_cl[syn_text.replace('positive', '+')] = node
                    if 'negative' in syn_text:
                        name_to_cl[syn_text.replace('negative', '-')] = node
                    if ', alpha-beta' in syn_text:
                        name_to_cl[syn_text.replace(', alpha-beta', '')] = node
                        name_to_cl[syn_text.replace(', alpha-beta', ' ab')] = node
                    if 'thymus-derived' in syn_text:
                        name_to_cl[syn_text.replace('thymus-derived ', '')] = node
                    if 'thymocyte' in syn_text and 'double' in syn_text:
                        name_to_cl[syn_text.replace('double-positive', 'dp')] = node
                        name_to_cl[syn_text.replace('double negative', 'dn')] = node

                    # Add developmental stage variations for synonyms
                    for stage, variants in developmental_stages.items():
                        if stage in syn_text:
                            for variant in variants:
                                name_to_cl[syn_text.replace(stage, variant)] = node

    return name_to_cl

def standardize_cell_type(name: str) -> str:
    """
    Standardize cell type names for better matching with Cell Ontology
    """
    name = name.lower()

    # Common replacements to match Cell Ontology format
    replacements = {
        'cd8+': 'cd8-positive',
        'cd4+': 'cd4-positive',
        'cd8-': 'cd8-negative',
        'cd4-': 'cd4-negative',
        ' ab ': ' alpha-beta ',
        ' αβ ': ' alpha-beta ',
        ' α-β ': ' alpha-beta ',
        ' a-b ': ' alpha-beta ',
        'nkt ': 'natural killer t ',
        'nk ': 'natural killer ',
        ' dc ': ' dendritic cell ',
        'tcell': 't cell',
        'bcell': 'b cell',
        ' t-cell': ' t cell',
        ' b-cell': ' b cell',
        'thymic epithelial': 'epithelial cell of thymus',
        'cortical thymocytes': 'double-positive, alpha-beta thymocyte',
        'double positive thymocyte': 'double-positive, alpha-beta thymocyte',
        'early t cells': 'double negative thymocyte',
        'early thymocytes': 'double negative thymocyte',
        'thymocytes': 'thymocyte',
        'macrophages': 'macrophage',
        'fibroblasts': 'fibroblast',
        'erythroid precursor': 'erythroid progenitor cell',
        'erythroid precursors': 'erythroid progenitor cell',
        ' cells': ' cell',
        '(': ' ',
        ')': ' ',
        '/': ' and '
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name

def find_best_match(cell_type: str, name_to_cl: Dict[str, str]) -> str:
    """
    Find the best matching Cell Ontology ID for a given cell type name
    """
    # Standardize input
    cell_type = standardize_cell_type(cell_type)

    # Try exact match first
    if cell_type in name_to_cl:
        return name_to_cl[cell_type]

    # Special handling for CD4/CD8 T cells
    if 'cd8' in cell_type or 'cd4' in cell_type:
        is_cd4 = 'cd4' in cell_type
        is_cd8 = 'cd8' in cell_type

        # Don't allow CD4 to match CD8 or vice versa
        if is_cd4:
            filtered_names = {name: cl_id for name, cl_id in name_to_cl.items()
                            if 'cd8' not in name or 'cd4' in name}
        elif is_cd8:
            filtered_names = {name: cl_id for name, cl_id in name_to_cl.items()
                            if 'cd4' not in name or 'cd8' in name}
        else:
            filtered_names = name_to_cl
    else:
        filtered_names = name_to_cl

    # Try matching by parts
    cell_type_parts = set(cell_type.split())

    best_match = None
    max_score = 0

    for name, cl_id in filtered_names.items():
        name_parts = set(name.split())

        # Calculate matching score
        common_parts = cell_type_parts & name_parts
        total_parts = cell_type_parts | name_parts

        # Base score using Jaccard similarity
        score = len(common_parts) / len(total_parts)

        # Bonus for matching important terms
        important_terms = {
            'cd4', 'cd8', 'positive', 'negative',
            'alpha-beta', 'gamma-delta', 'natural', 'killer',
            'thymocyte', 'double-positive', 'double negative',
            'cortical', 'medullary', 'thymic',
            'fibroblast', 'erythroid', 'erythrocyte',
            'progenitor', 'precursor', 'immature', 'mature',
            'naive', 'memory', 'regulatory', 'activated'
        }
        matching_important = len([w for w in common_parts if w in important_terms])
        score += matching_important * 0.1

        # Extra bonus for exact CD4/CD8 matches
        if ('cd4' in cell_type and 'cd4' in name) or ('cd8' in cell_type and 'cd8' in name):
            score += 0.3

        # Extra bonus for matching cell states
        cell_states = {
            'naive': 0.2,
            'memory': 0.2,
            'regulatory': 0.2,
            'activated': 0.2,
            'effector': 0.2
        }
        for state, bonus in cell_states.items():
            if state in cell_type and state in name:
                score += bonus

        # Extra bonus for specific contexts
        if 'thymocyte' in name and 'thymocyte' in cell_type:
            score += 0.2
        if 'fibroblast' in name and 'fibroblast' in cell_type:
            score += 0.2
        if ('erythroid' in name or 'erythrocyte' in name) and ('erythroid' in cell_type or 'erythrocyte' in cell_type):
            score += 0.2

        # Penalty for mismatched cell states
        for state in cell_states:
            if (state in cell_type and state not in name) or (state not in cell_type and state in name):
                score -= 0.1

        if score > max_score:
            max_score = score
            best_match = cl_id

    # Lower threshold for matching
    return best_match if max_score > 0.4 else None

def main():
    # Load the Cell Ontology
    print("Loading Cell Ontology...")
    graph = obonet.read_obo('data/ontology/cl.obo')

    # Get cell type mappings from the ontology
    print("Extracting cell type mappings...")
    name_to_cl = get_cell_type_synonyms(graph)

    # Read the results
    print("Reading results file...")
    results_df = pd.read_csv('results/benchmark/reference_comparison/2_evaluation/Thymus_results.csv')

    # Clean cell type names
    print("Processing cell type names...")
    results_df['reference_name_clean'] = results_df['reference_name'].apply(lambda x: x.split(': ')[-1].strip())
    results_df['final_consensus_clean'] = results_df['final_consensus'].apply(lambda x: x.split(': ')[-1].strip())

    # Convert names to CL IDs
    print("Mapping cell types to ontology IDs...")
    results_df['reference_cl'] = results_df['reference_name_clean'].apply(lambda x: find_best_match(x, name_to_cl))
    results_df['consensus_cl'] = results_df['final_consensus_clean'].apply(lambda x: find_best_match(x, name_to_cl))

    # Initialize the evaluator
    evaluator = CellOntologyEvaluator('data/ontology/cl.obo')

    # Get valid pairs (where both reference and consensus have CL IDs)
    valid_pairs = results_df.dropna(subset=['reference_cl', 'consensus_cl'])

    print(f"\nTotal pairs: {len(results_df)}")
    print(f"Valid pairs with CL IDs: {len(valid_pairs)}")

    if len(valid_pairs) > 0:
        # Evaluate predictions
        results = evaluator.evaluate_predictions(
            valid_pairs['reference_cl'].tolist(),
            valid_pairs['consensus_cl'].tolist()
        )

        # Add original names to results
        results['reference_name'] = valid_pairs['reference_name_clean'].reset_index(drop=True)
        results['consensus_name'] = valid_pairs['final_consensus_clean'].reset_index(drop=True)

        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)

        # Print results
        print("\nDetailed Results:")
        print(results.to_string())

        print("\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        # Print unmapped cell types
        print("\nUnmapped Reference Cell Types:")
        unmapped_ref = results_df[results_df['reference_cl'].isna()]['reference_name_clean'].unique()
        for name in unmapped_ref:
            print(f"- {name}")

        print("\nUnmapped Consensus Cell Types:")
        unmapped_cons = results_df[results_df['consensus_cl'].isna()]['final_consensus_clean'].unique()
        for name in unmapped_cons:
            print(f"- {name}")

        # Save results
        results.to_csv('results/benchmark/reference_comparison/2_evaluation/thymus_ontology_evaluation.csv', index=False)

        # Save mapping information for review
        mapping_df = pd.DataFrame({
            'original_name': list(results_df['reference_name_clean'].unique()) + list(results_df['final_consensus_clean'].unique()),
            'mapped_cl_id': [find_best_match(name, name_to_cl) for name in
                           list(results_df['reference_name_clean'].unique()) + list(results_df['final_consensus_clean'].unique())]
        }).drop_duplicates()

        mapping_df.to_csv('results/benchmark/reference_comparison/2_evaluation/cell_type_mapping.csv', index=False)

if __name__ == "__main__":
    main()
