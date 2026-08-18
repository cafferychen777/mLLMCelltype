#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Set
import obonet  # For reading OBO files

class CellOntologyEvaluator:
    def __init__(self, obo_file_path: str):
        """
        Initialize the evaluator with a Cell Ontology OBO file

        Args:
            obo_file_path: Path to the Cell Ontology OBO file
        """
        # Load the ontology graph
        self.graph = obonet.read_obo(obo_file_path)
        self.graph = nx.DiGraph(self.graph)  # Convert to directed graph

        # Create reverse graph for finding children
        self.reverse_graph = self.graph.reverse()

    def get_parent_terms(self, term_id: str) -> Set[str]:
        """Get immediate parent terms of a given term"""
        try:
            return set(self.graph.predecessors(term_id))
        except nx.NetworkXError:
            return set()

    def get_child_terms(self, term_id: str) -> Set[str]:
        """Get immediate child terms of a given term"""
        try:
            return set(self.reverse_graph.predecessors(term_id))
        except nx.NetworkXError:
            return set()

    def get_sibling_terms(self, term_id: str) -> Set[str]:
        """Get sibling terms (terms sharing the same parent) of a given term"""
        siblings = set()
        for parent in self.get_parent_terms(term_id):
            siblings.update(self.get_child_terms(parent))
        if term_id in siblings:
            siblings.remove(term_id)
        return siblings

    def get_term_depth(self, term_id: str) -> int:
        """Get the depth of a term in the ontology tree"""
        try:
            paths = nx.single_source_shortest_path_length(self.graph, term_id)
            return max(paths.values())
        except nx.NetworkXError:
            return 0

    def evaluate_prediction(self, true_term: str, predicted_term: str) -> Dict[str, bool]:
        """
        Evaluate a prediction against the true term

        Args:
            true_term: The true Cell Ontology term ID
            predicted_term: The predicted Cell Ontology term ID

        Returns:
            Dictionary containing match types (exact, parent, child, sibling)
        """
        results = {
            'exact_match': False,
            'parent_match': False,
            'child_match': False,
            'sibling_match': False
        }

        if true_term == predicted_term:
            results['exact_match'] = True
            return results

        # Check parent match
        if predicted_term in self.get_parent_terms(true_term):
            results['parent_match'] = True

        # Check child match
        if predicted_term in self.get_child_terms(true_term):
            results['child_match'] = True

        # Check sibling match
        if predicted_term in self.get_sibling_terms(true_term):
            results['sibling_match'] = True

        return results

    def evaluate_predictions(self, true_terms: List[str], predicted_terms: List[str]) -> pd.DataFrame:
        """
        Evaluate a list of predictions

        Args:
            true_terms: List of true Cell Ontology term IDs
            predicted_terms: List of predicted Cell Ontology term IDs

        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        for true_term, pred_term in zip(true_terms, predicted_terms):
            eval_result = self.evaluate_prediction(true_term, pred_term)
            eval_result.update({
                'true_term': true_term,
                'predicted_term': pred_term,
                'true_depth': self.get_term_depth(true_term),
                'pred_depth': self.get_term_depth(pred_term)
            })
            results.append(eval_result)

        return pd.DataFrame(results)

    def calculate_metrics(self, evaluation_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall metrics from evaluation results

        Args:
            evaluation_df: DataFrame from evaluate_predictions

        Returns:
            Dictionary containing various metrics
        """
        total = len(evaluation_df)
        metrics = {
            'exact_match_rate': evaluation_df['exact_match'].mean(),
            'parent_match_rate': evaluation_df['parent_match'].mean(),
            'child_match_rate': evaluation_df['child_match'].mean(),
            'sibling_match_rate': evaluation_df['sibling_match'].mean(),
            'total_predictions': total
        }

        return metrics

# Example usage
if __name__ == "__main__":
    # Example code to demonstrate usage
    obo_file = "path_to_cell_ontology.obo"  # Replace with actual path
    evaluator = CellOntologyEvaluator(obo_file)

    # Example data
    true_terms = ["CL:0000084", "CL:0000236"]  # Example cell ontology IDs
    pred_terms = ["CL:0000084", "CL:0000625"]  # Example predictions

    # Evaluate predictions
    results_df = evaluator.evaluate_predictions(true_terms, pred_terms)
    metrics = evaluator.calculate_metrics(results_df)

    print("Evaluation Results:")
    print(results_df)
    print("\nOverall Metrics:")
    print(metrics)
