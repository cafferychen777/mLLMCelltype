import json
import os
import re
from collections import defaultdict
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def query_ols(term):
    """Query a cell type through the OLS API."""
    base_url = "https://www.ebi.ac.uk/ols4/api/search"
    params = {
        "q": term,
        "ontology": "cl",
        "exact": "true",
        "queryFields": "label,synonym",
        "rows": 5,
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["response"]["numFound"] > 0:
                # Return the best match.
                best_match = data["response"]["docs"][0]
                return {
                    "id": best_match.get("obo_id"),
                    "label": best_match.get("label"),
                    "iri": best_match.get("iri"),
                    "score": best_match.get("score"),
                }
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error querying OLS: {e}")

    return None


def get_ancestors(term_iri, depth=2):
    """Get ancestor nodes for a term up to the requested depth."""
    if not term_iri:
        return set()

    ancestors = set()
    current_depth = 0
    current_terms = {term_iri}

    while current_depth < depth and current_terms:
        next_terms = set()
        for term in current_terms:
            try:
                encoded_iri = requests.utils.quote(term, safe="")
                response = requests.get(
                    f"https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms/{encoded_iri}/ancestors"
                )
                if response.status_code == 200:
                    data = response.json()
                    if "_embedded" in data and "terms" in data["_embedded"]:
                        for ancestor in data["_embedded"]["terms"]:
                            ancestors.add(ancestor["obo_id"])
                            next_terms.add(ancestor["iri"])
            except (requests.RequestException, KeyError, ValueError) as e:
                print(f"Error getting ancestors: {e}")

        current_terms = next_terms
        current_depth += 1

    return ancestors


def get_descendants(term_iri, depth=2):
    """Get descendant nodes for a term up to the requested depth."""
    if not term_iri:
        return set()

    descendants = set()
    current_depth = 0
    current_terms = {term_iri}

    while current_depth < depth and current_terms:
        next_terms = set()
        for term in current_terms:
            try:
                encoded_iri = requests.utils.quote(term, safe="")
                response = requests.get(
                    f"https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms/{encoded_iri}/descendants"
                )
                if response.status_code == 200:
                    data = response.json()
                    if "_embedded" in data and "terms" in data["_embedded"]:
                        for descendant in data["_embedded"]["terms"]:
                            descendants.add(descendant["obo_id"])
                            next_terms.add(descendant["iri"])
            except (requests.RequestException, KeyError, ValueError) as e:
                print(f"Error getting descendants: {e}")

        current_terms = next_terms
        current_depth += 1

    return descendants


def get_extended_siblings(term_iri):
    """Get extended siblings that share an ancestor within two steps."""
    if not term_iri:
        return set()

    # Get all ancestors within two steps.
    ancestors = get_ancestors(term_iri, depth=2)
    siblings = set()

    # Get the direct children of every ancestor.
    for ancestor_id in ancestors:
        ancestor_iri = f"http://purl.obolibrary.org/obo/{ancestor_id.replace(':', '_')}"
        try:
            encoded_iri = requests.utils.quote(ancestor_iri, safe="")
            response = requests.get(
                f"https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms/{encoded_iri}/children"
            )
            if response.status_code == 200:
                data = response.json()
                if "_embedded" in data and "terms" in data["_embedded"]:
                    for sibling in data["_embedded"]["terms"]:
                        siblings.add(sibling["obo_id"])
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"Error getting siblings: {e}")

    return siblings


def normalize_cell_type(cell_type):
    """Normalize a cell type name and handle common variants."""
    # Remove special characters and redundant whitespace.
    cell_type = re.sub(r"[^\w\s-]", " ", cell_type)
    cell_type = " ".join(cell_type.split())

    # Normalize common abbreviations and variants.
    replacements = {
        "cd4": "CD4",
        "cd8": "CD8",
        "tcell": "T cell",
        "bcell": "B cell",
        "nk cell": "natural killer cell",
        "dc": "dendritic cell",
        "tec": "thymic epithelial cell",
        "mtec": "medullary thymic epithelial cell",
        "ctec": "cortical thymic epithelial cell",
    }

    for old, new in replacements.items():
        cell_type = cell_type.replace(old, new)

    return cell_type


def build_ontology_mappings(unique_terms):
    """Build parent, child, and sibling mappings for unique cell types."""
    mappings = {}
    print(f"Building ontology mappings for {len(unique_terms)} unique terms...")

    for term in tqdm(unique_terms):
        normalized_term = normalize_cell_type(term)
        term_info = query_ols(normalized_term)

        if term_info and term_info["iri"]:
            term_iri = term_info["iri"]
            ancestors = get_ancestors(term_iri, depth=2)
            descendants = get_descendants(term_iri, depth=2)
            extended_siblings = get_extended_siblings(term_iri)

            print(f"\nTerm: {term}")
            print(f"IRI: {term_iri}")
            print(f"ID: {term_info['id']}")
            print(f"Ancestors: {ancestors}")
            print(f"Descendants: {descendants}")
            print(f"Extended Siblings: {extended_siblings}")

            mappings[term] = {
                "iri": term_iri,
                "id": term_info["id"],
                "ancestors": ancestors,
                "descendants": descendants,
                "extended_siblings": extended_siblings,
            }
            # Add delay to avoid overwhelming the API
            sleep(0.5)
        else:
            print(f"\nTerm not found: {term}")
            mappings[term] = {
                "iri": None,
                "id": None,
                "ancestors": set(),
                "descendants": set(),
                "extended_siblings": set(),
            }

    return mappings


def evaluate_matches(predictions_file, output_dir="results/evaluation"):
    """Evaluate ontology-aware matches for a prediction file."""
    print("Loading predictions...")
    df = pd.read_csv(predictions_file)

    # Collect unique cell type annotations.
    unique_true_types = df["original_celltype"].unique()
    unique_pred_types = df["popv_majority_vote_prediction"].unique()
    unique_types = np.union1d(unique_true_types, unique_pred_types)

    print(f"Found {len(unique_types)} unique cell type annotations")

    # Build ontology mappings.
    ontology_mappings = build_ontology_mappings(unique_types)

    # Save ontology mappings for diagnostics.
    with open(os.path.join(output_dir, "ontology_mappings.json"), "w") as f:
        json.dump(
            {
                k: {
                    "iri": v["iri"],
                    "id": v["id"],
                    "ancestors": list(v["ancestors"]),
                    "descendants": list(v["descendants"]),
                    "extended_siblings": list(v["extended_siblings"]),
                }
                for k, v in ontology_mappings.items()
            },
            f,
            indent=2,
        )

    # Initialize result counts.
    results = defaultdict(int)
    detailed_results = []

    print("Evaluating matches using pre-built mappings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        true_type = row["original_celltype"]
        pred_type = row["popv_majority_vote_prediction"]

        true_mapping = ontology_mappings[true_type]
        pred_mapping = ontology_mappings[pred_type]

        # Evaluate the match type.
        match_type = "No Match"
        match_details = {}

        if true_type == pred_type or (
            true_mapping["id"]
            and pred_mapping["id"]
            and true_mapping["id"] == pred_mapping["id"]
        ):
            match_type = "Exact Match"
        elif true_mapping["id"] and pred_mapping["id"]:
            # Check parent-child relationships within two steps.
            true_ancestors = {
                term.split("/")[-1].replace("_", ":")
                for term in true_mapping["ancestors"]
            }
            true_descendants = {
                term.split("/")[-1].replace("_", ":")
                for term in true_mapping["descendants"]
            }
            true_siblings = {
                term.split("/")[-1].replace("_", ":")
                for term in true_mapping["extended_siblings"]
            }

            if pred_mapping["id"] in true_ancestors:
                match_type = "Parent Match"
                match_details["distance"] = 1  # TODO: calculate the actual distance
            elif pred_mapping["id"] in true_descendants:
                match_type = "Child Match"
                match_details["distance"] = 1  # TODO: calculate the actual distance
            elif pred_mapping["id"] in true_siblings:
                match_type = "Sibling Match"

        results[match_type] += 1
        detailed_results.append(
            {
                "true_type": true_type,
                "predicted_type": pred_type,
                "match_type": match_type,
                "true_id": true_mapping["id"],
                "pred_id": pred_mapping["id"],
                **match_details,
            }
        )

    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results.
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(os.path.join(output_dir, "detailed_matches.csv"), index=False)

    # Save the summary.
    total = sum(results.values())
    summary = {
        k: {"count": v, "percentage": (v / total) * 100} for k, v in results.items()
    }
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(output_dir, "summary_matches.csv"))

    # Plot the match-type distribution.
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [v / total * 100 for v in results.values()])
    plt.title("Distribution of Cell Type Matches")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "match_distribution.png"))

    # Analyze mismatch patterns.
    print("\nDetailed Match Statistics:")
    for match_type, count in results.items():
        percentage = (count / total) * 100
        print(f"{match_type}: {count} ({percentage:.2f}%)")

    print("\nTop mismatched cell types:")
    mismatches = detailed_df[detailed_df["match_type"] == "No Match"]
    print(mismatches.groupby(["true_type", "predicted_type"]).size().nlargest(10))

    return results, detailed_results


if __name__ == "__main__":
    # Evaluate the latest prediction results.
    predictions_file = "results/thymus_annotation/cell_type_predictions.csv"
    results, detailed_results = evaluate_matches(predictions_file)

    print("\nMatch Statistics:")
    print(results)
