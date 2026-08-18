"""Tests for supported marker-gene table layouts."""

import pandas as pd
import pytest

from app import convert_dataframe_to_marker_genes, create_results_dataframe


def test_wide_marker_table_ignores_scores_and_deduplicates() -> None:
    frame = pd.DataFrame(
        {
            "Cluster 0": ["CD3D", "CD3D", "1.25"],
            "Cluster 1": ["MS4A1", "CD79A", None],
        }
    )

    assert convert_dataframe_to_marker_genes(frame) == {
        "Cluster 0": ["CD3D"],
        "Cluster 1": ["MS4A1", "CD79A"],
    }


def test_long_marker_table_accumulates_repeated_clusters() -> None:
    frame = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1],
            "gene_symbol": ["CD3D", "CD3E", "MS4A1", "MS4A1"],
            "score": [5.2, 4.8, 7.1, 6.9],
        }
    )

    assert convert_dataframe_to_marker_genes(frame) == {
        "Cluster_0": ["CD3D", "CD3E"],
        "Cluster_1": ["MS4A1"],
    }


def test_long_marker_table_normalizes_cluster_header_whitespace() -> None:
    frame = pd.DataFrame(
        {
            " Cluster ": [0, 1],
            "gene": ["CD3D", "MS4A1"],
        }
    )

    assert convert_dataframe_to_marker_genes(frame) == {
        "Cluster_0": ["CD3D"],
        "Cluster_1": ["MS4A1"],
    }


def test_numeric_only_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="No gene symbols"):
        convert_dataframe_to_marker_genes(pd.DataFrame({"Cluster 0": [1.0, 2.0]}))


def test_results_dataframe_rejects_missing_consensus() -> None:
    with pytest.raises(ValueError, match="consensus"):
        create_results_dataframe({"entropy": {"0": 0.2}})


def test_results_dataframe_neutralizes_spreadsheet_formulas() -> None:
    frame = create_results_dataframe(
        {
            "consensus": {"=1+1": "+SUM(A1:A2)"},
            "consensus_proportion": {"=1+1": 0.8},
            "entropy": {"=1+1": 0.2},
        }
    )

    assert frame.iloc[0]["cluster"] == "'=1+1"
    assert frame.iloc[0]["cell_type"] == "'+SUM(A1:A2)"
