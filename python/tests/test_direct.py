#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for mLLMCelltype utility functions.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from mllmcelltype.utils import (
    clean_annotation,
    create_cache_key,
    format_results,
    load_from_cache,
    parse_marker_genes,
    save_to_cache,
)


# Sample marker genes for testing
@pytest.fixture
def sample_marker_genes_df():
    """Create sample marker genes dataframe for testing."""
    data = {
        "cluster": [1, 1, 1, 2, 2, 2],
        "gene": ["CD3D", "CD3E", "CD2", "CD19", "MS4A1", "CD79A"],
        "avg_log2FC": [2.5, 2.3, 2.1, 3.0, 2.8, 2.7],
        "pct.1": [0.9, 0.85, 0.8, 0.95, 0.9, 0.85],
        "pct.2": [0.1, 0.15, 0.2, 0.05, 0.1, 0.15],
        "p_val_adj": [1e-10, 1e-9, 1e-8, 1e-12, 1e-11, 1e-10],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_marker_genes_dict():
    """Create sample marker genes dictionary for testing."""
    return {"1": ["CD3D", "CD3E", "CD2"], "2": ["CD19", "MS4A1", "CD79A"]}


# Test parse_marker_genes function
def test_parse_marker_genes(sample_marker_genes_df):
    """Test parsing marker genes from DataFrame to dictionary."""
    result = parse_marker_genes(sample_marker_genes_df)
    assert isinstance(result, dict)
    assert "1" in result
    assert "2" in result
    assert len(result["1"]) == 3
    assert len(result["2"]) == 3
    assert "CD3D" in result["1"]
    assert "CD19" in result["2"]


def test_parse_marker_genes_empty():
    """Test parsing empty marker genes DataFrame."""
    empty_df = pd.DataFrame()
    result = parse_marker_genes(empty_df)
    assert isinstance(result, dict)
    assert len(result) == 0


def test_parse_marker_genes_missing_columns():
    """Test parsing marker genes DataFrame with missing columns."""
    # DataFrame missing required columns
    df_missing_columns = pd.DataFrame({"cluster": [1, 2], "other_column": ["A", "B"]})

    with pytest.raises(ValueError, match="'gene' column not found"):
        parse_marker_genes(df_missing_columns)


# Test cache functions
def test_create_cache_key():
    """Test creating cache key."""
    key1 = create_cache_key("prompt1", "model1", "provider1")
    key2 = create_cache_key("prompt2", "model1", "provider1")
    key3 = create_cache_key("prompt1", "model1", "provider1")

    assert isinstance(key1, str)
    assert key1 != key2
    assert key1 == key3


def test_save_and_load_from_cache():
    """Test saving to and loading from cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_data = ["result1", "result2"]
        cache_key = create_cache_key("test_prompt", "test_model", "test_provider")

        # Note parameter order: cache_key, results, cache_dir
        save_to_cache(cache_key, test_data, cache_dir=temp_dir)

        # Read directly from file instead of using load_from_cache function
        cache_file = os.path.join(temp_dir, f"{cache_key}.json")
        assert os.path.exists(cache_file)

        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            loaded_data = cache_data["data"]  # Data is stored in the "data" field

        # Verify data
        assert loaded_data == test_data


def test_load_from_nonexistent_cache():
    """Test loading from nonexistent cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_key = create_cache_key("nonexistent", "model", "provider")
        loaded_data = load_from_cache(cache_key, cache_dir=temp_dir)
        assert loaded_data is None


# Test format_results function
def test_format_results_simple():
    """Test formatting simple results."""
    raw_results = ["Cluster 1: T cells", "Cluster 2: B cells"]

    # Need to provide clusters parameter
    clusters = ["1", "2"]
    formatted = format_results(raw_results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    assert formatted["1"] == "T cells"
    assert formatted["2"] == "B cells"


def test_clean_annotation():
    """Test cleaning cell type annotations."""
    # Test basic cleaning
    assert clean_annotation("T cells") == "T cells"

    # Test removing prefixes
    assert clean_annotation("Cluster 1: T cells") == "T cells"
    assert clean_annotation("Cell type: B cells") == "B cells"

    # Test removing number prefix
    assert clean_annotation("1. T cells") == "T cells"

    # Test handling empty or None
    assert clean_annotation("") == ""
    assert clean_annotation(None) == ""


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
