#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core functionality tests for mLLMCelltype.
Tests for utility functions and core features that don't depend on external APIs.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

# Import utility functions
from mllmcelltype.utils import (
    clean_annotation,
    create_cache_key,
    format_results,
    load_api_key,
    load_from_cache,
    parse_marker_genes,
    save_to_cache,
)


# Test parse_marker_genes function
def test_parse_marker_genes(sample_marker_genes_df):
    """Test parsing marker genes from DataFrame to dictionary."""
    parsed = parse_marker_genes(sample_marker_genes_df)

    assert isinstance(parsed, dict)
    assert "1" in parsed
    assert "2" in parsed
    assert len(parsed["1"]) == 3
    assert len(parsed["2"]) == 3
    assert "CD3D" in parsed["1"]
    assert "CD19" in parsed["2"]


def test_parse_marker_genes_empty():
    """Test parsing empty marker genes DataFrame."""
    empty_df = pd.DataFrame()
    parsed = parse_marker_genes(empty_df)

    assert isinstance(parsed, dict)
    assert len(parsed) == 0


def test_parse_marker_genes_missing_columns():
    """Test parsing marker genes DataFrame with missing columns."""
    # Missing 'gene' column
    df = pd.DataFrame({"cluster": [1, 2, 3]})

    with pytest.raises(ValueError, match="'gene' column not found"):
        parse_marker_genes(df)

    # Missing 'cluster' column
    df = pd.DataFrame({"gene": ["CD3D", "CD19"]})

    with pytest.raises(ValueError, match="'cluster' column not found"):
        parse_marker_genes(df)


# Test load_api_key function
def test_load_api_key_from_env(mock_env_with_api_keys):
    """Test loading API key from environment variables."""
    key = load_api_key("openai")
    assert key == os.environ["OPENAI_API_KEY"]

    key = load_api_key("anthropic")
    assert key == os.environ["ANTHROPIC_API_KEY"]


def test_load_api_key_missing():
    """Test loading missing API key."""
    # Ensure the environment variable doesn't exist
    with patch.dict(os.environ, {}, clear=True):
        key = load_api_key("nonexistent")
        assert key is None or key == ""


def test_load_api_key_unknown_provider():
    """Test loading API key for unknown provider."""
    with patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key-456"}):
        key = load_api_key("custom")
        # For unknown providers, it tries to find PROVIDER_API_KEY in environment
        assert key == "test-key-456"


# Test cache functions
def test_create_cache_key():
    """Test creating cache key."""
    key1 = create_cache_key("test prompt", "gpt-4", "openai")
    key2 = create_cache_key("test prompt", "gpt-4", "openai")
    key3 = create_cache_key("different prompt", "gpt-4", "openai")

    assert key1 == key2  # Same inputs should produce same key
    assert key1 != key3  # Different inputs should produce different keys
    assert isinstance(key1, str)
    assert len(key1) > 0


def test_save_and_load_from_cache():
    """Test saving to and loading from cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with dictionary
        data_dict = {"1": "T cells", "2": "B cells"}
        key = create_cache_key("test prompt", "gpt-4", "openai")

        save_to_cache(key, data_dict, cache_dir=temp_dir)
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded == data_dict

        # Test with list
        data_list = ["T cells", "B cells"]
        key = create_cache_key("test prompt 2", "gpt-4", "openai")

        save_to_cache(key, data_list, cache_dir=temp_dir)
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded == data_list


def test_load_from_nonexistent_cache():
    """Test loading from nonexistent cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        key = create_cache_key("nonexistent", "gpt-4", "openai")
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded is None


# Test format_results function
def test_format_results_simple():
    """Test formatting simple results."""
    results = ["Cluster 1: T cells", "Cluster 2: B cells"]
    clusters = ["1", "2"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    assert formatted["1"] == "T cells"
    assert formatted["2"] == "B cells"


def test_format_results_complex():
    """Test formatting complex results with different formats."""
    results = ["1. T cells", "Cluster 2 - B cells", "3: NK cells"]
    clusters = ["1", "2", "3"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    assert "3" in formatted
    # In simple mapping mode, format_results doesn't clean prefixes
    assert "1. T cells" in formatted["1"]
    assert "Cluster 2 - B cells" in formatted["2"]
    assert "3: NK cells" in formatted["3"]


def test_format_results_mismatched():
    """Test formatting results with mismatched clusters."""
    results = ["Cluster 1: T cells", "Cluster 2: B cells"]
    clusters = ["1", "2", "3"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    # The function adds "Unknown" for missing clusters
    assert "3" in formatted
    # In simple mapping mode, it doesn't clean prefixes
    assert "Cluster 1: T cells" in formatted["1"]
    assert "Cluster 2: B cells" in formatted["2"]
    assert formatted["3"] == "Unknown"


# Test clean_annotation function
def test_clean_annotation():
    """Test cleaning cell type annotations."""
    test_cases = [
        ("T cells", "T cells"),
        # Current implementation does not replace hyphens
        ("T-cells", "T-cells"),
        ("T-cell", "T-cell"),
        ("CD4+ T cells", "CD4+ T cells"),
        ("B-cells (naive)", "B-cells (naive)"),
        # The following are cleaning operations supported by the current implementation
        ("Cluster 1: T cells", "T cells"),
        ("1. T cells", "T cells"),
        ("T cells.", "T cells"),
        ('"T cells"', "T cells"),
        ("T cells (CD4+)", "T cells (CD4+)"),
    ]

    for input_str, expected in test_cases:
        assert clean_annotation(input_str) == expected


# Test format_results with JSON responses
def test_format_results_json_with_markers():
    """Test parsing JSON response with code block markers."""
    json_response = [
        "```json",
        "{",
        '  "annotations": [',
        '    {"cluster": "1", "cell_type": "T cells"},',
        '    {"cluster": "2", "cell_type": "B cells"},',
        '    {"cluster": "3", "cell_type": "Monocytes"}',
        "  ]",
        "}",
        "```",
    ]
    clusters = ["1", "2", "3"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells", "3": "Monocytes"}


def test_format_results_json_without_markers():
    """Test parsing JSON response without code block markers."""
    json_response = [
        "{",
        '  "annotations": [',
        '    {"cluster": "1", "cell_type": "T cells"},',
        '    {"cluster": "2", "cell_type": "B cells"}',
        "  ]",
        "}",
    ]
    clusters = ["1", "2"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells"}


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
