#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic tests for LLMCelltype.
Run with pytest: pytest -xvs tests/test_basic.py
"""

import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import mllmcelltype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mllmcelltype import annotate_clusters, batch_annotate_clusters
from mllmcelltype.utils import load_api_key, parse_marker_genes
from mllmcelltype.prompts import create_annotation_prompt

# Sample marker genes for testing
@pytest.fixture
def sample_marker_genes():
    """Create sample marker genes dataframe for testing."""
    data = {
        "cluster": [1, 1, 1, 2, 2, 2],
        "gene": ["CD3D", "CD3E", "CD2", "CD19", "MS4A1", "CD79A"],
        "avg_log2FC": [2.5, 2.3, 2.1, 3.0, 2.8, 2.7],
        "pct.1": [0.9, 0.85, 0.8, 0.95, 0.9, 0.85],
        "pct.2": [0.1, 0.15, 0.2, 0.05, 0.1, 0.15],
        "p_val_adj": [1e-10, 1e-9, 1e-8, 1e-12, 1e-11, 1e-10]
    }
    return pd.DataFrame(data)

# Test utility functions
def test_parse_marker_genes(sample_marker_genes):
    """Test parsing marker genes."""
    parsed = parse_marker_genes(sample_marker_genes)
    
    assert isinstance(parsed, dict)
    assert "1" in parsed
    assert "2" in parsed
    assert len(parsed["1"]) == 3
    assert len(parsed["2"]) == 3
    assert "CD3D" in parsed["1"]
    assert "CD19" in parsed["2"]

def test_create_annotation_prompt(sample_marker_genes):
    """Test creating annotation prompt."""
    prompt = create_annotation_prompt(
        marker_genes=sample_marker_genes,
        species="human",
        tissue="blood"
    )
    
    assert isinstance(prompt, str)
    assert "human" in prompt
    assert "blood" in prompt
    assert "CD3D" in prompt
    assert "CD19" in prompt

# Test API key loading
def test_load_api_key():
    """Test loading API key."""
    # Test with environment variable
    with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}):
        # Create a temporary mapping for the test provider
        with patch('mllmcelltype.utils.env_var_map', {'test': 'TEST_API_KEY'}):
            key = load_api_key("test")
            assert key == "test-key-123"
    
    # Test with missing key
    key = load_api_key("nonexistent")
    assert key is None or key == ""

# Mock LLM provider for testing
@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    mock = MagicMock()
    mock.return_value = {
        "1": "T cells",
        "2": "B cells"
    }
    return mock

# Test annotation function
def test_annotate_clusters(sample_marker_genes, mock_provider):
    """Test annotate_clusters function."""
    with patch("mllmcelltype.annotate.get_provider_function", return_value=mock_provider):
        result = annotate_clusters(
            marker_genes=sample_marker_genes,
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood"
        )
        
        assert isinstance(result, dict)
        assert "1" in result
        assert "2" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"
        
        # Check that provider was called with correct arguments
        mock_provider.assert_called_once()
        args, kwargs = mock_provider.call_args
        assert "prompt" in kwargs
        assert "model" in kwargs
        assert kwargs["model"] == "mock_model"

# Test batch annotation
def test_batch_annotate_clusters(sample_marker_genes, mock_provider):
    """Test batch_annotate_clusters function."""
    with patch("mllmcelltype.annotate.get_provider_function", return_value=mock_provider):
        result = batch_annotate_clusters(
            marker_genes_list=[sample_marker_genes, sample_marker_genes],
            species=["human", "human"],
            provider="mock_provider",
            model="mock_model",
            tissue=["blood", "blood"]
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        for annotations in result:
            assert isinstance(annotations, dict)
            assert "1" in annotations
            assert "2" in annotations
            assert annotations["1"] == "T cells"
            assert annotations["2"] == "B cells"
        
        # Check that provider was called twice
        assert mock_provider.call_count == 2

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
