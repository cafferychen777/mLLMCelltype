#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for annotation functionality in mLLMCelltype.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.annotate import (
    annotate_clusters,
    get_model_response,
)


# Test annotation functions
class TestAnnotation:
    """Test class for annotation functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_df, sample_marker_genes_dict, mock_api_response):
        """Set up test fixtures."""
        self.marker_genes_df = sample_marker_genes_df
        self.marker_genes_dict = sample_marker_genes_dict
        self.mock_api_response = mock_api_response

    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS", {"mock_provider": MagicMock()})
    def test_annotate_clusters(self, mock_get_default_model, mock_load_api_key):
        """Test annotate_clusters function."""
        # Setup mocks
        from mllmcelltype.annotate import PROVIDER_FUNCTIONS

        # Return a list because format_results function expects a list
        PROVIDER_FUNCTIONS["mock_provider"] = lambda *args, **kwargs: [
            "Cluster 1: T cells",
            "Cluster 2: B cells",
        ]
        mock_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "mock_model"

        # Test with DataFrame input - disable cache
        result = annotate_clusters(
            marker_genes=self.marker_genes_df,
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood",
            use_cache=False,  # disable cache
        )

        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert "2" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"

        # Test with dictionary input - disable cache
        result = annotate_clusters(
            marker_genes=self.marker_genes_dict,
            species="human",
            provider="mock_provider",
            model="mock_model",
            tissue="blood",
            use_cache=False,  # disable cache
        )

        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert "2" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"

    # Fix parameter name issues
    @patch("mllmcelltype.annotate.PROVIDER_FUNCTIONS")
    @patch("mllmcelltype.utils.load_api_key")
    @patch("mllmcelltype.annotate.load_api_key")
    @patch("mllmcelltype.annotate.get_default_model")
    def test_get_model_response(
        self,
        mock_get_default_model,
        mock_load_api_key,
        mock_utils_load_api_key,
        mock_provider_functions,
    ):
        """Test get_model_response function."""

        # Setup mocks
        def mock_provider_func(*args, **kwargs):
            return [
                "Cluster 1: T cells",
                "Cluster 2: B cells",
            ]

        # Set up PROVIDER_FUNCTIONS dictionary
        mock_provider_functions.get.return_value = mock_provider_func
        mock_provider_functions.__contains__.return_value = True

        mock_load_api_key.return_value = "test-key"
        mock_utils_load_api_key.return_value = "test-key"
        mock_get_default_model.return_value = "gpt-4o"

        # Test function
        result = get_model_response(
            prompt="Test prompt",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",  # Explicitly provide API key to avoid loading real keys
            use_cache=False,
        )

        # Verify results
        assert isinstance(result, list) or isinstance(result, str)
        if isinstance(result, list):
            assert len(result) == 2
            assert "Cluster 1: T cells" in result[0]
            assert "Cluster 2: B cells" in result[1]
        else:
            assert "Cluster 1: T cells" in result
            assert "Cluster 2: B cells" in result

    # Fix API key issues - use patch.dict to ensure environment variables are properly mocked
    @patch.dict(os.environ, {}, clear=True)  # Clear all environment variables
    @patch("mllmcelltype.utils.load_api_key")
    @patch("mllmcelltype.annotate.load_api_key")
    def test_get_model_response_missing_api_key(self, mock_load_api_key, mock_utils_load_api_key):
        """Test get_model_response with missing API key."""
        # Ensure both load_api_key functions return None
        mock_load_api_key.return_value = None
        mock_utils_load_api_key.return_value = None

        # Test function with missing API key
        with pytest.raises(ValueError, match="API key not found"):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model="gpt-4o",
                api_key=None,
                use_cache=False,
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
