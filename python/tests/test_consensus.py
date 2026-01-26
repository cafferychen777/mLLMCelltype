#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for consensus and comparison functionality in mLLMCelltype.
"""

from unittest.mock import patch

import pytest

from mllmcelltype.consensus import (
    check_consensus,
    interactive_consensus_annotation,
)


class TestConsensus:
    """Test class for consensus functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_df, sample_marker_genes_dict):
        """Set up test fixtures."""
        self.marker_genes_df = sample_marker_genes_df
        self.marker_genes_dict = sample_marker_genes_dict

        # Sample annotations from different models
        self.model_annotations = {
            "gpt-5": {
                "1": "T cells",
                "2": "B cells",
                "3": "NK cells",
            },
            "claude-sonnet-4-5-20250929": {
                "1": "T lymphocytes",
                "2": "B lymphocytes",
                "3": "Natural killer cells",
            },
            "gemini-3-pro": {
                "1": "CD4+ T cells",
                "2": "Plasma B cells",
                "3": "NK cells",
            },
        }

    def test_check_consensus_basic(self):
        """Test basic check_consensus function behavior."""
        # Create a simple prediction dictionary where all models have the same prediction for cluster 3
        simple_predictions = {
            "model1": {"3": "NK cells"},
            "model2": {"3": "NK cells"},
            "model3": {"3": "NK cells"},
        }

        # Test function without LLM (should fall back to simple consensus)
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            predictions=simple_predictions,
            api_keys={},  # No API keys to force fallback
        )

        # Verify results
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert isinstance(controversial, list)
        assert "3" in consensus
        assert consensus["3"] == "NK cells"
        assert consensus_proportion["3"] == 1.0  # Complete agreement, should be 1.0
        assert entropy["3"] == 0.0  # Complete agreement, entropy should be 0

    def test_check_consensus_fallback(self):
        """Test check_consensus function fallback behavior."""
        # Test with disagreement to trigger fallback logic
        disagreement_predictions = {
            "model1": {"1": "T cells", "2": "B cells"},
            "model2": {"1": "T lymphocytes", "2": "Plasma cells"},
            "model3": {"1": "CD4+ T cells", "2": "Memory B cells"},
        }

        # Test function (should use fallback consensus since no working API keys)
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            predictions=disagreement_predictions,
            consensus_threshold=0.7,
            entropy_threshold=0.6,
            api_keys={},  # No API keys to force fallback
        )

        # Verify results
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert isinstance(controversial, list)
        assert "1" in consensus
        assert "2" in consensus
        # Should have some consensus values
        assert len(consensus) == 2
        assert len(controversial) >= 0  # May or may not have controversial clusters

    @patch("mllmcelltype.functions.get_provider")
    @patch("mllmcelltype.annotate.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
        mock_get_provider,
    ):
        """Test interactive_consensus_annotation function."""
        # Setup mocks
        mock_get_provider.return_value = "openai"  # Ensure get_provider returns a valid provider
        mock_annotate_clusters.side_effect = [
            {"1": "T cells", "2": "B cells", "3": "NK cells"},
            {"1": "T lymphocytes", "2": "B lymphocytes", "3": "Natural killer cells"},
            {"1": "CD4+ T cells", "2": "Plasma B cells", "3": "NK cells"},
        ]
        mock_check_consensus.return_value = (
            {
                "1": "T cells",
                "3": "NK cells",
            },  # Consensus for non-controversial clusters
            {"1": 0.85, "2": 0.60, "3": 0.95},  # Consensus proportions
            {"1": 0.45, "2": 0.70, "3": 0.20},  # Entropy values
            ["2"],  # Controversial clusters
        )
        mock_process_controversial.return_value = (
            {"2": "B cells"},  # Resolved annotations
            {"2": ["Discussion round 1", "Discussion round 2"]},  # Discussion history
            {"2": 0.85},  # Updated consensus proportions
            {"2": 0.40},  # Updated entropy values
        )

        # Test function
        result = interactive_consensus_annotation(
            marker_genes=self.marker_genes_dict,
            species="human",
            models=[
                "gpt-5",
                "claude-sonnet-4-5-20250929",
                "gemini-3-pro",
            ],
            api_keys={
                "openai": "test-key",
                "anthropic": "test-key",
                "gemini": "test-key",
            },
            tissue="blood",
            consensus_threshold=0.7,
            entropy_threshold=0.6,
            max_discussion_rounds=2,
            use_cache=False,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "consensus" in result
        assert "consensus_proportion" in result
        assert "entropy" in result
        assert "model_annotations" in result
        assert "controversial_clusters" in result
        assert (
            "discussion_logs" in result
        )  # Note: field name is discussion_logs not discussion_history
        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "B cells"
        assert result["consensus"]["3"] == "NK cells"
        assert result["controversial_clusters"] == ["2"]
        assert len(result["model_annotations"]) == 3
        assert result["consensus_proportion"]["2"] == 0.85
        assert result["entropy"]["2"] == 0.40


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
