#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for prompt generation functionality in mLLMCelltype.
"""

import os
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from mllmcelltype.prompts import create_batch_prompt
from mllmcelltype.prompts import create_consensus_check_prompt
from mllmcelltype.prompts import create_discussion_prompt
from mllmcelltype.prompts import create_json_prompt
from mllmcelltype.prompts import create_prompt


class TestPrompts:
    """Test class for prompt generation functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_dict, sample_marker_genes_list):
        """Set up test fixtures."""
        self.marker_genes = sample_marker_genes_dict
        self.marker_genes_list = sample_marker_genes_list

    def test_create_prompt_basic(self):
        """Test basic prompt creation."""
        prompt = create_prompt(
            marker_genes=self.marker_genes, species="human", tissue="blood"
        )

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "human" in prompt
        assert "blood" in prompt
        assert "CD3D" in prompt
        assert "CD19" in prompt
        assert "Cluster 1:" in prompt
        assert "Cluster 2:" in prompt

    def test_create_prompt_with_additional_context(self):
        """Test prompt creation with additional context."""
        additional_context = "Sample from a healthy donor."
        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human",
            tissue="blood",
            additional_context=additional_context,
        )

        # Check that the prompt contains the additional context
        assert additional_context in prompt

    def test_create_prompt_with_custom_template(self):
        """Test prompt creation with custom template."""
        custom_template = """Custom template for {species} cells from {tissue}.
Marker genes: {markers}"""

        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human",
            tissue="blood",
            prompt_template=custom_template,
        )

        # Check that the custom template was used
        assert "Custom template for human cells from blood" in prompt

    def test_create_batch_prompt(self):
        """Test batch prompt creation."""
        prompt = create_batch_prompt(
            marker_genes_list=self.marker_genes_list, species="human", tissue="blood"
        )

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "human" in prompt
        assert "blood" in prompt
        assert "Set 1:" in prompt or "Dataset 1:" in prompt
        assert "Set 2:" in prompt or "Dataset 2:" in prompt

        # Check if the prompt contains necessary guidance information, rather than specific genes
        # Current implementation does not include specific marker genes
        assert "cell type" in prompt.lower()
        assert "cluster" in prompt.lower()

    def test_create_json_prompt(self):
        """Test JSON prompt creation."""
        prompt = create_json_prompt(
            marker_genes=self.marker_genes, species="human", tissue="blood"
        )

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "human" in prompt
        assert "blood" in prompt
        assert "CD3D" in prompt
        assert "JSON" in prompt.upper() or "json" in prompt
        assert "cluster" in prompt
        assert "cell_type" in prompt

    def test_create_discussion_prompt(self):
        """Test discussion prompt creation."""
        cluster_id = "1"
        marker_genes = ["CD3D", "CD3E", "CD2"]
        model_votes = {"model1": "T cells", "model2": "T cells", "model3": "NK cells"}

        prompt = create_discussion_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            model_votes=model_votes,
            species="human",
            tissue="blood",
        )

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "human" in prompt
        assert "blood" in prompt
        assert "CD3D" in prompt
        assert "T cells" in prompt
        assert "NK cells" in prompt
        assert (
            "Cluster 1" in prompt or "cluster 1" in prompt or "Cluster ID: 1" in prompt
        )

    def test_create_consensus_check_prompt(self):
        """Test consensus check prompt creation."""
        annotations = ["T cells", "T cells", "NK cells"]
        prompt = create_consensus_check_prompt(annotations)

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "T cells" in prompt
        assert "NK cells" in prompt
        assert "consensus" in prompt.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
