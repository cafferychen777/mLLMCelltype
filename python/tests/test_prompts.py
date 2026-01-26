#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for prompt generation functionality in mLLMCelltype.
"""

import pytest

from mllmcelltype.prompts import (
    create_consensus_check_prompt,
    create_discussion_prompt,
    create_initial_discussion_prompt,
    create_prompt,
)


class TestPrompts:
    """Test class for prompt generation functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_dict, sample_marker_genes_list):
        """Set up test fixtures."""
        self.marker_genes = sample_marker_genes_dict
        self.marker_genes_list = sample_marker_genes_list

    def test_create_prompt_basic(self):
        """Test basic prompt creation."""
        prompt = create_prompt(marker_genes=self.marker_genes, species="human", tissue="blood")

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

    def test_create_initial_discussion_prompt(self):
        """Test initial discussion prompt creation."""
        cluster_id = "1"
        marker_genes = ["CD3D", "CD3E", "CD2"]
        initial_predictions = {"model1": "T cells", "model2": "T cells", "model3": "NK cells"}

        prompt = create_initial_discussion_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            initial_predictions=initial_predictions,
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
        assert "cluster 1" in prompt or "Cluster 1" in prompt

    def test_create_discussion_prompt(self):
        """Test subsequent discussion prompt creation."""
        cluster_id = "1"
        marker_genes = ["CD3D", "CD3E", "CD2"]
        previous_rounds = [
            {"model1": "T cells response", "model2": "NK cells response"},
        ]

        prompt = create_discussion_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            previous_rounds=previous_rounds,
            round_number=2,
            species="human",
            tissue="blood",
        )

        # Check that the prompt contains the expected elements
        assert isinstance(prompt, str)
        assert "human" in prompt
        assert "blood" in prompt
        assert "CD3D" in prompt
        assert "cluster 1" in prompt or "Cluster 1" in prompt
        assert "round 2" in prompt.lower() or "Round 2" in prompt

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
