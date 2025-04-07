#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for prompt templates in LLMCelltype.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to import mllmcelltype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mllmcelltype.prompts import create_prompt, create_batch_prompt
from mllmcelltype.prompts import DEFAULT_PROMPT_TEMPLATE, DEFAULT_BATCH_PROMPT_TEMPLATE


class TestPrompts(unittest.TestCase):
    """Test prompt generation functionality."""

    def setUp(self):
        """Set up test data."""
        self.marker_genes = {
            "1": ["CD3D", "CD3E", "CD2"],
            "2": ["CD19", "MS4A1", "CD79A"],
            "3": ["FCGR3A", "CD14", "LYZ"]
        }
        
        self.marker_genes_list = [
            {
                "1": ["CD3D", "CD3E", "CD2"],
                "2": ["CD19", "MS4A1", "CD79A"]
            },
            {
                "1": ["FCGR3A", "CD14", "LYZ"],
                "2": ["CD4", "CD8A", "IL7R"]
            }
        ]

    def test_create_prompt_basic(self):
        """Test basic prompt creation."""
        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human",
            tissue="blood"
        )
        
        # Check that the prompt contains the expected elements
        self.assertIn("human blood", prompt)
        self.assertIn("Cluster 1: CD3D, CD3E, CD2", prompt)
        self.assertIn("Cluster 2: CD19, MS4A1, CD79A", prompt)
        self.assertIn("Cluster 3: FCGR3A, CD14, LYZ", prompt)
        
    def test_create_prompt_no_tissue(self):
        """Test prompt creation without tissue."""
        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human"
        )
        
        # Check that the prompt contains the expected elements
        self.assertIn("human", prompt)
        self.assertNotIn("blood", prompt)
        
    def test_create_prompt_with_additional_context(self):
        """Test prompt creation with additional context."""
        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human",
            tissue="blood",
            additional_context="Sample from a healthy donor."
        )
        
        # Check that the prompt contains the expected elements
        self.assertIn("human blood", prompt)
        self.assertIn("Sample from a healthy donor", prompt)
        
    def test_create_prompt_custom_template(self):
        """Test prompt creation with custom template."""
        custom_template = """Custom template for {context}.
        
        Genes:
        {clusters}
        """
        
        prompt = create_prompt(
            marker_genes=self.marker_genes,
            species="human",
            tissue="blood",
            prompt_template=custom_template
        )
        
        # Check that the prompt contains the expected elements
        self.assertIn("Custom template for human blood", prompt)
        self.assertIn("Cluster 1: CD3D, CD3E, CD2", prompt)
        
    def test_create_batch_prompt(self):
        """Test batch prompt creation."""
        prompt = create_batch_prompt(
            marker_genes_list=self.marker_genes_list,
            species="human",
            tissue="blood"
        )
        
        # Check that the prompt contains the expected elements
        self.assertIn("human blood", prompt)
        self.assertIn("Set 1:", prompt)
        self.assertIn("Set 2:", prompt)
        self.assertIn("Cluster 1: CD3D, CD3E, CD2", prompt)
        self.assertIn("Cluster 2: CD19, MS4A1, CD79A", prompt)
        self.assertIn("Cluster 1: FCGR3A, CD14, LYZ", prompt)
        self.assertIn("Cluster 2: CD4, CD8A, IL7R", prompt)


if __name__ == "__main__":
    unittest.main()
