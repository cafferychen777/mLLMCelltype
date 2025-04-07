#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for JSON response parsing in LLMCelltype.
"""

import os
import sys
import unittest

# Add parent directory to path to import mllmcelltype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mllmcelltype.utils import format_results


class TestJsonParsing(unittest.TestCase):
    """Test JSON response parsing functionality."""

    def setUp(self):
        """Set up test data."""
        self.clusters = ["1", "2", "3"]
        
        # Example JSON response from OpenAI
        self.json_response = [
            "```json",
            "{",
            '  "annotations": [',
            "    {",
            '      "cluster": "1",',
            '      "cell_type": "T cells",',
            '      "confidence": "high",',
            '      "key_markers": ["CD3D", "CD3E", "CD2"]',
            "    },",
            "    {",
            '      "cluster": "2",',
            '      "cell_type": "B cells",',
            '      "confidence": "high",',
            '      "key_markers": ["CD19", "MS4A1", "CD79A"]',
            "    },",
            "    {",
            '      "cluster": "3",',
            '      "cell_type": "Monocytes",',
            '      "confidence": "high",',
            '      "key_markers": ["FCGR3A", "CD14", "LYZ"]',
            "    }",
            "  ]",
            "}",
            "```"
        ]
        
        # JSON response without code block markers
        self.json_response_no_markers = [
            "{",
            '  "annotations": [',
            "    {",
            '      "cluster": "1",',
            '      "cell_type": "T cells",',
            '      "confidence": "high",',
            '      "key_markers": ["CD3D", "CD3E", "CD2"]',
            "    },",
            "    {",
            '      "cluster": "2",',
            '      "cell_type": "B cells",',
            '      "confidence": "high",',
            '      "key_markers": ["CD19", "MS4A1", "CD79A"]',
            "    },",
            "    {",
            '      "cluster": "3",',
            '      "cell_type": "Monocytes",',
            '      "confidence": "high",',
            '      "key_markers": ["FCGR3A", "CD14", "LYZ"]',
            "    }",
            "  ]",
            "}"
        ]
        
        # JSON response with incorrect format (missing commas)
        self.json_response_incorrect = [
            "```json",
            "{",
            '  "annotations": [',
            "    {",
            '      "cluster": "1"',
            '      "cell_type": "T cells"',
            '      "confidence": "high"',
            '      "key_markers": ["CD3D", "CD3E", "CD2"]',
            "    }",
            "    {",
            '      "cluster": "2"',
            '      "cell_type": "B cells"',
            '      "confidence": "high"',
            '      "key_markers": ["CD19", "MS4A1", "CD79A"]',
            "    }",
            "    {",
            '      "cluster": "3"',
            '      "cell_type": "Monocytes"',
            '      "confidence": "high"',
            '      "key_markers": ["FCGR3A", "CD14", "LYZ"]',
            "    }",
            "  ]",
            "}",
            "```"
        ]
        
        # Expected result after parsing
        self.expected_result = {
            "1": "T cells",
            "2": "B cells",
            "3": "Monocytes"
        }

    def test_json_parsing_with_markers(self):
        """Test parsing JSON response with code block markers."""
        result = format_results(self.json_response, self.clusters)
        print("JSON parsing result with markers:", result)
        self.assertEqual(result, self.expected_result)
        
    def test_json_parsing_without_markers(self):
        """Test parsing JSON response without code block markers."""
        result = format_results(self.json_response_no_markers, self.clusters)
        print("JSON parsing result without markers:", result)
        self.assertEqual(result, self.expected_result)
        
    def test_json_parsing_incorrect_format(self):
        """Test parsing JSON response with incorrect format (missing commas)."""
        result = format_results(self.json_response_incorrect, self.clusters)
        print("JSON parsing result with incorrect format:", result)
        self.assertEqual(result, self.expected_result)


if __name__ == "__main__":
    unittest.main()
