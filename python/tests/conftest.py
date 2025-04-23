#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixtures and configuration for pytest.
"""

import pandas as pd
import pytest


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


@pytest.fixture
def sample_marker_genes_list():
    """Create a list of sample marker genes dictionaries for testing."""
    return [
        {"1": ["CD3D", "CD3E", "CD2"], "2": ["CD19", "MS4A1", "CD79A"]},
        {"3": ["NCAM1", "KLRB1", "KLRD1"], "4": ["CD14", "CD68", "FCGR3A"]},
    ]


@pytest.fixture
def mock_api_response():
    """Create a mock API response for testing."""
    return [
        "Cluster 1: T cells",
        "Cluster 2: B cells",
        "Cluster 3: NK cells",
        "Cluster 4: Monocytes",
    ]


@pytest.fixture
def mock_env_with_api_keys(monkeypatch):
    """Set up environment variables with mock API keys."""
    env_vars = {
        "OPENAI_API_KEY": "test-key-openai-xxxxxxxxxxxx",
        "ANTHROPIC_API_KEY": "test-key-anthropic-xxxxxxxxxxxx",
        "DEEPSEEK_API_KEY": "test-key-deepseek-xxxxxxxxxxxx",
        "GEMINI_API_KEY": "test-key-gemini-xxxxxxxxxxxx",
        "QWEN_API_KEY": "test-key-qwen-xxxxxxxxxxxx",
        "ZHIPU_API_KEY": "test-key-zhipu-xxxxxxxxxxxx",
        "STEPFUN_API_KEY": "test-key-stepfun-xxxxxxxxxxxx",
        "MINIMAX_API_KEY": "test-key-minimax-xxxxxxxxxxxx",
        "MINIMAX_GROUP_ID": "test-group-id-xxxxxxxxxxxx",
        "GROK_API_KEY": "test-key-grok-xxxxxxxxxxxx",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
