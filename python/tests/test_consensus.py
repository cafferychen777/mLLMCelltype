#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for consensus and comparison functionality in mLLMCelltype.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from mllmcelltype.consensus import (
    check_consensus,
    interactive_consensus_annotation,
    check_consensus_with_llm
)
from mllmcelltype.utils import find_agreement

class TestConsensus:
    """Test class for consensus functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_df, sample_marker_genes_dict):
        """Set up test fixtures."""
        self.marker_genes_df = sample_marker_genes_df
        self.marker_genes_dict = sample_marker_genes_dict
        
        # Sample annotations from different models
        self.model_annotations = {
            "gpt-4o": {  # 使用真实的模型名称
                "1": "T cells",
                "2": "B cells",
                "3": "NK cells"
            },
            "claude-3-opus": {  # 使用真实的模型名称
                "1": "T lymphocytes",
                "2": "B lymphocytes",
                "3": "Natural killer cells"
            },
            "gemini-1.5-pro": {  # 使用真实的模型名称
                "1": "CD4+ T cells",
                "2": "Plasma B cells",
                "3": "NK cells"
            }
        }
    
    def test_check_consensus_with_llm(self):
        """Test check_consensus_with_llm function."""
        # 由于 check_consensus_with_llm 函数的复杂性，我们将直接测试函数的基本行为
        # 而不是尝试模拟所有内部细节
        
        # 创建一个简单的预测字典，其中所有模型对 cluster 3 都有相同的预测
        simple_predictions = {
            "model1": {"3": "NK cells"},
            "model2": {"3": "NK cells"},
            "model3": {"3": "NK cells"}
        }
        
        # 测试函数
        consensus, consensus_proportion, entropy = check_consensus_with_llm(
            predictions=simple_predictions
        )
        
        # 验证结果
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert "3" in consensus
        assert consensus["3"] == "NK cells"
        assert consensus_proportion["3"] == 1.0  # 完全一致，应该是 1.0
        assert entropy["3"] == 0.0  # 完全一致，熵应该是 0
    
    @patch("mllmcelltype.consensus.check_consensus_with_llm")
    def test_check_consensus(self, mock_check_consensus_with_llm):
        """Test check_consensus function."""
        # Setup mocks
        mock_check_consensus_with_llm.return_value = (
            {"1": "T cells", "2": "B cells", "3": "NK cells"},
            {"1": 0.85, "2": 0.60, "3": 0.95},  # 注意：cluster 2 的共识比例低于阈值
            {"1": 0.45, "2": 0.50, "3": 0.20}
        )
        
        # Test function
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            predictions=self.model_annotations,
            consensus_threshold=0.7,  # 设置阈值为 0.7，这样 cluster 2 就会被识别为有争议的
            entropy_threshold=0.6,
            api_keys={"openai": "test-key"}
        )
        
        # Verify results
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert isinstance(controversial, list)
        assert "1" in consensus
        assert "2" in consensus
        assert "3" in consensus
        assert consensus["1"] == "T cells"
        assert consensus["2"] == "B cells"
        assert consensus["3"] == "NK cells"
        assert "2" in controversial  # cluster 2 应该被识别为有争议的
    
    @patch("mllmcelltype.functions.get_provider")
    @patch("mllmcelltype.annotate.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation(self, mock_process_controversial, mock_check_consensus, 
                                             mock_annotate_clusters, mock_get_provider):
        """Test interactive_consensus_annotation function."""
        # Setup mocks
        mock_get_provider.return_value = "openai"  # 确保 get_provider 返回有效的提供者
        mock_annotate_clusters.side_effect = [
            {"1": "T cells", "2": "B cells", "3": "NK cells"},
            {"1": "T lymphocytes", "2": "B lymphocytes", "3": "Natural killer cells"},
            {"1": "CD4+ T cells", "2": "Plasma B cells", "3": "NK cells"}
        ]
        mock_check_consensus.return_value = (
            {"1": "T cells", "3": "NK cells"},  # Consensus for non-controversial clusters
            {"1": 0.85, "2": 0.60, "3": 0.95},  # Consensus proportions
            {"1": 0.45, "2": 0.70, "3": 0.20},  # Entropy values
            ["2"]  # Controversial clusters
        )
        mock_process_controversial.return_value = (
            {"2": "B cells"},  # Resolved annotations
            {"2": ["Discussion round 1", "Discussion round 2"]},  # Discussion history
            {"2": 0.85},  # Updated consensus proportions
            {"2": 0.40}  # Updated entropy values
        )
        
        # Test function
        result = interactive_consensus_annotation(
            marker_genes=self.marker_genes_dict,
            species="human",
            models=["gpt-4o", "claude-3-opus", "gemini-1.5-pro"],  # 使用真实的模型名称
            api_keys={"openai": "test-key", "anthropic": "test-key", "gemini": "test-key"},
            tissue="blood",
            consensus_threshold=0.7,
            entropy_threshold=0.6,
            max_discussion_rounds=2,
            use_cache=False
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "consensus" in result
        assert "consensus_proportion" in result
        assert "entropy" in result
        assert "model_annotations" in result
        assert "controversial_clusters" in result
        assert "discussion_logs" in result  # 注意：字段名是 discussion_logs 而不是 discussion_history
        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "B cells"
        assert result["consensus"]["3"] == "NK cells"
        assert result["controversial_clusters"] == ["2"]
        assert len(result["model_annotations"]) == 3
        assert result["consensus_proportion"]["2"] == 0.85
        assert result["entropy"]["2"] == 0.40

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])