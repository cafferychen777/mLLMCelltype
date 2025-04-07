#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for LLMCelltype provider modules.
Run with pytest: pytest -xvs tests/test_providers.py
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import mllmcelltype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mllmcelltype.providers import openai, anthropic, deepseek, gemini, qwen, zhipu, minimax, stepfun

# Test OpenAI provider
class TestOpenAIProvider:
    """Tests for OpenAI provider module."""
    
    @patch('openai.OpenAI')
    def test_request_openai(self, mock_openai):
        """Test OpenAI API request function."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "1": "T cells",
            "2": "B cells"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test function
        result = openai.request_openai(
            prompt="Test prompt",
            model="gpt-4o",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"
        
        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert len(kwargs["messages"]) > 0
        assert kwargs["messages"][0]["content"] == "Test prompt"

# Test Anthropic provider
class TestAnthropicProvider:
    """Tests for Anthropic provider module."""
    
    @patch('anthropic.Anthropic')
    def test_request_anthropic(self, mock_anthropic):
        """Test Anthropic API request function."""
        # Setup mock response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [{"text": json.dumps({
            "1": "T cells",
            "2": "B cells"
        })}]
        mock_client.messages.create.return_value = mock_response
        
        # Test function
        result = anthropic.request_anthropic(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"
        
        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        args, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-3-opus-20240229"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert kwargs["messages"][0]["content"] == "Test prompt"

# Test Gemini provider
class TestGeminiProvider:
    """Tests for Google Gemini provider module."""
    
    @patch('google.generativeai.GenerativeModel')
    def test_request_gemini(self, mock_genai):
        """Test Gemini API request function."""
        # Setup mock response
        mock_model = MagicMock()
        mock_genai.return_value = mock_model
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "1": "T cells",
            "2": "B cells"
        })
        mock_model.generate_content.return_value = mock_response
        
        # Test function with patch for the GenerativeModel constructor
        with patch('google.generativeai.configure') as mock_configure:
            result = gemini.request_gemini(
                prompt="Test prompt",
                model="gemini-1.5-pro",
                api_key="test-key",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Verify configure was called
            mock_configure.assert_called_once_with(api_key="test-key")
        
        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"
        
        # Verify model generation was called correctly
        mock_model.generate_content.assert_called_once()
        args, kwargs = mock_model.generate_content.call_args
        assert args[0] == "Test prompt"
        assert "generation_config" in kwargs

# Helper function to create generic provider tests
def create_provider_test(provider_module, request_function_name, mock_path):
    """Create a generic test for a provider module."""
    
    @patch(mock_path)
    def test_function(mock_request):
        """Test provider API request function."""
        # Setup mock response
        mock_response = json.dumps({
            "1": "T cells",
            "2": "B cells"
        })
        
        if isinstance(mock_request, MagicMock):
            # Direct mock
            mock_request.return_value = mock_response
        else:
            # Mock for more complex paths
            mock_obj = MagicMock()
            mock_obj.return_value = mock_response
            mock_request.return_value = mock_obj
        
        # Get the request function
        request_function = getattr(provider_module, request_function_name)
        
        # Test function
        result = request_function(
            prompt="Test prompt",
            model="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"] == "T cells"
        assert result["2"] == "B cells"
        
        # Verify API was called
        if isinstance(mock_request, MagicMock):
            mock_request.assert_called_once()
        
    return test_function

# Create tests for other providers
test_request_deepseek = create_provider_test(
    deepseek, "request_deepseek", "requests.post"
)

test_request_qwen = create_provider_test(
    qwen, "request_qwen", "requests.post"
)

test_request_zhipu = create_provider_test(
    zhipu, "request_zhipu", "zhipuai.ZhipuAI"
)

test_request_minimax = create_provider_test(
    minimax, "request_minimax", "requests.post"
)

test_request_stepfun = create_provider_test(
    stepfun, "request_stepfun", "requests.post"
)

# Test error handling
def test_error_handling():
    """Test error handling in provider modules."""
    
    # Test with invalid API key
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception) as excinfo:
            deepseek.request_deepseek(
                prompt="Test prompt",
                model="deepseek-chat",
                api_key="invalid-key"
            )
        
        assert "API request failed" in str(excinfo.value)
    
    # Test with rate limit error
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception) as excinfo:
            qwen.request_qwen(
                prompt="Test prompt",
                model="qwen-turbo",
                api_key="test-key"
            )
        
        assert "API request failed" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
