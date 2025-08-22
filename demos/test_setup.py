#!/usr/bin/env python3
"""
Basic setup and configuration for langextract testing with mLLMCelltype.

This module provides the foundation for testing langextract effectiveness
on real LLM responses from the mLLMCelltype project.
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# Add the parent directory to the path to import mllmcelltype modules
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import langextract
    from langextract import extract
except ImportError as e:
    print(f"Error importing langextract: {e}")
    print("Please install langextract: pip install langextract")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LangExtractTester:
    """Test framework for evaluating langextract on mLLMCelltype outputs."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize the tester with API keys."""
        self.api_keys = api_keys or {}
        self.default_model = "gemini-2.5-flash"
        self.default_api_key = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment variables for API keys."""
        # Default API keys from environment
        default_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'), 
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
            'QWEN_API_KEY': os.getenv('QWEN_API_KEY'),
            'GROK_API_KEY': os.getenv('GROK_API_KEY'),
            'LANGEXTRACT_API_KEY': os.getenv('LANGEXTRACT_API_KEY'),
        }
        
        # Use provided keys or fallback to environment
        for key, value in default_keys.items():
            if value and key not in self.api_keys:
                self.api_keys[key] = value
        
        # Set environment variables for langextract
        for key, value in self.api_keys.items():
            if value:
                os.environ[key] = value
        
        # Set default API key for langextract (prefer Google/Gemini)
        if self.api_keys.get('GOOGLE_API_KEY'):
            self.default_api_key = self.api_keys['GOOGLE_API_KEY']
            os.environ['LANGEXTRACT_API_KEY'] = self.default_api_key
        elif self.api_keys.get('GEMINI_API_KEY'):
            self.default_api_key = self.api_keys['GEMINI_API_KEY'] 
            os.environ['LANGEXTRACT_API_KEY'] = self.default_api_key
        elif self.api_keys.get('OPENAI_API_KEY'):
            self.default_model = "gpt-4o-mini"
            self.default_api_key = self.api_keys['OPENAI_API_KEY']
            os.environ['LANGEXTRACT_API_KEY'] = self.default_api_key
        
        logger.info(f"Configured API keys for: {list(k for k, v in self.api_keys.items() if v)}")
    
    def initialize_langextract(self, model: str = None, api_key: str = None):
        """Initialize langextract with specific model and API key."""
        try:
            # Use provided parameters or defaults
            use_model = model or self.default_model
            use_api_key = api_key or self.default_api_key
            
            if not use_api_key:
                logger.error("No API key available for langextract")
                return False
            
            # Test basic functionality
            test_result = extract(
                "Test input", 
                "Extract any information",
                model_id=use_model,
                api_key=use_api_key
            )
            
            logger.info(f"Initialized langextract with model: {use_model}")
            self.current_model = use_model
            self.current_api_key = use_api_key
            return True
        except Exception as e:
            logger.error(f"Failed to initialize langextract: {e}")
            return False
    
    def test_basic_extraction(self, sample_text: str, prompt_description: str, 
                             model: str = None, api_key: str = None) -> Dict[str, Any]:
        """Test basic extraction functionality."""
        try:
            use_model = model or self.current_model if hasattr(self, 'current_model') else self.default_model
            use_api_key = api_key or self.current_api_key if hasattr(self, 'current_api_key') else self.default_api_key
            
            if not use_api_key:
                logger.error("No API key available for extraction")
                return {
                    "result": None,
                    "execution_time": 0,
                    "success": False,
                    "error": "No API key available"
                }
            
            start_time = time.time()
            result = extract(
                sample_text,
                prompt_description=prompt_description,
                model_id=use_model,
                api_key=use_api_key,
                debug=False  # Disable debug for cleaner output
            )
            end_time = time.time()
            
            logger.info(f"Extraction completed in {end_time - start_time:.2f} seconds")
            return {
                "result": result,
                "execution_time": end_time - start_time,
                "success": True,
                "model_used": use_model
            }
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                "result": None,
                "execution_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def compare_extraction_methods(self, sample_text: str, prompt_description: str, 
                                 traditional_parser: callable = None) -> Dict[str, Any]:
        """Compare langextract results with traditional parsing methods."""
        results = {}
        
        # Test langextract
        langextract_result = self.test_basic_extraction(sample_text, prompt_description)
        results["langextract"] = langextract_result
        
        # Test traditional parser if provided
        if traditional_parser:
            try:
                start_time = time.time()
                traditional_result = traditional_parser(sample_text)
                end_time = time.time()
                
                results["traditional"] = {
                    "result": traditional_result,
                    "execution_time": end_time - start_time,
                    "success": True
                }
            except Exception as e:
                results["traditional"] = {
                    "result": None,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def benchmark_performance(self, test_cases: List[Dict[str, Any]], iterations: int = 3) -> Dict[str, Any]:
        """Benchmark langextract performance on multiple test cases."""
        results = {
            "total_cases": len(test_cases),
            "iterations": iterations,
            "individual_results": [],
            "summary": {
                "success_rate": 0,
                "avg_execution_time": 0,
                "total_time": 0
            }
        }
        
        total_time = 0
        successful_extractions = 0
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}/{len(test_cases)}")
            
            case_results = []
            for j in range(iterations):
                result = self.test_basic_extraction(
                    test_case["text"], 
                    test_case.get("prompt_description", "Extract structured information from this text")
                )
                case_results.append(result)
                total_time += result["execution_time"]
                
                if result["success"]:
                    successful_extractions += 1
            
            results["individual_results"].append({
                "case_id": i,
                "case_name": test_case.get("name", f"Case_{i}"),
                "results": case_results,
                "avg_time": sum(r["execution_time"] for r in case_results) / iterations,
                "success_rate": sum(r["success"] for r in case_results) / iterations
            })
        
        # Calculate summary statistics
        total_extractions = len(test_cases) * iterations
        results["summary"]["success_rate"] = successful_extractions / total_extractions if total_extractions > 0 else 0
        results["summary"]["avg_execution_time"] = total_time / total_extractions if total_extractions > 0 else 0
        results["summary"]["total_time"] = total_time
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save test results to JSON file."""
        output_path = Path(__file__).parent / filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load test results from JSON file."""
        results_path = Path(__file__).parent / filename
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}


def create_cell_type_schema() -> Dict[str, Any]:
    """Create a schema for extracting cell type annotations."""
    return {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "cell_type": {"type": "string"},
                        "confidence": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["cluster_id", "cell_type"]
                }
            }
        },
        "required": ["annotations"]
    }


def create_consensus_schema() -> Dict[str, Any]:
    """Create a schema for extracting consensus results."""
    return {
        "type": "object",
        "properties": {
            "consensus_reached": {"type": "boolean"},
            "majority_prediction": {"type": "string"},
            "consensus_proportion": {"type": "number"},
            "entropy": {"type": "number"},
            "prediction_counts": {
                "type": "object",
                "additionalProperties": {"type": "integer"}
            }
        },
        "required": ["consensus_reached", "majority_prediction"]
    }


def create_discussion_schema() -> Dict[str, Any]:
    """Create a schema for extracting discussion results."""
    return {
        "type": "object",
        "properties": {
            "round": {"type": "integer"},
            "participants": {
                "type": "array",
                "items": {"type": "string"}
            },
            "responses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "response": {"type": "string"},
                        "cell_type": {"type": "string"}
                    },
                    "required": ["model", "response"]
                }
            },
            "summary": {"type": "string"}
        },
        "required": ["round", "responses"]
    }


def traditional_cluster_parser(text: str) -> List[Dict[str, str]]:
    """Traditional regex-based parser for cluster annotations."""
    import re
    
    annotations = []
    
    # Pattern for "cluster_id: cell_type" format
    pattern1 = r'(?:cluster\s*)?(\d+)\s*:\s*([^\n\r]+)'
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    
    for cluster_id, cell_type in matches1:
        annotations.append({
            "cluster_id": cluster_id.strip(),
            "cell_type": cell_type.strip()
        })
    
    # Pattern for numbered list format "1. cell_type"
    if not annotations:
        pattern2 = r'(\d+)[\.\-\s]+([^\n\r]+)'
        matches2 = re.findall(pattern2, text)
        
        for idx, cell_type in matches2:
            # Convert 1-based to 0-based indexing
            cluster_id = str(int(idx) - 1) if idx.isdigit() else idx
            annotations.append({
                "cluster_id": cluster_id,
                "cell_type": cell_type.strip()
            })
    
    return annotations


def traditional_consensus_parser(text: str) -> Dict[str, Any]:
    """Traditional parser for consensus results."""
    import re
    
    result = {
        "consensus_reached": False,
        "majority_prediction": "",
        "consensus_proportion": 0.0,
        "entropy": 0.0
    }
    
    # Search for consensus indicators
    if re.search(r'consensus\s+reached', text, re.IGNORECASE):
        result["consensus_reached"] = True
    
    # Extract majority prediction
    prediction_patterns = [
        r'majority\s+prediction\s*:\s*([^\n\r]+)',
        r'consensus\s*:\s*([^\n\r]+)',
        r'agreed\s+on\s*:\s*([^\n\r]+)'
    ]
    
    for pattern in prediction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["majority_prediction"] = match.group(1).strip()
            break
    
    # Extract numeric values
    prop_match = re.search(r'consensus\s+proportion\s*:\s*([\d\.]+)', text, re.IGNORECASE)
    if prop_match:
        result["consensus_proportion"] = float(prop_match.group(1))
    
    entropy_match = re.search(r'entropy\s*:\s*([\d\.]+)', text, re.IGNORECASE)
    if entropy_match:
        result["entropy"] = float(entropy_match.group(1))
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Testing LangExtract setup...")
    tester = LangExtractTester()
    
    # Check if we have any API keys
    available_keys = [k for k, v in tester.api_keys.items() if v]
    if available_keys:
        print(f"Found API keys for: {', '.join(available_keys)}")
        
        # Try to run a simple test if we have keys
        try:
            result = tester.test_basic_extraction(
                "Cluster 0: T cells\nCluster 1: B cells",
                "Extract cell type annotations from this text. For each cluster, identify the cluster ID and cell type."
            )
            if result["success"]:
                print("✓ Basic extraction test successful!")
                print("LangExtract is ready for testing.")
            else:
                print(f"✗ Basic extraction test failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    else:
        print("No API keys found. Please set environment variables or provide keys to the tester.")
        print("Ready for testing once API keys are configured.")