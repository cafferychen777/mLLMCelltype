#!/usr/bin/env python3
"""
Demo script for testing langextract with mLLMCelltype data.

This script demonstrates how to use langextract to extract structured information
from LLM responses in the mLLMCelltype project context.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Import our test framework
from test_setup import LangExtractTester, traditional_cluster_parser, traditional_consensus_parser
from sample_responses import (
    CELL_TYPE_RESPONSES, CONSENSUS_RESPONSES, DISCUSSION_RESPONSES, 
    TEST_CASES, get_sample_by_type
)

def main():
    """Run the langextract demo."""
    print("🧬 LangExtract Demo for mLLMCelltype")
    print("=" * 50)
    
    # Check for API keys
    api_keys = {
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
    }
    
    available_keys = {k: v for k, v in api_keys.items() if v}
    
    if not available_keys:
        print("❌ No API keys found!")
        print("\nTo run this demo, you need to set at least one of these environment variables:")
        print("- GOOGLE_API_KEY (for Gemini models)")
        print("- GEMINI_API_KEY (for Gemini models)")  
        print("- OPENAI_API_KEY (for OpenAI models)")
        print("- ANTHROPIC_API_KEY (for Claude models)")
        print("\nExample:")
        print("export GOOGLE_API_KEY='your_api_key_here'")
        print("python3 run_demo.py")
        return
    
    print(f"✅ Found API keys for: {', '.join(available_keys.keys())}")
    
    # Initialize tester
    tester = LangExtractTester(api_keys=available_keys)
    
    print("\n📊 Available Demo Tests")
    print("-" * 30)
    print("1. Basic cell type annotation extraction")
    print("2. Consensus result extraction")
    print("3. Advanced consensus parsing comparison (NEW)")
    print("4. Traditional parsing comparison")
    print("5. Performance benchmarking")
    print()
    
    # Let user choose which demos to run
    choice = input("Enter demo number(s) to run (1-5, or 'all'): ").strip().lower()
    
    if choice == 'all':
        demos_to_run = [1, 2, 3, 4, 5]
    else:
        try:
            demos_to_run = [int(x.strip()) for x in choice.split(',')]
        except ValueError:
            print("Invalid input. Running all demos.")
            demos_to_run = [1, 2, 3, 4, 5]
    
    if 1 in demos_to_run:
        demo_basic_extraction(tester)
    
    if 2 in demos_to_run:
        demo_consensus_extraction(tester)
    
    if 3 in demos_to_run:
        demo_advanced_consensus_parsing()
    
    if 4 in demos_to_run:
        demo_comparison_test(tester)
    
    if 5 in demos_to_run:
        demo_performance_benchmark(tester)
    
    print("\n✅ Demo completed!")
    print("\nNext steps:")
    print("1. Review the generated results files")
    print("2. Adjust extraction prompts for your specific needs")
    print("3. Run comprehensive benchmarks with your data")


def demo_basic_extraction(tester: LangExtractTester):
    """Demo basic cell type extraction."""
    print("\n🔬 Demo 1: Basic Cell Type Extraction")
    print("-" * 40)
    
    # Test with a simple response
    sample_text = CELL_TYPE_RESPONSES["openai_gpt4_response"]
    print(f"Sample text (first 100 chars): {sample_text[:100]}...")
    
    # Define extraction prompt
    prompt = """
    Extract cell type annotations from this text. For each cluster mentioned, identify:
    - cluster_id: The cluster identifier (usually a number)
    - cell_type: The assigned cell type name
    
    Return as JSON with an 'annotations' array.
    """
    
    print("\n🔄 Running extraction...")
    result = tester.test_basic_extraction(sample_text, prompt)
    
    if result["success"]:
        print("✅ Extraction successful!")
        print(f"⏱️  Time: {result['execution_time']:.2f} seconds")
        print(f"🤖 Model: {result.get('model_used', 'unknown')}")
        
        # Try to parse the result
        try:
            if hasattr(result['result'], 'extractions'):
                extracted_data = result['result'].extractions
                print(f"📄 Extracted {len(extracted_data)} items")
                if extracted_data:
                    print("Sample extraction:", json.dumps(extracted_data[0], indent=2, default=str))
        except Exception as e:
            print(f"Note: Could not parse extraction details: {e}")
    else:
        print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
    
    # Save result
    tester.save_results(result, "demo1_basic_extraction.json")
    print("💾 Results saved to demo1_basic_extraction.json")


def demo_consensus_extraction(tester: LangExtractTester):
    """Demo consensus result extraction."""
    print("\n🤝 Demo 2: Consensus Result Extraction")
    print("-" * 40)
    
    sample_text = CONSENSUS_RESPONSES["consensus_reached"]
    print(f"Sample text (first 100 chars): {sample_text[:100]}...")
    
    prompt = """
    Extract consensus analysis results from this text. Identify:
    - consensus_reached: whether consensus was reached (true/false)
    - majority_prediction: the final cell type prediction
    - consensus_proportion: numeric value of agreement level
    - entropy: measure of uncertainty
    
    Return as JSON.
    """
    
    print("\n🔄 Running extraction...")
    result = tester.test_basic_extraction(sample_text, prompt)
    
    if result["success"]:
        print("✅ Consensus extraction successful!")
        print(f"⏱️  Time: {result['execution_time']:.2f} seconds")
    else:
        print(f"❌ Consensus extraction failed: {result.get('error', 'Unknown error')}")
    
    tester.save_results(result, "demo2_consensus_extraction.json")
    print("💾 Results saved to demo2_consensus_extraction.json")


def demo_comparison_test(tester: LangExtractTester):
    """Demo comparison between langextract and traditional parsing."""
    print("\n⚖️  Demo 3: Comparison with Traditional Parsing")
    print("-" * 50)
    
    # Use mixed format response for a challenging test
    sample_text = CELL_TYPE_RESPONSES["mixed_format_response"]
    print(f"Testing with mixed format response...")
    
    prompt = "Extract all cell type annotations from this text, including cluster IDs and cell types."
    
    print("\n🔄 Running comparison...")
    results = tester.compare_extraction_methods(
        sample_text, 
        prompt,
        traditional_cluster_parser
    )
    
    print("\n📊 Comparison Results:")
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        if result["success"]:
            print(f"  ✅ Success - Time: {result['execution_time']:.3f}s")
            if method == "traditional":
                print(f"  📊 Found {len(result['result'])} annotations")
            else:
                print(f"  🤖 Model: {result.get('model_used', 'unknown')}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    tester.save_results(results, "demo3_comparison.json")
    print("\n💾 Results saved to demo3_comparison.json")


def demo_performance_benchmark(tester: LangExtractTester):
    """Demo performance benchmarking."""
    print("\n🏃‍♂️ Demo 4: Performance Benchmark")
    print("-" * 35)
    
    # Create test cases with different complexity levels
    test_cases = [
        {
            "name": "Simple_Format",
            "text": CELL_TYPE_RESPONSES["openai_gpt4_response"],
            "prompt_description": "Extract cell type annotations with cluster IDs and cell types.",
            "difficulty": "easy"
        },
        {
            "name": "Detailed_Format", 
            "text": CELL_TYPE_RESPONSES["claude_response"],
            "prompt_description": "Extract detailed cell type annotations including markers and reasoning.",
            "difficulty": "medium"
        },
        {
            "name": "Mixed_Format",
            "text": CELL_TYPE_RESPONSES["mixed_format_response"],
            "prompt_description": "Extract all cell type information from mixed formatting.",
            "difficulty": "hard"
        }
    ]
    
    print(f"Running benchmark with {len(test_cases)} test cases...")
    print("This may take a few minutes...")
    
    # Run with just 1 iteration for demo purposes
    results = tester.benchmark_performance(test_cases, iterations=1)
    
    print("\n📈 Benchmark Results:")
    print(f"Total cases: {results['total_cases']}")
    print(f"Success rate: {results['summary']['success_rate']:.1%}")
    print(f"Average time per extraction: {results['summary']['avg_execution_time']:.2f}s")
    print(f"Total time: {results['summary']['total_time']:.2f}s")
    
    print("\n📋 Individual Results:")
    for result in results["individual_results"]:
        print(f"  {result['case_name']}: {result['success_rate']:.1%} success, {result['avg_time']:.2f}s avg")
    
    tester.save_results(results, "demo4_benchmark.json")
    print("\n💾 Results saved to demo4_benchmark.json")


def demo_advanced_consensus_parsing():
    """Run the advanced consensus parsing comparison demo."""
    print("\n🤝 Demo 3: Advanced Consensus Parsing Comparison")
    print("-" * 50)
    print("This is a specialized test focusing on consensus indicator extraction.")
    print("Running dedicated demo script...")
    print()
    
    try:
        # Import and run the specialized demo
        import demo3_consensus_parsing
        demo3_consensus_parsing.main()
    except ImportError as e:
        print(f"❌ Could not import demo3_consensus_parsing: {e}")
        print("Make sure demo3_consensus_parsing.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error running consensus parsing demo: {e}")


if __name__ == "__main__":
    main()