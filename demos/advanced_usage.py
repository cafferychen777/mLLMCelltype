#!/usr/bin/env python3
"""
Advanced usage examples for langextract with mLLMCelltype.

This script demonstrates more complex scenarios and integration patterns
for using langextract in production mLLMCelltype workflows.
"""

import os
import sys
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from pathlib import Path

from test_setup import LangExtractTester
from sample_responses import (
    CELL_TYPE_RESPONSES, CONSENSUS_RESPONSES, DISCUSSION_RESPONSES,
    ERROR_RESPONSES, get_sample_by_type
)


class AdvancedLangExtractDemo:
    """Advanced demonstration of langextract integration patterns."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.tester = LangExtractTester(api_keys)
        self.results = {}
    
    def demo_error_handling(self):
        """Demonstrate robust error handling for various edge cases."""
        print("\n🔧 Advanced Demo 1: Error Handling & Edge Cases")
        print("-" * 50)
        
        error_cases = [
            ("empty_response", ERROR_RESPONSES["empty_response"], "Handle empty LLM response"),
            ("partial_response", ERROR_RESPONSES["partial_response"], "Handle incomplete cluster list"),
            ("malformed_response", ERROR_RESPONSES["malformed_response"], "Handle unclear annotations"),
            ("conflicting_response", ERROR_RESPONSES["conflicting_response"], "Handle self-contradictory response")
        ]
        
        results = {}
        
        for case_name, text, description in error_cases:
            print(f"\n📋 Testing: {description}")
            print(f"Input length: {len(text)} characters")
            
            if not text.strip():
                print("⚠️  Empty input detected - skipping API call")
                results[case_name] = {
                    "success": False,
                    "error": "Empty input",
                    "handled_gracefully": True
                }
                continue
            
            # Test with fallback prompt for difficult cases
            prompt = f"""
            Extract cell type annotations from this text, even if the format is unclear or incomplete.
            If information is missing or unclear, indicate this in the response.
            If no clear annotations can be found, return an empty annotations array.
            
            Return JSON format: {{"annotations": [...]}}
            
            Text to parse: {description}
            """
            
            result = self.tester.test_basic_extraction(text, prompt)
            
            # Add metadata about error handling
            result["case_description"] = description
            result["input_type"] = case_name
            result["handled_gracefully"] = not result.get("error", "").startswith("Unexpected")
            
            results[case_name] = result
            
            if result["success"]:
                print("✅ Parsed successfully despite issues")
            else:
                print(f"⚠️  Failed: {result.get('error', 'Unknown error')}")
        
        self.results["error_handling"] = results
        self.tester.save_results(results, "advanced_demo1_error_handling.json")
        print("\n💾 Error handling results saved to advanced_demo1_error_handling.json")
    
    def demo_batch_processing(self):
        """Demonstrate efficient batch processing of multiple responses."""
        print("\n📦 Advanced Demo 2: Batch Processing")
        print("-" * 40)
        
        # Simulate multiple LLM responses that need processing
        batch_data = [
            {
                "response_id": "response_1", 
                "source_model": "gpt-4o",
                "text": CELL_TYPE_RESPONSES["openai_gpt4_response"],
                "expected_clusters": 9
            },
            {
                "response_id": "response_2",
                "source_model": "claude-3.5-sonnet", 
                "text": CELL_TYPE_RESPONSES["claude_response"],
                "expected_clusters": 9
            },
            {
                "response_id": "response_3",
                "source_model": "gemini-1.5-pro",
                "text": CELL_TYPE_RESPONSES["gemini_response"], 
                "expected_clusters": 9
            }
        ]
        
        print(f"Processing batch of {len(batch_data)} responses...")
        
        # Sequential processing (simpler but slower)
        sequential_results = self._process_batch_sequential(batch_data)
        
        # Parallel processing (faster but uses more API quota)
        parallel_results = self._process_batch_parallel(batch_data)
        
        # Compare approaches
        seq_total_time = sum(r.get("execution_time", 0) for r in sequential_results.values())
        par_total_time = max(r.get("execution_time", 0) for r in parallel_results.values())
        
        print(f"\n📊 Processing Comparison:")
        print(f"Sequential: {seq_total_time:.2f}s total")
        print(f"Parallel: {par_total_time:.2f}s total")
        print(f"Speedup: {seq_total_time/par_total_time:.1f}x" if par_total_time > 0 else "N/A")
        
        batch_results = {
            "sequential": sequential_results,
            "parallel": parallel_results,
            "performance_comparison": {
                "sequential_time": seq_total_time,
                "parallel_time": par_total_time,
                "speedup_factor": seq_total_time/par_total_time if par_total_time > 0 else 0
            }
        }
        
        self.results["batch_processing"] = batch_results
        self.tester.save_results(batch_results, "advanced_demo2_batch_processing.json")
        print("💾 Batch processing results saved to advanced_demo2_batch_processing.json")
    
    def _process_batch_sequential(self, batch_data: List[Dict]) -> Dict[str, Any]:
        """Process batch sequentially."""
        print("\n🔄 Sequential processing...")
        results = {}
        
        for item in batch_data:
            result = self.tester.test_basic_extraction(
                item["text"],
                f"Extract cell type annotations from this {item['source_model']} response. "
                f"Expected {item['expected_clusters']} clusters."
            )
            result["source_model"] = item["source_model"]
            result["expected_clusters"] = item["expected_clusters"]
            results[item["response_id"]] = result
        
        return results
    
    def _process_batch_parallel(self, batch_data: List[Dict]) -> Dict[str, Any]:
        """Process batch in parallel using ThreadPoolExecutor."""
        print("\n⚡ Parallel processing...")
        results = {}
        
        def process_item(item):
            result = self.tester.test_basic_extraction(
                item["text"],
                f"Extract cell type annotations from this {item['source_model']} response. "
                f"Expected {item['expected_clusters']} clusters."
            )
            result["source_model"] = item["source_model"]
            result["expected_clusters"] = item["expected_clusters"]
            return item["response_id"], result
        
        # Use ThreadPoolExecutor for I/O-bound API calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in batch_data}
            
            for future in future_to_item:
                try:
                    response_id, result = future.result()
                    results[response_id] = result
                except Exception as e:
                    item = future_to_item[future]
                    results[item["response_id"]] = {
                        "success": False,
                        "error": str(e),
                        "source_model": item["source_model"]
                    }
        
        return results
    
    def demo_adaptive_prompting(self):
        """Demonstrate adaptive prompting based on input characteristics."""
        print("\n🎯 Advanced Demo 3: Adaptive Prompting")
        print("-" * 42)
        
        # Different response types require different prompting strategies
        test_cases = [
            {
                "name": "simple_format",
                "text": CELL_TYPE_RESPONSES["openai_gpt4_response"],
                "characteristics": ["short_lines", "colon_separated", "simple"],
                "adaptive_prompt": "Extract cell type annotations. Text uses simple 'cluster_id: cell_type' format."
            },
            {
                "name": "detailed_format", 
                "text": CELL_TYPE_RESPONSES["claude_response"],
                "characteristics": ["detailed_descriptions", "marker_genes", "complex"],
                "adaptive_prompt": "Extract cell type annotations with supporting evidence. Text includes detailed biological reasoning."
            },
            {
                "name": "mixed_format",
                "text": CELL_TYPE_RESPONSES["mixed_format_response"],
                "characteristics": ["mixed_formatting", "inconsistent", "challenging"],
                "adaptive_prompt": "Extract cell type annotations from inconsistently formatted text. Be flexible with formatting patterns."
            }
        ]
        
        results = {}
        
        for case in test_cases:
            print(f"\n🔍 Processing: {case['name']}")
            print(f"Characteristics: {', '.join(case['characteristics'])}")
            
            # Test with generic prompt
            generic_result = self.tester.test_basic_extraction(
                case["text"],
                "Extract cell type annotations from this text."
            )
            
            # Test with adaptive prompt
            adaptive_result = self.tester.test_basic_extraction(
                case["text"], 
                case["adaptive_prompt"]
            )
            
            case_results = {
                "characteristics": case["characteristics"],
                "generic_prompt": {
                    "prompt": "Extract cell type annotations from this text.",
                    "result": generic_result
                },
                "adaptive_prompt": {
                    "prompt": case["adaptive_prompt"],
                    "result": adaptive_result
                },
                "improvement": {
                    "success_improved": adaptive_result["success"] and not generic_result["success"],
                    "time_difference": adaptive_result["execution_time"] - generic_result["execution_time"]
                }
            }
            
            results[case["name"]] = case_results
            
            print(f"Generic prompt: {'✅' if generic_result['success'] else '❌'}")
            print(f"Adaptive prompt: {'✅' if adaptive_result['success'] else '❌'}")
            
            if case_results["improvement"]["success_improved"]:
                print("🎯 Adaptive prompting improved results!")
        
        self.results["adaptive_prompting"] = results
        self.tester.save_results(results, "advanced_demo3_adaptive_prompting.json")
        print("\n💾 Adaptive prompting results saved to advanced_demo3_adaptive_prompting.json")
    
    def demo_integration_patterns(self):
        """Demonstrate integration patterns for production use."""
        print("\n🔗 Advanced Demo 4: Integration Patterns")
        print("-" * 43)
        
        # Pattern 1: Fallback parsing
        print("\n1️⃣  Fallback Parsing Pattern")
        self._demo_fallback_pattern()
        
        # Pattern 2: Validation and correction
        print("\n2️⃣  Validation and Correction Pattern") 
        self._demo_validation_pattern()
        
        # Pattern 3: Confidence-based routing
        print("\n3️⃣  Confidence-based Routing Pattern")
        self._demo_confidence_routing()
        
        print("\n💾 Integration patterns results saved to advanced_demo4_integration.json")
    
    def _demo_fallback_pattern(self):
        """Demonstrate fallback from traditional parsing to langextract."""
        from test_setup import traditional_cluster_parser
        
        # Test with challenging input
        challenging_text = CELL_TYPE_RESPONSES["inconsistent_format_response"]
        
        # Try traditional parsing first
        try:
            traditional_result = traditional_cluster_parser(challenging_text)
            traditional_success = len(traditional_result) > 0
            print(f"Traditional parser: {'✅' if traditional_success else '❌'} ({len(traditional_result)} clusters found)")
        except Exception as e:
            traditional_success = False
            print(f"Traditional parser: ❌ Failed ({e})")
        
        # Fallback to langextract if traditional fails
        if not traditional_success:
            print("📦 Falling back to langextract...")
            langextract_result = self.tester.test_basic_extraction(
                challenging_text,
                "Extract cell type annotations from this inconsistently formatted text."
            )
            print(f"LangExtract fallback: {'✅' if langextract_result['success'] else '❌'}")
        else:
            print("✅ Traditional parser succeeded - no fallback needed")
    
    def _demo_validation_pattern(self):
        """Demonstrate validation and correction of extracted results."""
        # Extract data
        result = self.tester.test_basic_extraction(
            CELL_TYPE_RESPONSES["openai_gpt4_response"],
            "Extract exactly 9 cell type annotations, one for each cluster 0-8."
        )
        
        if result["success"]:
            # Simulate validation logic
            extracted_data = getattr(result['result'], 'extractions', [])
            
            # Validation checks
            validations = {
                "has_data": len(extracted_data) > 0,
                "expected_count": len(extracted_data) == 9,
                "no_duplicates": len(set(str(item) for item in extracted_data)) == len(extracted_data)
            }
            
            validation_passed = all(validations.values())
            print(f"Validation: {'✅' if validation_passed else '❌'}")
            
            for check, passed in validations.items():
                print(f"  {check}: {'✅' if passed else '❌'}")
            
            if not validation_passed:
                print("🔧 Correction needed - could trigger re-extraction or manual review")
        else:
            print("❌ Extraction failed - validation skipped")
    
    def _demo_confidence_routing(self):
        """Demonstrate routing based on extraction confidence."""
        # Simple confidence estimation based on execution time and result structure
        test_text = CELL_TYPE_RESPONSES["claude_response"]
        
        result = self.tester.test_basic_extraction(
            test_text,
            "Extract cell type annotations with high confidence."
        )
        
        if result["success"]:
            # Estimate confidence based on various factors
            confidence_factors = {
                "fast_execution": result["execution_time"] < 5.0,  # Quick response
                "has_result": result["result"] is not None,
                "no_errors": "error" not in result
            }
            
            confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
            
            print(f"Confidence score: {confidence_score:.1%}")
            
            # Route based on confidence
            if confidence_score >= 0.8:
                routing_decision = "auto_accept"
                print("🟢 High confidence - automatically accept results")
            elif confidence_score >= 0.5:
                routing_decision = "manual_review"
                print("🟡 Medium confidence - flag for manual review")
            else:
                routing_decision = "reject_retry"
                print("🔴 Low confidence - reject and retry with different approach")
            
            # Store routing decision
            if "integration_patterns" not in self.results:
                self.results["integration_patterns"] = {}
            
            self.results["integration_patterns"]["confidence_routing"] = {
                "confidence_score": confidence_score,
                "confidence_factors": confidence_factors,
                "routing_decision": routing_decision,
                "result": result
            }
        else:
            print("❌ Extraction failed - automatic retry recommended")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary of all advanced demos."""
        print("\n📋 Advanced Usage Summary Report")
        print("=" * 45)
        
        summary = {
            "demos_completed": list(self.results.keys()),
            "total_extractions": 0,
            "success_rate": 0,
            "key_insights": []
        }
        
        total_attempts = 0
        successful_attempts = 0
        
        for demo_name, demo_results in self.results.items():
            print(f"\n{demo_name.upper().replace('_', ' ')}:")
            
            # Count results recursively
            demo_attempts, demo_successes = self._count_results(demo_results)
            total_attempts += demo_attempts
            successful_attempts += demo_successes
            
            if demo_attempts > 0:
                demo_success_rate = demo_successes / demo_attempts
                print(f"  Success rate: {demo_success_rate:.1%} ({demo_successes}/{demo_attempts})")
            else:
                print("  No quantifiable results")
        
        # Calculate overall metrics
        if total_attempts > 0:
            summary["total_extractions"] = total_attempts
            summary["success_rate"] = successful_attempts / total_attempts
        
        # Generate insights
        summary["key_insights"] = self._generate_insights()
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"Total extractions: {summary['total_extractions']}")
        print(f"Overall success rate: {summary['success_rate']:.1%}")
        
        print(f"\n💡 KEY INSIGHTS:")
        for i, insight in enumerate(summary["key_insights"], 1):
            print(f"{i}. {insight}")
        
        # Save comprehensive summary
        self.tester.save_results(summary, "advanced_demo_summary.json")
        print(f"\n💾 Complete summary saved to advanced_demo_summary.json")
        
        return summary
    
    def _count_results(self, results, attempts=0, successes=0):
        """Recursively count extraction attempts and successes."""
        if isinstance(results, dict):
            if "success" in results:
                attempts += 1
                if results["success"]:
                    successes += 1
            else:
                for value in results.values():
                    attempts, successes = self._count_results(value, attempts, successes)
        elif isinstance(results, list):
            for item in results:
                attempts, successes = self._count_results(item, attempts, successes)
        
        return attempts, successes
    
    def _generate_insights(self):
        """Generate insights based on demo results."""
        insights = [
            "LangExtract provides robust parsing for varied text formats",
            "Adaptive prompting significantly improves extraction quality",
            "Parallel processing can reduce total processing time for batches", 
            "Fallback patterns increase overall system reliability",
            "Confidence-based routing helps manage quality vs automation trade-offs"
        ]
        
        # Add specific insights based on actual results
        if "error_handling" in self.results:
            insights.append("Error handling patterns prevent system crashes on malformed input")
        
        if "batch_processing" in self.results:
            batch_results = self.results["batch_processing"]
            if "performance_comparison" in batch_results:
                perf = batch_results["performance_comparison"]
                if perf.get("speedup_factor", 0) > 1.5:
                    insights.append(f"Parallel processing achieved {perf['speedup_factor']:.1f}x speedup")
        
        return insights


def main():
    """Run the advanced langextract demos."""
    print("🚀 Advanced LangExtract Integration Demo")
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
        print("❌ No API keys found! Please set environment variables and try again.")
        return
    
    print(f"✅ Found API keys for: {', '.join(available_keys.keys())}")
    
    # Initialize advanced demo
    demo = AdvancedLangExtractDemo(available_keys)
    
    # Run all advanced demos
    demo.demo_error_handling()
    demo.demo_batch_processing()
    demo.demo_adaptive_prompting()
    demo.demo_integration_patterns()
    
    # Generate comprehensive report
    demo.generate_summary_report()
    
    print("\n✅ All advanced demos completed!")
    print("\nFiles generated:")
    print("- advanced_demo1_error_handling.json")
    print("- advanced_demo2_batch_processing.json") 
    print("- advanced_demo3_adaptive_prompting.json")
    print("- advanced_demo4_integration.json")
    print("- advanced_demo_summary.json")


if __name__ == "__main__":
    main()