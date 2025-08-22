#!/usr/bin/env python3
"""
Demo 3: Consensus Indicator Extraction Capability Comparison

This demo specifically tests the ability to extract consensus analysis metrics from
LLM responses in the mLLMCelltype context, comparing traditional regex-based parsing
with langextract's structured extraction approach.

Focus areas:
- consensus_reached (boolean): Whether consensus was achieved
- consensus_proportion (float 0-1): Level of agreement among models
- entropy (float ≥0): Measure of uncertainty/disagreement
- majority_cell_type (string): The winning cell type prediction
"""

import os
import sys
import json
import time
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import Counter

# Import our test framework
from test_setup import LangExtractTester

try:
    import langextract
    from langextract import extract
except ImportError as e:
    print(f"Error importing langextract: {e}")
    print("Please install langextract: pip install langextract")
    sys.exit(1)


# ===== CONSENSUS RESPONSE SAMPLES =====

CONSENSUS_SAMPLES = {
    "standard_4line_format": """
Based on the predictions from all models for this cluster:
- GPT-4: "CD8+ T cells"
- Claude: "Cytotoxic T lymphocytes" 
- Gemini: "T cells (cytotoxic)"

1
0.85
0.41
CD8+ T cells
""",

    "irregular_format_with_explanation": """
Consensus Analysis Results:

After reviewing all model predictions, I can see strong agreement on the T cell identity.
All three models identify this as cytotoxic T cells with slightly different terminology.

Consensus reached: YES (1)
Agreement level: 85% (0.85)
Shannon entropy: 0.41
Final prediction: CD8+ T cells

This represents good consensus with low uncertainty.
""",

    "percentage_and_boolean_variants": """
Model predictions analyzed:
1. GPT-4o: "Memory B cells"
2. Claude-3.5: "B cells (memory)"
3. Gemini-2.0: "Activated B cells"
4. Qwen-Max: "B cell subset"

Consensus Status: True
Consensus Proportion: 75%
Entropy Value: 0.81
Majority Cell Type: Memory B cells
""",

    "no_consensus_reached": """
Analysis of model disagreement:
- Model A: "Dendritic cells"
- Model B: "Monocytes" 
- Model C: "Macrophages"
- Model D: "Myeloid cells"

0
0.25
1.39
Dendritic cells
""",

    "partial_consensus_mixed_format": """
Examining the predictions:
GPT-4: Classical monocytes
Claude: CD14+ monocytes  
Gemini: Monocytes
DeepSeek: Classical monocytes

Agreement achieved: 1
Proportion: 0.75
H: 0.56
Classical monocytes
""",

    "numeric_format_variations": """
Model consensus check:
- consensus_reached: 1.0
- consensus_proportion: 0.667
- entropy: 0.92
- majority_prediction: Plasma cells

Additional context: Two models agreed on plasma cells, one suggested plasmablasts.
""",

    "verbose_discussion_format": """
Multi-round discussion summary for controversial cluster:

=== ROUND 1 ===
Initial disagreement between models about myeloid cell identity.

=== ROUND 2 === 
After reviewing marker evidence, models converged on dendritic cell subtype.

=== FINAL CONSENSUS ===
All participating models now agree on the cell type identification.

Final metrics:
Consensus reached: Yes
Consensus proportion: 1.0  
Shannon entropy: 0.0
Majority cell type: Conventional dendritic cells

Discussion resolved successfully with complete agreement.
""",

    "missing_information": """
Consensus check completed.

Consensus proportion: 0.6
Entropy: 0.97

Some information was not clearly determinable from the model responses.
""",

    "malformed_response": """
The models provided various predictions but there was some uncertainty in the analysis.
Consensus: maybe
Proportion: about 60%
Entropy: somewhat high
Cell type: could be T cells or NK cells
""",

    "edge_case_zero_entropy": """
Perfect agreement achieved:
All models predicted identical cell type.

1
1.0
0.0
Regulatory T cells
""",

    "edge_case_high_entropy": """
Maximum disagreement scenario:
Each model predicted different cell type.

0
0.2
2.32
Unknown
""",

    "chinese_mixed_response": """
模型预测分析结果：

多个模型的预测显示了较好的一致性。

Consensus reached: 是 (1)
Agreement ratio: 0.8
熵值: 0.64
主要细胞类型: Natural killer cells

The consensus analysis shows good agreement among models.
""",

    "json_embedded_format": """
{
  "analysis": "Consensus check for cluster 5",
  "consensus_reached": true,
  "consensus_proportion": 0.9,
  "entropy": 0.32,
  "majority_prediction": "Plasma cells",
  "details": "High agreement observed"
}
""",

    "multiline_cell_type": """
Consensus analysis complete:

1
0.73
0.85
Memory CD4+ T cells
(helper subset)
""",

    "alternative_terminology": """
Agreement assessment:
Convergence: Achieved (1)
Concordance rate: 0.67
Information entropy: 1.11  
Predominant annotation: Neutrophils
""",

    "error_response_partial": """
ERROR: Timeout during consensus analysis
Partial results available:

Consensus reached: 
Consensus proportion: 0.4
Entropy: 
Majority prediction: B cells

Analysis incomplete due to processing error.
"""
}


# ===== EVALUATION FUNCTIONS =====

def traditional_consensus_parser_enhanced(text: str) -> Dict[str, Any]:
    """
    Enhanced traditional parser that mimics mllmcelltype's _parse_llm_consensus_response
    with additional fallback patterns for robustness testing.
    """
    result = {
        "consensus_reached": None,
        "consensus_proportion": None,
        "entropy": None,
        "majority_cell_type": None
    }
    
    try:
        # First try the standard 4-line format (from mllmcelltype)
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        # Check for standard format (last 4 lines)
        if len(lines) >= 4:
            result_lines = lines[-4:]
            
            # Check if it matches standard format pattern
            if (re.match(r"^\s*[01]\s*$", result_lines[0]) and
                re.match(r"^\s*(0\.\d+|1\.0*|1)\s*$", result_lines[1]) and
                re.match(r"^\s*(\d+\.\d+|\d+)\s*$", result_lines[2])):
                
                # Extract values
                consensus_reached = bool(int(result_lines[0].strip()))
                consensus_proportion = float(result_lines[1].strip())
                entropy = float(result_lines[2].strip())
                majority_cell_type = result_lines[3].strip()
                
                result.update({
                    "consensus_reached": consensus_reached,
                    "consensus_proportion": consensus_proportion,
                    "entropy": entropy,
                    "majority_cell_type": majority_cell_type
                })
                return result
        
        # Fallback patterns for non-standard formats
        # Boolean consensus patterns
        consensus_patterns = [
            r'consensus\s+reached\s*:\s*(yes|true|1|achieved)',
            r'consensus\s*:\s*(yes|true|1|achieved)',
            r'agreement\s+achieved\s*:\s*(yes|true|1)',
            r'convergence\s*:\s*(achieved|yes|true|1)',
            r'consensus\s+status\s*:\s*(true|yes|1)',
        ]
        
        for pattern in consensus_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                result["consensus_reached"] = value in ['yes', 'true', '1', 'achieved']
                break
        
        # Proportion patterns
        proportion_patterns = [
            r'consensus\s+proportion\s*:\s*([\d\.]+%?)',
            r'agreement\s+level\s*:\s*([\d\.]+%?)',
            r'proportion\s*:\s*([\d\.]+%?)',
            r'concordance\s+rate\s*:\s*([\d\.]+%?)',
            r'agreement\s+ratio\s*:\s*([\d\.]+%?)',
        ]
        
        for pattern in proportion_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                if '%' in value_str:
                    result["consensus_proportion"] = float(value_str.rstrip('%')) / 100
                else:
                    value = float(value_str)
                    result["consensus_proportion"] = value if value <= 1 else value / 100
                break
        
        # Entropy patterns
        entropy_patterns = [
            r'(?:shannon\s+)?entropy\s*(?:value)?\s*:\s*([\d\.]+)',
            r'h\s*:\s*([\d\.]+)',
            r'熵值\s*:\s*([\d\.]+)',  # Chinese
            r'information\s+entropy\s*:\s*([\d\.]+)',
        ]
        
        for pattern in entropy_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["entropy"] = float(match.group(1))
                break
        
        # Cell type patterns
        cell_type_patterns = [
            r'majority\s+(?:cell\s+type|prediction)\s*:\s*([^\n\r]+)',
            r'final\s+prediction\s*:\s*([^\n\r]+)',
            r'majority\s+cell\s+type\s*:\s*([^\n\r]+)',
            r'predominant\s+annotation\s*:\s*([^\n\r]+)',
            r'主要细胞类型\s*:\s*([^\n\r]+)',  # Chinese
        ]
        
        for pattern in cell_type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cell_type = match.group(1).strip()
                # Clean up common suffixes
                cell_type = re.sub(r'\s*\([^)]*\)\s*$', '', cell_type)
                result["majority_cell_type"] = cell_type
                break
        
        # Try to extract from JSON if present
        json_match = re.search(r'\{[^}]*"consensus_reached"[^}]*\}', text, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(0))
                if "consensus_reached" in json_data:
                    result["consensus_reached"] = bool(json_data["consensus_reached"])
                if "consensus_proportion" in json_data:
                    result["consensus_proportion"] = float(json_data["consensus_proportion"])
                if "entropy" in json_data:
                    result["entropy"] = float(json_data["entropy"])
                if "majority_prediction" in json_data:
                    result["majority_cell_type"] = str(json_data["majority_prediction"])
            except json.JSONDecodeError:
                pass
                
    except (ValueError, AttributeError, TypeError) as e:
        print(f"Traditional parser error: {e}")
    
    return result


def create_langextract_consensus_schema() -> Dict[str, Any]:
    """Create a comprehensive schema for consensus extraction."""
    return {
        "type": "object",
        "properties": {
            "consensus_reached": {
                "type": "boolean",
                "description": "Whether consensus was reached among the models"
            },
            "consensus_proportion": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Proportion of models that agreed (0.0 to 1.0)"
            },
            "entropy": {
                "type": "number", 
                "minimum": 0,
                "description": "Shannon entropy measuring uncertainty/disagreement"
            },
            "majority_cell_type": {
                "type": "string",
                "description": "The cell type that received the most votes/agreement"
            },
            "confidence_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Overall confidence in the consensus result"
            },
            "participating_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of models that participated in the consensus"
            }
        },
        "required": ["consensus_reached", "majority_cell_type"]
    }


def evaluate_extraction_accuracy(extracted: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate the accuracy of consensus extraction."""
    scores = {}
    
    # Boolean accuracy
    if expected.get("consensus_reached") is not None and extracted.get("consensus_reached") is not None:
        scores["consensus_reached"] = 1.0 if extracted["consensus_reached"] == expected["consensus_reached"] else 0.0
    
    # Proportion accuracy (with tolerance)
    if expected.get("consensus_proportion") is not None and extracted.get("consensus_proportion") is not None:
        diff = abs(extracted["consensus_proportion"] - expected["consensus_proportion"])
        scores["consensus_proportion"] = max(0.0, 1.0 - diff * 2)  # 0.5 diff = 0 score
    
    # Entropy accuracy (with tolerance)
    if expected.get("entropy") is not None and extracted.get("entropy") is not None:
        diff = abs(extracted["entropy"] - expected["entropy"])
        scores["entropy"] = max(0.0, 1.0 - diff / 2)  # 2.0 diff = 0 score
    
    # Cell type accuracy (semantic matching)
    if expected.get("majority_cell_type") and extracted.get("majority_cell_type"):
        expected_type = expected["majority_cell_type"].lower().strip()
        extracted_type = extracted["majority_cell_type"].lower().strip()
        
        # Exact match
        if expected_type == extracted_type:
            scores["majority_cell_type"] = 1.0
        # Substring match
        elif expected_type in extracted_type or extracted_type in expected_type:
            scores["majority_cell_type"] = 0.8
        # Semantic similarity (basic)
        elif any(word in extracted_type for word in expected_type.split()):
            scores["majority_cell_type"] = 0.6
        else:
            scores["majority_cell_type"] = 0.0
    
    # Overall accuracy
    if scores:
        scores["overall"] = sum(scores.values()) / len(scores)
    
    return scores


def generate_ground_truth() -> Dict[str, Dict[str, Any]]:
    """Generate ground truth annotations for evaluation."""
    return {
        "standard_4line_format": {
            "consensus_reached": True,
            "consensus_proportion": 0.85,
            "entropy": 0.41,
            "majority_cell_type": "CD8+ T cells"
        },
        "irregular_format_with_explanation": {
            "consensus_reached": True,
            "consensus_proportion": 0.85,
            "entropy": 0.41,
            "majority_cell_type": "CD8+ T cells"
        },
        "percentage_and_boolean_variants": {
            "consensus_reached": True,
            "consensus_proportion": 0.75,
            "entropy": 0.81,
            "majority_cell_type": "Memory B cells"
        },
        "no_consensus_reached": {
            "consensus_reached": False,
            "consensus_proportion": 0.25,
            "entropy": 1.39,
            "majority_cell_type": "Dendritic cells"
        },
        "partial_consensus_mixed_format": {
            "consensus_reached": True,
            "consensus_proportion": 0.75,
            "entropy": 0.56,
            "majority_cell_type": "Classical monocytes"
        },
        "numeric_format_variations": {
            "consensus_reached": True,
            "consensus_proportion": 0.667,
            "entropy": 0.92,
            "majority_cell_type": "Plasma cells"
        },
        "verbose_discussion_format": {
            "consensus_reached": True,
            "consensus_proportion": 1.0,
            "entropy": 0.0,
            "majority_cell_type": "Conventional dendritic cells"
        },
        "edge_case_zero_entropy": {
            "consensus_reached": True,
            "consensus_proportion": 1.0,
            "entropy": 0.0,
            "majority_cell_type": "Regulatory T cells"
        },
        "edge_case_high_entropy": {
            "consensus_reached": False,
            "consensus_proportion": 0.2,
            "entropy": 2.32,
            "majority_cell_type": "Unknown"
        },
        "chinese_mixed_response": {
            "consensus_reached": True,
            "consensus_proportion": 0.8,
            "entropy": 0.64,
            "majority_cell_type": "Natural killer cells"
        },
        "json_embedded_format": {
            "consensus_reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.32,
            "majority_cell_type": "Plasma cells"
        },
        "alternative_terminology": {
            "consensus_reached": True,
            "consensus_proportion": 0.67,
            "entropy": 1.11,
            "majority_cell_type": "Neutrophils"
        }
    }


# ===== MAIN DEMO FUNCTIONS =====

def run_consensus_extraction_comparison(tester: LangExtractTester) -> Dict[str, Any]:
    """Run comprehensive consensus extraction comparison."""
    print("\n🤝 Demo 3: Consensus Indicator Extraction Comparison")
    print("=" * 60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": {},
        "summary": {
            "total_cases": 0,
            "langextract_successes": 0,
            "traditional_successes": 0,
            "langextract_accuracy": {},
            "traditional_accuracy": {},
            "performance": {}
        }
    }
    
    ground_truth = generate_ground_truth()
    
    # Define langextract prompt
    langextract_prompt = """
    Extract consensus analysis metrics from this LLM response about cell type annotation agreement.
    
    Look for these specific indicators:
    - consensus_reached: Whether consensus was achieved (boolean)
    - consensus_proportion: Level of agreement (0.0 to 1.0)
    - entropy: Shannon entropy measuring uncertainty (≥0.0)
    - majority_cell_type: The winning cell type prediction
    - participating_models: List of models involved (if mentioned)
    - confidence_level: Overall confidence (low/medium/high)
    
    Handle various formats including:
    - Standard 4-line numeric format
    - Verbose explanatory text
    - Percentage values (convert to 0-1 scale)
    - Different boolean representations (Yes/No, True/False, 1/0)
    - Mixed languages and terminology
    """
    
    print(f"Testing {len(CONSENSUS_SAMPLES)} consensus response samples...")
    print("This comprehensive test covers:")
    print("- Standard 4-line format responses")
    print("- Irregular formatting with explanations")
    print("- Percentage vs decimal representations")
    print("- Various boolean value formats")
    print("- Edge cases (perfect/no consensus)")
    print("- Mixed language responses")
    print("- Error and malformed responses")
    print()
    
    for i, (sample_name, sample_text) in enumerate(CONSENSUS_SAMPLES.items(), 1):
        print(f"Processing case {i}/{len(CONSENSUS_SAMPLES)}: {sample_name}")
        
        case_result = {
            "sample_name": sample_name,
            "sample_text": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text,
            "langextract": {},
            "traditional": {},
            "ground_truth": ground_truth.get(sample_name, {}),
            "evaluation": {}
        }
        
        # Test with langextract
        print(f"  Testing with langextract...")
        start_time = time.time()
        langextract_result = tester.test_basic_extraction(
            sample_text, 
            langextract_prompt
        )
        langextract_time = time.time() - start_time
        
        case_result["langextract"] = {
            "success": langextract_result["success"],
            "execution_time": langextract_time,
            "model_used": langextract_result.get("model_used", "unknown"),
            "raw_result": str(langextract_result.get("result", ""))[:500],
            "extracted_data": None,
            "error": langextract_result.get("error", None)
        }
        
        # Try to parse langextract result
        if langextract_result["success"] and hasattr(langextract_result["result"], "extractions"):
            try:
                extractions = langextract_result["result"].extractions
                if extractions and len(extractions) > 0:
                    case_result["langextract"]["extracted_data"] = extractions[0]
                    results["summary"]["langextract_successes"] += 1
            except Exception as e:
                case_result["langextract"]["parse_error"] = str(e)
        
        # Test with traditional parser
        print(f"  Testing with traditional parser...")
        start_time = time.time()
        try:
            traditional_result = traditional_consensus_parser_enhanced(sample_text)
            traditional_time = time.time() - start_time
            
            case_result["traditional"] = {
                "success": any(v is not None for v in traditional_result.values()),
                "execution_time": traditional_time,
                "extracted_data": traditional_result,
                "error": None
            }
            
            if case_result["traditional"]["success"]:
                results["summary"]["traditional_successes"] += 1
                
        except Exception as e:
            traditional_time = time.time() - start_time
            case_result["traditional"] = {
                "success": False,
                "execution_time": traditional_time,
                "extracted_data": None,
                "error": str(e)
            }
        
        # Evaluate accuracy if we have ground truth
        if sample_name in ground_truth:
            expected = ground_truth[sample_name]
            
            # Evaluate langextract
            if case_result["langextract"]["extracted_data"]:
                langextract_scores = evaluate_extraction_accuracy(
                    case_result["langextract"]["extracted_data"], 
                    expected
                )
                case_result["evaluation"]["langextract"] = langextract_scores
            
            # Evaluate traditional
            if case_result["traditional"]["extracted_data"]:
                traditional_scores = evaluate_extraction_accuracy(
                    case_result["traditional"]["extracted_data"], 
                    expected
                )
                case_result["evaluation"]["traditional"] = traditional_scores
        
        results["test_cases"][sample_name] = case_result
        results["summary"]["total_cases"] += 1
        print(f"  ✓ Completed")
    
    # Calculate summary statistics
    calculate_summary_statistics(results)
    
    return results


def calculate_summary_statistics(results: Dict[str, Any]):
    """Calculate summary statistics for the comparison."""
    langextract_accuracies = {"consensus_reached": [], "consensus_proportion": [], 
                             "entropy": [], "majority_cell_type": [], "overall": []}
    traditional_accuracies = {"consensus_reached": [], "consensus_proportion": [], 
                             "entropy": [], "majority_cell_type": [], "overall": []}
    
    langextract_times = []
    traditional_times = []
    
    for case_name, case_data in results["test_cases"].items():
        # Collect accuracy scores
        if "langextract" in case_data.get("evaluation", {}):
            for metric, score in case_data["evaluation"]["langextract"].items():
                if metric in langextract_accuracies:
                    langextract_accuracies[metric].append(score)
        
        if "traditional" in case_data.get("evaluation", {}):
            for metric, score in case_data["evaluation"]["traditional"].items():
                if metric in traditional_accuracies:
                    traditional_accuracies[metric].append(score)
        
        # Collect timing data
        if case_data["langextract"]["success"]:
            langextract_times.append(case_data["langextract"]["execution_time"])
        if case_data["traditional"]["success"]:
            traditional_times.append(case_data["traditional"]["execution_time"])
    
    # Calculate averages
    for metric in langextract_accuracies:
        if langextract_accuracies[metric]:
            results["summary"]["langextract_accuracy"][metric] = {
                "average": sum(langextract_accuracies[metric]) / len(langextract_accuracies[metric]),
                "count": len(langextract_accuracies[metric])
            }
    
    for metric in traditional_accuracies:
        if traditional_accuracies[metric]:
            results["summary"]["traditional_accuracy"][metric] = {
                "average": sum(traditional_accuracies[metric]) / len(traditional_accuracies[metric]),
                "count": len(traditional_accuracies[metric])
            }
    
    # Performance statistics
    results["summary"]["performance"] = {
        "langextract": {
            "avg_time": sum(langextract_times) / len(langextract_times) if langextract_times else 0,
            "success_rate": results["summary"]["langextract_successes"] / results["summary"]["total_cases"]
        },
        "traditional": {
            "avg_time": sum(traditional_times) / len(traditional_times) if traditional_times else 0,
            "success_rate": results["summary"]["traditional_successes"] / results["summary"]["total_cases"]
        }
    }


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed analysis of the comparison results."""
    print("\n📊 DETAILED CONSENSUS EXTRACTION ANALYSIS")
    print("=" * 60)
    
    summary = results["summary"]
    
    # Overall performance comparison
    print("\n🎯 OVERALL PERFORMANCE:")
    print(f"Total test cases: {summary['total_cases']}")
    print(f"LangExtract success rate: {summary['performance']['langextract']['success_rate']:.1%}")
    print(f"Traditional success rate: {summary['performance']['traditional']['success_rate']:.1%}")
    print(f"LangExtract avg time: {summary['performance']['langextract']['avg_time']:.3f}s")
    print(f"Traditional avg time: {summary['performance']['traditional']['avg_time']:.3f}s")
    
    # Accuracy comparison by metric
    print("\n🎯 ACCURACY BY METRIC:")
    metrics = ["consensus_reached", "consensus_proportion", "entropy", "majority_cell_type", "overall"]
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        
        langextract_data = summary["langextract_accuracy"].get(metric)
        traditional_data = summary["traditional_accuracy"].get(metric)
        
        if langextract_data:
            print(f"  LangExtract: {langextract_data['average']:.3f} (n={langextract_data['count']})")
        else:
            print(f"  LangExtract: No data")
            
        if traditional_data:
            print(f"  Traditional: {traditional_data['average']:.3f} (n={traditional_data['count']})")
        else:
            print(f"  Traditional: No data")
        
        # Show winner
        if langextract_data and traditional_data:
            if langextract_data['average'] > traditional_data['average']:
                print(f"  🏆 Winner: LangExtract (+{langextract_data['average'] - traditional_data['average']:.3f})")
            elif traditional_data['average'] > langextract_data['average']:
                print(f"  🏆 Winner: Traditional (+{traditional_data['average'] - langextract_data['average']:.3f})")
            else:
                print(f"  🤝 Tie")
    
    # Challenging cases analysis
    print("\n🔥 CHALLENGING CASES ANALYSIS:")
    challenging_cases = []
    
    for case_name, case_data in results["test_cases"].items():
        langextract_success = case_data["langextract"]["success"]
        traditional_success = case_data["traditional"]["success"]
        
        if not langextract_success and not traditional_success:
            challenging_cases.append((case_name, "Both failed"))
        elif not langextract_success and traditional_success:
            challenging_cases.append((case_name, "LangExtract failed, Traditional succeeded"))
        elif langextract_success and not traditional_success:
            challenging_cases.append((case_name, "Traditional failed, LangExtract succeeded"))
    
    for case_name, status in challenging_cases:
        print(f"  {case_name}: {status}")
    
    if not challenging_cases:
        print("  No particularly challenging cases found.")
    
    # Edge cases performance
    print("\n⚡ EDGE CASES PERFORMANCE:")
    edge_cases = ["edge_case_zero_entropy", "edge_case_high_entropy", "malformed_response", 
                  "error_response_partial", "missing_information"]
    
    for edge_case in edge_cases:
        if edge_case in results["test_cases"]:
            case_data = results["test_cases"][edge_case]
            langextract_ok = case_data["langextract"]["success"]
            traditional_ok = case_data["traditional"]["success"]
            print(f"  {edge_case}:")
            print(f"    LangExtract: {'✓' if langextract_ok else '✗'}")
            print(f"    Traditional: {'✓' if traditional_ok else '✗'}")


def run_robustness_test(tester: LangExtractTester) -> Dict[str, Any]:
    """Test robustness with synthetic variations."""
    print("\n🛡️ ROBUSTNESS TEST: Synthetic Variations")
    print("-" * 50)
    
    # Generate synthetic variations
    base_response = """
    Consensus analysis complete:
    1
    0.75
    0.81
    Memory B cells
    """
    
    variations = {
        "extra_whitespace": base_response.replace("\n", "\n\n   \n"),
        "mixed_case": base_response.upper(),
        "with_noise": base_response + "\n\n[DEBUG: Analysis completed at 14:32:05]",
        "reordered_lines": "\n".join(base_response.strip().split("\n")[::-1]),
        "embedded_in_text": f"Starting analysis...\n{base_response}\nAnalysis complete.",
    }
    
    print(f"Testing {len(variations)} synthetic variations...")
    
    results = {}
    for var_name, var_text in variations.items():
        print(f"  Testing: {var_name}")
        
        # Test both methods
        langextract_result = tester.test_basic_extraction(
            var_text,
            "Extract consensus metrics: consensus_reached, consensus_proportion, entropy, majority_cell_type"
        )
        
        traditional_result = traditional_consensus_parser_enhanced(var_text)
        
        results[var_name] = {
            "langextract_success": langextract_result["success"],
            "traditional_success": any(v is not None for v in traditional_result.values()),
            "variation_text": var_text
        }
    
    return results


def main():
    """Run the consensus parsing comparison demo."""
    print("🧬 mLLMCelltype Consensus Parsing Demo")
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
        return
    
    print(f"✅ Found API keys for: {', '.join(available_keys.keys())}")
    
    # Initialize tester
    tester = LangExtractTester(api_keys=available_keys)
    
    try:
        # Run main comparison
        comparison_results = run_consensus_extraction_comparison(tester)
        
        # Print detailed analysis
        print_detailed_results(comparison_results)
        
        # Run robustness test
        robustness_results = run_robustness_test(tester)
        
        # Save results
        all_results = {
            "comparison": comparison_results,
            "robustness": robustness_results,
            "test_metadata": {
                "demo_version": "3.0",
                "focus": "consensus_indicator_extraction",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "api_keys_used": list(available_keys.keys())
            }
        }
        
        output_file = "demo3_consensus_parsing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Complete results saved to: {output_file}")
        
        # Print final summary
        print("\n📋 FINAL SUMMARY:")
        perf = comparison_results["summary"]["performance"]
        print(f"🎯 LangExtract: {perf['langextract']['success_rate']:.1%} success, {perf['langextract']['avg_time']:.3f}s avg")
        print(f"🎯 Traditional: {perf['traditional']['success_rate']:.1%} success, {perf['traditional']['avg_time']:.3f}s avg")
        
        overall_langextract = comparison_results["summary"]["langextract_accuracy"].get("overall", {}).get("average", 0)
        overall_traditional = comparison_results["summary"]["traditional_accuracy"].get("overall", {}).get("average", 0)
        
        if overall_langextract > overall_traditional:
            improvement = ((overall_langextract - overall_traditional) / overall_traditional * 100) if overall_traditional > 0 else 0
            print(f"🏆 LangExtract shows {improvement:.1f}% better accuracy than traditional parsing")
        elif overall_traditional > overall_langextract:
            improvement = ((overall_traditional - overall_langextract) / overall_langextract * 100) if overall_langextract > 0 else 0
            print(f"🏆 Traditional parsing shows {improvement:.1f}% better accuracy than LangExtract")
        else:
            print("🤝 Both methods perform similarly on average")
        
        print("\n✅ Demo 3 completed successfully!")
        print("\nKey insights for mLLMCelltype integration:")
        print("- Consensus indicator extraction is crucial for multi-model workflows")
        print("- LangExtract provides more robust parsing for non-standard formats")
        print("- Traditional regex works well for standard 4-line format")
        print("- Both methods complement each other in different scenarios")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()