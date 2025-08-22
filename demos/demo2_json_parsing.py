#!/usr/bin/env python3
"""
Demo 2: JSON格式修复能力对比测试

专门测试JSON解析和修复能力的对比分析：
- 测试各种格式错误的JSON响应样本，基于mllmcelltype实际遇到的问题
- 对比现有JSON修复逻辑 vs langextract的处理
- 评估成功率、准确性、稳定性
- 重点展示langextract在处理格式不规范JSON时的优势
"""

import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add the parent directory to the path to import mllmcelltype modules
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import langextract
    from langextract import extract
except ImportError as e:
    print(f"Error importing langextract: {e}")
    print("Please install langextract: pip install langextract")
    sys.exit(1)

# Import existing mllmcelltype utilities
try:
    from mllmcelltype.logger import write_log
    from mllmcelltype.utils import clean_annotation, format_results
except ImportError as e:
    print(f"Error importing mllmcelltype modules: {e}")
    print("Please ensure mllmcelltype is properly installed")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONParsingComparison:
    """Comprehensive JSON parsing and fixing capability comparison."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """Initialize the comparison tool."""
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        self.model = model
        self.setup_environment()

        # Test results storage
        self.test_results = {
            "metadata": {
                "timestamp": time.time(),
                "model_used": model,
                "test_count": 0,
                "langextract_version": getattr(langextract, "__version__", "unknown"),
            },
            "individual_tests": [],
            "summary": {
                "traditional_success_rate": 0.0,
                "langextract_success_rate": 0.0,
                "traditional_avg_time": 0.0,
                "langextract_avg_time": 0.0,
                "data_quality_scores": {},
                "error_recovery_scores": {},
            },
        }

    def setup_environment(self):
        """Setup environment for langextract."""
        if self.api_key:
            os.environ["LANGEXTRACT_API_KEY"] = self.api_key
            logger.info(f"Configured langextract with model: {self.model}")
        else:
            logger.warning("No API key found - langextract tests will fail")

    def create_problematic_json_samples(self) -> Dict[str, Dict[str, Any]]:
        """Create various JSON samples with format errors based on real mllmcelltype issues."""
        samples = {
            "missing_commas": {
                "name": "JSON缺少逗号",
                "description": "在对象属性之间缺少逗号分隔符",
                "raw_response": """```json
{
  "annotations": [
    {
      "cluster": "0"
      "cell_type": "T cells"
      "confidence": "high"
    }
    {
      "cluster": "1"
      "cell_type": "B cells"
      "confidence": "medium"
    }
  ]
}
```""",
                "expected_clusters": ["0", "1"],
                "difficulty": "medium",
            },
            "trailing_commas": {
                "name": "JSON多余逗号",
                "description": "数组和对象末尾有多余的逗号",
                "raw_response": """```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": "CD8+ T cells",
      "key_markers": ["CD8A", "CD8B", "GZMB",],
      "confidence": "high",
    },
    {
      "cluster": "1", 
      "cell_type": "B cells",
      "key_markers": ["CD19", "MS4A1",],
      "confidence": "medium",
    },
  ],
}
```""",
                "expected_clusters": ["0", "1"],
                "difficulty": "easy",
            },
            "quote_mismatches": {
                "name": "JSON引号不匹配",
                "description": "字符串值的引号不匹配或使用了错误的引号类型",
                "raw_response": """```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": 'CD4+ T cells",
      "confidence": "high'
    },
    {
      'cluster': "1",
      "cell_type": "Natural Killer cells",
      "confidence": 'medium"
    }
  ]
}
```""",
                "expected_clusters": ["0", "1"],
                "difficulty": "hard",
            },
            "nested_structure_errors": {
                "name": "JSON嵌套结构错误",
                "description": "嵌套对象和数组的括号不匹配",
                "raw_response": """```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": "Monocytes",
      "subtypes": {
        "classical": {"proportion": 0.7, "markers": ["CD14"]
        "intermediate": {"proportion": 0.3, "markers": ["CD14", "CD16"]}
      },
      "confidence": "high"
    }
    {
      "cluster": "1",
      "cell_type": "Dendritic cells",
      "subtypes": [
        {"type": "cDC1", "markers": ["CLEC9A"
        {"type": "cDC2", "markers": ["CD1C", "FCER1A"]},
      ],
      "confidence": "medium"
    ]
  ]
}
```""",
                "expected_clusters": ["0", "1"],
                "difficulty": "very_hard",
            },
            "markdown_mixed": {
                "name": "JSON混合markdown标记",
                "description": "JSON中混合了markdown格式标记",
                "raw_response": """The analysis results are:

```json
{
  **"annotations"**: [
    {
      "cluster": "0",
      "cell_type": "_T cells_", // CD3+ T lymphocytes
      "confidence": **"high"**
    },
    {
      "cluster": "1", 
      "cell_type": "*B cells*", /* B lymphocytes with high CD19 */
      "confidence": "medium"
    }
  ]
}
```

*Note: High confidence based on marker expression*""",
                "expected_clusters": ["0", "1"],
                "difficulty": "hard",
            },
            "incomplete_json": {
                "name": "JSON不完整结构",
                "description": "JSON响应被截断或不完整",
                "raw_response": '''Based on the marker genes, here are the annotations:

```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": "Plasma cells",
      "key_markers": ["IGHG1", "JCHAIN", "XBP1"],
      "confidence": "high"
    },
    {
      "cluster": "1",
      "cell_type": "Memory B cells",
      "key_markers": ["CD27", "IGHD"
      "confidence"''',
                "expected_clusters": ["0", "1"],
                "difficulty": "very_hard",
            },
            "malformed_values": {
                "name": "JSON值格式错误",
                "description": "JSON中的值不符合预期格式",
                "raw_response": """```json
{
  "annotations": [
    {
      "cluster": 0,  // Should be string
      "cell_type": "Neutrophils",
      "confidence": high,  // Should be quoted
      "markers": "S100A8,S100A9,FCGR3B"  // Should be array
    },
    {
      "cluster": "1",
      "cell_type": ["Eosinophils"],  // Should be string
      "confidence": 0.85,  // Should be string category
      "markers": ["IL5RA", "SIGLEC8"]
    }
  ]
}
```""",
                "expected_clusters": ["0", "1"],
                "difficulty": "medium",
            },
            "real_world_complex": {
                "name": "真实复杂JSON错误",
                "description": "基于实际mllmcelltype输出的复杂格式错误",
                "raw_response": """Looking at the differential gene expression, I can provide these annotations:

```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": "CD8+ T cells", 
      "reasoning": "High expression of CD8A, CD8B, and cytotoxic markers like GZMB and PRF1",
      "key_markers": ["CD8A", "CD8B", "GZMB", "PRF1"],
      "confidence": "high",
      "subtype_analysis": {
        "naive_proportion": 0.3,
        "effector_proportion": 0.5, 
        "memory_proportion": 0.2,
      }
    },
    {
      "cluster": "1",
      "cell_type": "Classical monocytes",
      "reasoning": "Strong CD14 expression with low CD16, typical inflammatory signature"
      "key_markers": ["CD14", "S100A8", "S100A9", "LYZ"],
      "confidence": "high"
      "activation_state": "resting",
    }
    {
      "cluster": "2",
      "cell_type": "Natural Killer cells"
      "reasoning": "Expression of GNLY, NKG7, KLRD1 and other NK-specific markers",
      "key_markers": ["GNLY", "NKG7", "KLRD1", "NCAM1"]
      "confidence": "high",
      "cytotoxicity": "activated"
    },
  ],
  "total_clusters_analyzed": 3,
  "tissue_context": "PBMC",
  "method": "differential_gene_expression"
}
```

Additional notes: The analysis was performed using...""",
                "expected_clusters": ["0", "1", "2"],
                "difficulty": "very_hard",
            },
        }

        return samples

    def traditional_json_parser(
        self, raw_response: str, expected_clusters: List[str]
    ) -> Dict[str, Any]:
        """Use mllmcelltype's existing JSON fixing logic."""
        start_time = time.time()

        try:
            # Use the existing format_results function
            # First, clean the response text
            lines = raw_response.split("\n")
            clean_lines = [line.strip() for line in lines if line.strip()]

            # Try to call the existing format_results function
            result = format_results(clean_lines, expected_clusters)

            execution_time = time.time() - start_time

            # Validate the result
            success = (
                isinstance(result, dict)
                and len(result) >= len(expected_clusters)
                and all(cluster in result for cluster in expected_clusters)
            )

            return {
                "success": success,
                "result": result,
                "execution_time": execution_time,
                "extracted_clusters": len(result) if isinstance(result, dict) else 0,
                "error": None,
                "method": "traditional_mllmcelltype",
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Traditional parsing failed: {str(e)}")
            return {
                "success": False,
                "result": None,
                "execution_time": execution_time,
                "extracted_clusters": 0,
                "error": str(e),
                "method": "traditional_mllmcelltype",
            }

    def langextract_json_parser(
        self, raw_response: str, expected_clusters: List[str]
    ) -> Dict[str, Any]:
        """Use langextract for structured JSON extraction."""
        start_time = time.time()

        try:
            if not self.api_key:
                raise ValueError("No API key available for langextract")

            # Define the extraction schema
            extraction_prompt = f"""
            Extract cell type annotations from this response. The response should contain annotations 
            for {len(expected_clusters)} clusters (clusters {', '.join(expected_clusters)}).
            
            Extract the following information for each cluster:
            - cluster_id: The cluster identifier (as string)
            - cell_type: The predicted cell type name
            - confidence: Confidence level if mentioned
            - key_markers: List of marker genes if mentioned
            
            Return as a structured JSON with an 'annotations' array containing objects with these fields.
            If the original JSON is malformed, fix it and extract the intended information.
            """

            result = extract(
                raw_response,
                extraction_prompt,
                model_id=self.model,
                api_key=self.api_key,
            )

            execution_time = time.time() - start_time

            # Validate the langextract result
            success = False
            extracted_clusters = 0
            formatted_result = {}

            if isinstance(result, dict) and "annotations" in result:
                annotations = result["annotations"]
                if isinstance(annotations, list):
                    for annotation in annotations:
                        if (
                            isinstance(annotation, dict)
                            and "cluster_id" in annotation
                            and "cell_type" in annotation
                        ):
                            cluster_id = str(annotation["cluster_id"])
                            formatted_result[cluster_id] = annotation["cell_type"]
                            extracted_clusters += 1

                    success = extracted_clusters >= len(expected_clusters) and all(
                        cluster in formatted_result for cluster in expected_clusters
                    )

            return {
                "success": success,
                "result": formatted_result,
                "raw_result": result,
                "execution_time": execution_time,
                "extracted_clusters": extracted_clusters,
                "error": None,
                "method": "langextract",
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Langextract parsing failed: {str(e)}")
            return {
                "success": False,
                "result": None,
                "raw_result": None,
                "execution_time": execution_time,
                "extracted_clusters": 0,
                "error": str(e),
                "method": "langextract",
            }

    def evaluate_data_quality(
        self,
        traditional_result: Dict[str, Any],
        langextract_result: Dict[str, Any],
        expected_clusters: List[str],
    ) -> Dict[str, Any]:
        """Evaluate the quality of extracted data."""
        evaluation = {
            "completeness": {},
            "accuracy": {},
            "consistency": {},
            "robustness": {},
        }

        # Completeness: How many expected clusters were extracted
        for method, result in [
            ("traditional", traditional_result),
            ("langextract", langextract_result),
        ]:
            if result["success"] and isinstance(result["result"], dict):
                found_clusters = set(result["result"].keys())
                expected_set = set(expected_clusters)
                completeness = len(found_clusters.intersection(expected_set)) / len(
                    expected_set
                )
                evaluation["completeness"][method] = completeness
            else:
                evaluation["completeness"][method] = 0.0

        # Accuracy: Quality of extracted cell type names (basic heuristics)
        for method, result in [
            ("traditional", traditional_result),
            ("langextract", langextract_result),
        ]:
            accuracy_score = 0.0
            if result["success"] and isinstance(result["result"], dict):
                valid_annotations = 0
                total_annotations = 0

                for cluster_id, cell_type in result["result"].items():
                    total_annotations += 1
                    if isinstance(cell_type, str) and len(cell_type.strip()) > 0:
                        clean_type = clean_annotation(cell_type)
                        if clean_type and clean_type.lower() not in [
                            "unknown",
                            "unclear",
                            "?",
                        ]:
                            valid_annotations += 1

                accuracy_score = (
                    valid_annotations / total_annotations
                    if total_annotations > 0
                    else 0.0
                )

            evaluation["accuracy"][method] = accuracy_score

        # Consistency: Format consistency of results
        for method, result in [
            ("traditional", traditional_result),
            ("langextract", langextract_result),
        ]:
            consistency_score = 1.0 if result["success"] else 0.0
            evaluation["consistency"][method] = consistency_score

        # Robustness: Error handling capability
        for method, result in [
            ("traditional", traditional_result),
            ("langextract", langextract_result),
        ]:
            if result["success"]:
                robustness_score = 1.0
            elif result["error"] and "timeout" not in result["error"].lower():
                robustness_score = 0.5  # Failed but handled gracefully
            else:
                robustness_score = 0.0

            evaluation["robustness"][method] = robustness_score

        return evaluation

    def run_single_test(
        self, sample_name: str, sample_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single JSON parsing comparison test."""
        logger.info(f"Testing: {sample_name} - {sample_data.get('description', '')}")

        raw_response = sample_data["raw_response"]
        expected_clusters = sample_data["expected_clusters"]

        # Run traditional parsing
        traditional_result = self.traditional_json_parser(
            raw_response, expected_clusters
        )

        # Run langextract parsing
        langextract_result = self.langextract_json_parser(
            raw_response, expected_clusters
        )

        # Evaluate data quality
        quality_evaluation = self.evaluate_data_quality(
            traditional_result, langextract_result, expected_clusters
        )

        test_result = {
            "sample_name": sample_name,
            "sample_info": {
                "description": sample_data.get("description", ""),
                "difficulty": sample_data.get("difficulty", "unknown"),
                "expected_clusters": expected_clusters,
            },
            "traditional_result": traditional_result,
            "langextract_result": langextract_result,
            "quality_evaluation": quality_evaluation,
            "winner": None,
            "performance_comparison": {},
        }

        # Determine winner based on success and quality
        traditional_score = (
            float(traditional_result["success"]) * 0.4
            + quality_evaluation["completeness"].get("traditional", 0) * 0.3
            + quality_evaluation["accuracy"].get("traditional", 0) * 0.3
        )

        langextract_score = (
            float(langextract_result["success"]) * 0.4
            + quality_evaluation["completeness"].get("langextract", 0) * 0.3
            + quality_evaluation["accuracy"].get("langextract", 0) * 0.3
        )

        if langextract_score > traditional_score:
            test_result["winner"] = "langextract"
        elif traditional_score > langextract_score:
            test_result["winner"] = "traditional"
        else:
            test_result["winner"] = "tie"

        # Performance comparison
        test_result["performance_comparison"] = {
            "speed_advantage": "traditional"
            if traditional_result["execution_time"]
            < langextract_result["execution_time"]
            else "langextract",
            "time_difference": abs(
                traditional_result["execution_time"]
                - langextract_result["execution_time"]
            ),
            "traditional_time": traditional_result["execution_time"],
            "langextract_time": langextract_result["execution_time"],
        }

        logger.info(f"Test completed: {sample_name} - Winner: {test_result['winner']}")
        return test_result

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive JSON parsing comparison across all test samples."""
        logger.info("Starting comprehensive JSON parsing comparison...")

        samples = self.create_problematic_json_samples()

        # Run individual tests
        for sample_name, sample_data in samples.items():
            try:
                test_result = self.run_single_test(sample_name, sample_data)
                self.test_results["individual_tests"].append(test_result)
                self.test_results["metadata"]["test_count"] += 1
            except Exception as e:
                logger.error(f"Test failed for {sample_name}: {str(e)}")
                logger.error(traceback.format_exc())

        # Calculate summary statistics
        self._calculate_summary_statistics()

        return self.test_results

    def _calculate_summary_statistics(self):
        """Calculate summary statistics from individual test results."""
        if not self.test_results["individual_tests"]:
            return

        traditional_successes = 0
        langextract_successes = 0
        traditional_times = []
        langextract_times = []

        quality_scores = {
            "traditional": {
                "completeness": [],
                "accuracy": [],
                "consistency": [],
                "robustness": [],
            },
            "langextract": {
                "completeness": [],
                "accuracy": [],
                "consistency": [],
                "robustness": [],
            },
        }

        winners = {"traditional": 0, "langextract": 0, "tie": 0}

        for test in self.test_results["individual_tests"]:
            # Success rates
            if test["traditional_result"]["success"]:
                traditional_successes += 1
            if test["langextract_result"]["success"]:
                langextract_successes += 1

            # Execution times
            traditional_times.append(test["traditional_result"]["execution_time"])
            langextract_times.append(test["langextract_result"]["execution_time"])

            # Quality scores
            for metric in ["completeness", "accuracy", "consistency", "robustness"]:
                for method in ["traditional", "langextract"]:
                    score = test["quality_evaluation"][metric].get(method, 0.0)
                    quality_scores[method][metric].append(score)

            # Winners
            winners[test["winner"]] += 1

        total_tests = len(self.test_results["individual_tests"])

        # Update summary
        self.test_results["summary"].update(
            {
                "traditional_success_rate": traditional_successes / total_tests,
                "langextract_success_rate": langextract_successes / total_tests,
                "traditional_avg_time": sum(traditional_times) / len(traditional_times),
                "langextract_avg_time": sum(langextract_times) / len(langextract_times),
                "winner_distribution": winners,
                "quality_scores": {
                    method: {
                        metric: {
                            "average": sum(scores) / len(scores) if scores else 0.0,
                            "min": min(scores) if scores else 0.0,
                            "max": max(scores) if scores else 0.0,
                        }
                        for metric, scores in method_scores.items()
                    }
                    for method, method_scores in quality_scores.items()
                },
            }
        )

    def print_detailed_results(self):
        """Print detailed test results to console."""
        print("\n" + "=" * 80)
        print("JSON解析能力对比测试 - 详细结果")
        print("=" * 80)

        # Overall summary
        summary = self.test_results["summary"]
        print(f"\n📊 总体统计:")
        print(f"   测试样本数量: {self.test_results['metadata']['test_count']}")
        print(f"   传统解析成功率: {summary['traditional_success_rate']:.2%}")
        print(f"   LangExtract成功率: {summary['langextract_success_rate']:.2%}")
        print(f"   传统解析平均耗时: {summary['traditional_avg_time']:.3f}s")
        print(f"   LangExtract平均耗时: {summary['langextract_avg_time']:.3f}s")

        # Winner distribution
        winners = summary.get("winner_distribution", {})
        print(f"\n🏆 整体表现:")
        print(
            f"   LangExtract获胜: {winners.get('langextract', 0)}/{self.test_results['metadata']['test_count']}"
        )
        print(
            f"   传统方法获胜: {winners.get('traditional', 0)}/{self.test_results['metadata']['test_count']}"
        )
        print(
            f"   平局: {winners.get('tie', 0)}/{self.test_results['metadata']['test_count']}"
        )

        # Quality scores
        if "quality_scores" in summary:
            print(f"\n📈 质量评估 (平均分):")
            for method in ["traditional", "langextract"]:
                method_name = "传统方法" if method == "traditional" else "LangExtract"
                print(f"   {method_name}:")
                for metric, scores in summary["quality_scores"][method].items():
                    metric_name = {
                        "completeness": "完整性",
                        "accuracy": "准确性",
                        "consistency": "一致性",
                        "robustness": "健壮性",
                    }.get(metric, metric)
                    print(f"     {metric_name}: {scores['average']:.3f}")

        # Individual test results
        print(f"\n📋 单项测试详情:")
        for test in self.test_results["individual_tests"]:
            sample_name = test["sample_name"]
            winner = test["winner"]
            difficulty = test["sample_info"]["difficulty"]

            winner_emoji = (
                "🟢"
                if winner == "langextract"
                else "🔴"
                if winner == "traditional"
                else "🟡"
            )
            winner_text = (
                "LangExtract"
                if winner == "langextract"
                else "传统方法"
                if winner == "traditional"
                else "平局"
            )

            print(f"\n   {winner_emoji} {sample_name} (难度: {difficulty})")
            print(f"      描述: {test['sample_info']['description']}")
            print(f"      获胜者: {winner_text}")
            print(
                f"      传统方法: {'✓' if test['traditional_result']['success'] else '✗'} "
                f"({test['traditional_result']['execution_time']:.3f}s)"
            )
            print(
                f"      LangExtract: {'✓' if test['langextract_result']['success'] else '✗'} "
                f"({test['langextract_result']['execution_time']:.3f}s)"
            )

            if (
                not test["traditional_result"]["success"]
                and test["traditional_result"]["error"]
            ):
                print(
                    f"        传统方法错误: {test['traditional_result']['error'][:100]}..."
                )
            if (
                not test["langextract_result"]["success"]
                and test["langextract_result"]["error"]
            ):
                print(
                    f"        LangExtract错误: {test['langextract_result']['error'][:100]}..."
                )

    def save_results(self, filename: str = "json_parsing_comparison_results.json"):
        """Save detailed results to JSON file."""
        output_path = Path(__file__).parent / filename
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.test_results, f, indent=2, ensure_ascii=False, default=str
                )
            logger.info(f"Detailed results saved to: {output_path}")
            print(f"\n💾 详细结果已保存至: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"\n❌ 保存结果失败: {e}")

    def generate_markdown_report(self) -> str:
        """Generate a markdown report of the test results."""
        report = [
            "# JSON解析能力对比测试报告",
            "",
            f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_results['metadata']['timestamp']))}",
            f"**测试模型**: {self.test_results['metadata']['model_used']}",
            f"**测试样本数**: {self.test_results['metadata']['test_count']}",
            "",
            "## 测试目的",
            "",
            "本测试专门评估JSON解析和修复能力，对比mLLMCelltype现有的JSON修复逻辑与langextract的结构化提取能力。",
            "测试重点关注处理格式不规范JSON时的优势。",
            "",
            "## 测试样本",
            "",
            "基于mLLMCelltype实际遇到的问题，创建了以下类型的错误JSON样本：",
            "",
            "- 缺少逗号的JSON",
            "- 多余逗号的JSON",
            "- 引号不匹配的JSON",
            "- 嵌套结构错误的JSON",
            "- 混合了markdown标记的JSON",
            "- 不完整的JSON结构",
            "- 值格式错误的JSON",
            "- 真实复杂场景的JSON错误",
            "",
            "## 总体结果",
            "",
        ]

        summary = self.test_results["summary"]

        # Summary table
        report.extend(
            [
                "| 指标 | 传统方法 | LangExtract | 优势 |",
                "|------|----------|-------------|------|",
                f"| 成功率 | {summary['traditional_success_rate']:.2%} | {summary['langextract_success_rate']:.2%} | {'LangExtract' if summary['langextract_success_rate'] > summary['traditional_success_rate'] else '传统方法' if summary['traditional_success_rate'] > summary['langextract_success_rate'] else '相当'} |",
                f"| 平均耗时 | {summary['traditional_avg_time']:.3f}s | {summary['langextract_avg_time']:.3f}s | {'传统方法' if summary['traditional_avg_time'] < summary['langextract_avg_time'] else 'LangExtract'} |",
                "",
            ]
        )

        # Winner distribution
        winners = summary.get("winner_distribution", {})
        if winners:
            report.extend(
                [
                    "### 获胜分布",
                    "",
                    f"- **LangExtract获胜**: {winners.get('langextract', 0)}次",
                    f"- **传统方法获胜**: {winners.get('traditional', 0)}次",
                    f"- **平局**: {winners.get('tie', 0)}次",
                    "",
                ]
            )

        # Quality scores
        if "quality_scores" in summary:
            report.extend(
                [
                    "### 质量评估详情",
                    "",
                    "| 评估维度 | 传统方法 | LangExtract |",
                    "|----------|----------|-------------|",
                ]
            )

            for metric in ["completeness", "accuracy", "consistency", "robustness"]:
                metric_name = {
                    "completeness": "完整性",
                    "accuracy": "准确性",
                    "consistency": "一致性",
                    "robustness": "健壮性",
                }.get(metric, metric)

                trad_score = summary["quality_scores"]["traditional"][metric]["average"]
                lang_score = summary["quality_scores"]["langextract"][metric]["average"]

                report.append(
                    f"| {metric_name} | {trad_score:.3f} | {lang_score:.3f} |"
                )

            report.append("")

        # Individual test details
        report.extend(["## 详细测试结果", ""])

        for test in self.test_results["individual_tests"]:
            sample_name = test["sample_name"]
            winner = test["winner"]
            difficulty = test["sample_info"]["difficulty"]
            description = test["sample_info"]["description"]

            winner_text = (
                "LangExtract"
                if winner == "langextract"
                else "传统方法"
                if winner == "traditional"
                else "平局"
            )

            report.extend(
                [
                    f"### {sample_name}",
                    "",
                    f"**描述**: {description}",
                    f"**难度**: {difficulty}",
                    f"**获胜者**: {winner_text}",
                    "",
                    "| 方法 | 成功 | 耗时 | 提取集群数 | 错误信息 |",
                    "|------|------|------|------------|----------|",
                    f"| 传统方法 | {'✓' if test['traditional_result']['success'] else '✗'} | {test['traditional_result']['execution_time']:.3f}s | {test['traditional_result']['extracted_clusters']} | {test['traditional_result']['error'] or '-'} |",
                    f"| LangExtract | {'✓' if test['langextract_result']['success'] else '✗'} | {test['langextract_result']['execution_time']:.3f}s | {test['langextract_result']['extracted_clusters']} | {test['langextract_result']['error'] or '-'} |",
                    "",
                ]
            )

        report.extend(
            [
                "## 结论",
                "",
                "基于测试结果，可以得出以下结论：",
                "",
                f"1. **成功率对比**: {'LangExtract具有明显优势' if summary['langextract_success_rate'] > summary['traditional_success_rate'] + 0.1 else '传统方法具有优势' if summary['traditional_success_rate'] > summary['langextract_success_rate'] + 0.1 else '两种方法表现相当'}",
                f"2. **性能对比**: {'传统方法在速度上有优势' if summary['traditional_avg_time'] < summary['langextract_avg_time'] else 'LangExtract在速度上有优势'}",
                "",
                "### 推荐使用场景",
                "",
                "- **复杂JSON修复**: 推荐使用LangExtract处理格式严重错误的JSON",
                "- **简单格式问题**: 传统方法足以处理基本的JSON格式问题",
                "- **性能敏感场景**: 根据具体的性能要求选择合适的方法",
                "",
            ]
        )

        return "\n".join(report)

    def save_markdown_report(self, filename: str = "json_parsing_comparison_report.md"):
        """Save markdown report to file."""
        output_path = Path(__file__).parent / filename
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown_report())
            logger.info(f"Markdown report saved to: {output_path}")
            print(f"📄 Markdown报告已保存至: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save markdown report: {e}")
            print(f"❌ 保存Markdown报告失败: {e}")


def main():
    """Main function to run the JSON parsing comparison demo."""
    print("=" * 80)
    print("JSON解析能力对比测试 - Demo 2")
    print("=" * 80)
    print("\n本测试将对比mLLMCelltype现有的JSON修复逻辑与langextract的处理能力")
    print("重点测试各种格式错误的JSON响应样本的处理效果\n")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  警告: 未找到Google/Gemini API密钥")
        print("请设置环境变量 GOOGLE_API_KEY 或 GEMINI_API_KEY")
        print("LangExtract测试将无法执行，但可以测试传统方法\n")
    else:
        print("✅ API密钥已配置，将执行完整对比测试\n")

    # Initialize comparison tool
    comparison = JSONParsingComparison(api_key=api_key)

    # Run comprehensive test
    try:
        results = comparison.run_comprehensive_test()

        # Print results
        comparison.print_detailed_results()

        # Save results
        comparison.save_results()
        comparison.save_markdown_report()

        print("\n🎉 测试完成!")
        print("详细结果已保存为JSON和Markdown格式文件")

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n❌ 测试执行失败: {str(e)}")


if __name__ == "__main__":
    main()
