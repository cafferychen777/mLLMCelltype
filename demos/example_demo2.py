#!/usr/bin/env python3
"""
Demo 2 示例 - 展示JSON解析修复能力测试

这个文件展示了demo2_json_parsing.py中一个具体测试样本的处理过程
"""

import json
import sys
from pathlib import Path

# Add the demo directory to path
sys.path.insert(0, str(Path(__file__).parent))


def show_sample_test():
    """展示一个具体的测试样本和处理过程"""

    print("=" * 80)
    print("Demo 2 示例 - JSON格式修复能力测试")
    print("=" * 80)

    # 展示一个有问题的JSON样本
    problematic_json = """```json
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
```"""

    print("\n📋 测试样本: 缺少逗号的JSON")
    print("问题描述: 在对象属性之间缺少逗号分隔符")
    print("\n原始LLM响应:")
    print("-" * 40)
    print(problematic_json)
    print("-" * 40)

    print("\n🔍 识别的问题:")
    print("1. 'cluster': '0' 和 'cell_type': 'T cells' 之间缺少逗号")
    print("2. 'cell_type': 'T cells' 和 'confidence': 'high' 之间缺少逗号")
    print("3. 第一个对象和第二个对象之间缺少逗号")
    print("4. 第二个对象内部也有类似的逗号缺失问题")

    # 展示传统方法可能的处理结果
    print("\n🔧 传统方法处理:")
    print("- 使用正则表达式尝试修复常见的JSON格式问题")
    print("- 可能成功修复一些简单的逗号缺失")
    print("- 复杂的嵌套结构错误可能处理困难")
    print("- 依赖于预定义的修复规则")

    # 展示langextract的处理方式
    print("\n🧠 LangExtract处理:")
    print("- 理解JSON的语义意图，而非仅仅修复语法")
    print("- 能够识别这是一个细胞类型注释结构")
    print("- 即使格式严重错误，也能提取出正确的信息")
    print("- 返回结构化的数据，确保数据完整性")

    # 展示期望的输出格式
    expected_output = {
        "annotations": [
            {"cluster_id": "0", "cell_type": "T cells", "confidence": "high"},
            {"cluster_id": "1", "cell_type": "B cells", "confidence": "medium"},
        ]
    }

    print("\n🎯 期望的提取结果:")
    print(json.dumps(expected_output, indent=2, ensure_ascii=False))

    print("\n📊 评估维度:")
    print("- ✅ 成功率: 是否能成功解析出集群信息")
    print("- ✅ 完整性: 是否提取了所有预期的集群")
    print("- ✅ 准确性: 细胞类型注释是否正确")
    print("- ✅ 一致性: 输出格式是否一致")
    print("- ✅ 健壮性: 错误处理是否优雅")
    print("- ⚡ 性能: 处理速度和资源消耗")


def show_test_categories():
    """展示所有测试类别"""

    print("\n" + "=" * 80)
    print("所有测试类别概览")
    print("=" * 80)

    categories = [
        {
            "name": "missing_commas",
            "title": "缺少逗号的JSON",
            "difficulty": "medium",
            "description": "对象属性之间缺少逗号分隔符",
        },
        {
            "name": "trailing_commas",
            "title": "多余逗号的JSON",
            "difficulty": "easy",
            "description": "数组和对象末尾有多余的逗号",
        },
        {
            "name": "quote_mismatches",
            "title": "引号不匹配的JSON",
            "difficulty": "hard",
            "description": "字符串值的引号不匹配或类型错误",
        },
        {
            "name": "nested_structure_errors",
            "title": "嵌套结构错误",
            "difficulty": "very_hard",
            "description": "嵌套对象和数组的括号不匹配",
        },
        {
            "name": "markdown_mixed",
            "title": "混合markdown标记",
            "difficulty": "hard",
            "description": "JSON中混合了markdown格式标记",
        },
        {
            "name": "incomplete_json",
            "title": "不完整的JSON",
            "difficulty": "very_hard",
            "description": "JSON响应被截断或不完整",
        },
        {
            "name": "malformed_values",
            "title": "值格式错误",
            "difficulty": "medium",
            "description": "JSON中的值不符合预期格式",
        },
        {
            "name": "real_world_complex",
            "title": "真实复杂场景",
            "difficulty": "very_hard",
            "description": "基于实际mllmcelltype输出的复杂错误",
        },
    ]

    for i, category in enumerate(categories, 1):
        difficulty_emoji = {
            "easy": "🟢",
            "medium": "🟡",
            "hard": "🟠",
            "very_hard": "🔴",
        }.get(category["difficulty"], "⚪")

        print(f"\n{i}. {difficulty_emoji} {category['title']} ({category['name']})")
        print(f"   难度: {category['difficulty']}")
        print(f"   描述: {category['description']}")


def show_usage_instructions():
    """展示使用说明"""

    print("\n" + "=" * 80)
    print("使用说明")
    print("=" * 80)

    print("\n🚀 运行完整测试:")
    print("python demo2_json_parsing.py")
    print("# 或者")
    print("python run_demo2.py")

    print("\n🔧 配置API密钥:")
    print("export GOOGLE_API_KEY='your-api-key'")
    print("# 或者")
    print("export GEMINI_API_KEY='your-api-key'")

    print("\n📁 输出文件:")
    print("- json_parsing_comparison_results.json  (详细测试数据)")
    print("- json_parsing_comparison_report.md     (格式化报告)")

    print("\n💡 提示:")
    print("- 确保已安装所有依赖: pip install -r requirements.txt")
    print("- 如果没有API密钥，仍可测试传统方法的效果")
    print("- 测试过程中会显示实时进度和结果")
    print("- 所有测试数据都会保存到文件中供进一步分析")


if __name__ == "__main__":
    show_sample_test()
    show_test_categories()
    show_usage_instructions()

    print("\n🎉 准备好运行完整的JSON解析能力对比测试了！")
    print("执行 'python demo2_json_parsing.py' 开始测试")
