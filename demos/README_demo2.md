# Demo 2: JSON格式修复能力对比测试

## 概述

这个演示专门测试和对比JSON解析和修复能力，重点评估mLLMCelltype现有的JSON修复逻辑与langextract结构化提取的处理效果。

## 测试目标

- 🔍 **格式修复测试**: 测试各种JSON格式错误的修复能力
- ⚖️ **能力对比**: 对比传统正则表达式方法 vs langextract智能提取
- 📊 **性能评估**: 评估成功率、准确性、稳定性、执行速度
- 🚨 **错误恢复**: 测试复杂错误场景下的恢复能力

## 测试样本类型

基于mLLMCelltype实际遇到的问题，创建了以下错误JSON样本：

### 1. 缺少逗号的JSON (`missing_commas`)
```json
{
  "annotations": [
    {
      "cluster": "0"      // 缺少逗号
      "cell_type": "T cells"
      "confidence": "high"
    }
    {                     // 缺少逗号
      "cluster": "1"
      "cell_type": "B cells"
    }
  ]
}
```

### 2. 多余逗号的JSON (`trailing_commas`)
```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": "CD8+ T cells",
      "key_markers": ["CD8A", "CD8B", "GZMB",],  // 多余逗号
      "confidence": "high",                       // 多余逗号
    },
  ],                                            // 多余逗号
}
```

### 3. 引号不匹配的JSON (`quote_mismatches`)
```json
{
  "annotations": [
    {
      "cluster": "0",
      "cell_type": 'CD4+ T cells",    // 单双引号混用
      "confidence": "high'            // 引号不匹配
    }
  ]
}
```

### 4. 嵌套结构错误 (`nested_structure_errors`)
- 嵌套对象和数组的括号不匹配
- 复杂数据结构的格式错误

### 5. markdown标记混合 (`markdown_mixed`)
- JSON中混合了markdown格式标记
- 注释和特殊字符干扰

### 6. 不完整的JSON (`incomplete_json`)
- 响应被截断的JSON
- 部分数据缺失的情况

### 7. 值格式错误 (`malformed_values`)
- 数值类型不匹配
- 字符串/数组格式错误

### 8. 真实复杂场景 (`real_world_complex`)
- 基于实际mllmcelltype输出的复杂错误
- 多种错误类型组合

## 安装和运行

### 1. 安装依赖

```bash
cd /Users/apple/Research/mLLMCelltype/demos/
pip install -r requirements.txt
```

### 2. 配置API密钥

设置环境变量（选择其中一个）：

```bash
export GOOGLE_API_KEY="your-google-api-key"
# 或者
export GEMINI_API_KEY="your-gemini-api-key"
```

### 3. 运行测试

#### 方式1: 直接运行主程序
```bash
python demo2_json_parsing.py
```

#### 方式2: 使用运行器脚本
```bash
python run_demo2.py
```

## 评估指标

测试从以下几个维度评估两种方法的表现：

### 1. 成功率 (Success Rate)
- 能否成功解析并提取出预期的集群注释
- 是否处理了所有预期的集群

### 2. 完整性 (Completeness)
- 提取出的集群数量 vs 预期集群数量
- 数据完整性评分

### 3. 准确性 (Accuracy) 
- 提取的细胞类型注释的质量
- 是否包含有效的细胞类型名称

### 4. 一致性 (Consistency)
- 输出格式的一致性
- 错误处理的一致性

### 5. 健壮性 (Robustness)
- 错误恢复能力
- 异常情况处理

### 6. 性能 (Performance)
- 执行速度对比
- 资源消耗分析

## 输出文件

测试完成后会生成以下文件：

### 1. `json_parsing_comparison_results.json`
详细的测试结果数据，包含：
- 每个测试样本的结果
- 性能指标和质量评估
- 错误信息和诊断数据

### 2. `json_parsing_comparison_report.md`
格式化的Markdown测试报告，包含：
- 测试概述和结果摘要
- 详细的对比分析
- 推荐使用场景

## 预期结果

基于langextract的设计目标，预期在以下场景中会展现优势：

### LangExtract优势场景
- 🔧 **复杂格式错误**: 多种错误类型组合的JSON
- 🧠 **语义理解**: 能理解JSON意图，即使格式严重错误
- 🔄 **自动修复**: 智能推断缺失的结构和数据
- 📝 **上下文处理**: 处理混合markdown和注释的内容

### 传统方法优势场景
- ⚡ **执行速度**: 正则表达式处理速度较快
- 💾 **资源消耗**: 不需要额外的API调用
- 🔒 **数据安全**: 本地处理，不发送数据到外部API
- ✅ **简单错误**: 处理标准化的简单JSON错误

## 技术实现细节

### 传统JSON修复逻辑
使用mLLMCelltype现有的`format_results`函数：
- 正则表达式匹配集群模式
- 固定的格式修复规则
- 基于行解析的后备机制

### LangExtract结构化提取
- 使用大语言模型理解JSON意图
- 自定义提取Schema定义
- 智能错误恢复和数据补全

## 故障排除

### 常见问题

#### 1. API密钥问题
```
❌ No API key available for langextract
```
**解决方案**: 确保设置了正确的环境变量

#### 2. 导入错误
```
❌ Error importing langextract
```
**解决方案**: 
```bash
pip install langextract>=1.0.8
```

#### 3. 模型不可用
```
❌ Model timeout or unavailable
```
**解决方案**: 检查网络连接或尝试其他模型

### 调试模式

在代码中设置`debug=True`可以获得更详细的诊断信息：

```python
comparison = JSONParsingComparison(api_key=api_key)
# 启用调试模式
logging.getLogger().setLevel(logging.DEBUG)
```

## 扩展和定制

### 添加新的测试样本

在`create_problematic_json_samples()`方法中添加新的样本：

```python
"new_test_case": {
    "name": "测试案例名称",
    "description": "测试案例描述", 
    "raw_response": "JSON响应内容",
    "expected_clusters": ["0", "1", "2"],
    "difficulty": "medium"
}
```

### 自定义评估指标

修改`evaluate_data_quality()`方法以添加新的评估维度。

### 使用不同的LLM模型

```python
comparison = JSONParsingComparison(
    api_key=api_key,
    model="gpt-4o-mini"  # 或其他支持的模型
)
```

## 贡献和反馈

如果您发现了新的JSON格式问题或有改进建议：

1. 收集问题样本
2. 在GitHub创建issue或提交PR
3. 描述问题场景和预期行为

## 许可证

本演示代码遵循与mLLMCelltype项目相同的许可证条款。