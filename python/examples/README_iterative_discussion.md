# 迭代讨论功能使用指南

本文档介绍了LLMCelltype框架中新增的迭代讨论功能，该功能允许多轮讨论以解决有争议的细胞簇注释。

## 功能概述

迭代讨论功能允许LLMCelltype框架在处理有争议的细胞簇时，进行多轮深入讨论，而不是仅依赖单轮讨论。这种方法与R版本的功能保持一致，能够更好地解决复杂的细胞类型注释问题。

主要特点：
- 支持多轮讨论（可配置最大讨论轮次）
- 每轮讨论后检查是否达成共识
- 使用共识比例(CP)和香农熵(H)量化注释的确定性
- 保存完整的讨论日志，便于后续分析

## 使用方法

### 基本用法

在调用`interactive_consensus_annotation`函数时，添加`max_discussion_rounds`参数：

```python
result = lct.interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='lung',
    models=models,
    api_keys=api_keys,
    consensus_threshold=0.6,
    max_discussion_rounds=3,  # 设置最大讨论轮次为3
    use_cache=True,
    verbose=True
)
```

### 查看讨论日志

讨论日志存储在结果字典的`discussion_logs`键中：

```python
# 打印特定细胞簇的讨论日志
if "discussion_logs" in result and "4" in result["discussion_logs"]:
    print("\n=== 细胞簇 #4 的讨论日志 ===\n")
    print(result["discussion_logs"]["4"])
```

### 示例代码

我们提供了两个示例脚本：
1. `consensus_example.py` - 基本的共识注释示例，包含迭代讨论功能
2. `iterative_discussion_example.py` - 专门展示迭代讨论功能的详细示例

运行示例：
```bash
python examples/iterative_discussion_example.py
```

## 共识指标说明

迭代讨论功能使用两个主要指标来量化注释的确定性：

1. **共识比例(Consensus Proportion, CP)**
   - 范围：0-1
   - 含义：支持主要细胞类型的证据比例
   - 值越高表示证据支持越一致

2. **香农熵(Shannon Entropy, H)**
   - 范围：0+（理论上无上限，但实际值通常较小）
   - 含义：注释的不确定性
   - 值越低表示不确定性越小

这些指标在讨论过程中会动态更新，帮助跟踪共识形成的过程。

## 参数调整建议

- `max_discussion_rounds`：通常设置为2-5轮。轮次过多可能导致讨论冗余，而轮次过少可能无法充分解决争议。
- `consensus_threshold`：建议设置在0.5-0.7之间。较高的阈值会导致更多的细胞簇被标记为有争议，从而进入讨论流程。

## 注意事项

1. 迭代讨论会增加API调用次数，因此可能增加成本。
2. 使用`use_cache=True`可以避免重复的API调用，特别是在开发和测试阶段。
3. 讨论日志可能会很长，建议在处理大量细胞簇时有选择地查看日志。
