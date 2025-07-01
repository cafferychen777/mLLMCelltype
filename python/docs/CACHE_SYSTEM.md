# mLLMCelltype 缓存系统文档

## 问题描述

用户在使用 `interactive_consensus_annotation` 函数切换不同模型时遇到缓存问题，具体表现为：
- 切换到 OpenRouter 模型后，仍然返回之前模型的缓存结果
- 不同的 API 调用失败，但系统继续使用旧的缓存数据
- 缓存键没有正确区分不同的 provider

## 根本原因

缓存键生成时没有正确处理 OpenRouter 模型的特殊情况。OpenRouter 模型使用 "provider/model" 格式（如 "openai/gpt-4o-mini"），但缓存系统可能使用错误的 provider 信息生成缓存键。

## 已实施的长期解决方案

### 1. 修改缓存键生成逻辑 (utils.py)

```python
def create_cache_key(prompt: str, model: str, provider: str) -> str:
    # Normalize inputs to ensure consistent keys
    normalized_provider = str(provider).lower().strip()
    normalized_model = str(model).lower().strip()
    normalized_prompt = str(prompt).strip()

    # For OpenRouter models (containing '/'), ensure provider is 'openrouter'
    # This prevents cache key collisions between different providers
    if "/" in normalized_model:
        normalized_provider = "openrouter"

    # Create a string to hash with clear separators to avoid collisions
    hash_string = (
        f"provider:{normalized_provider}||model:{normalized_model}||prompt:{normalized_prompt}"
    )

    # Create hash
    hash_object = hashlib.sha256(hash_string.encode("utf-8"))
    return hash_object.hexdigest()
```

### 2. 测试验证

创建了完整的测试套件 `tests/test_cache_system.py`，验证：
- ✅ 缓存键正确包含 provider 信息
- ✅ OpenRouter 模型始终使用 "openrouter" 作为 provider
- ✅ 不同 provider 的相同模型不会共享缓存
- ✅ 真实场景测试通过

### 3. 缓存管理模块

提供了 `cache_manager` 模块，支持：
- 查看缓存信息：`python -m mllmcelltype.cache_manager --info`
- 清理缓存：`python -m mllmcelltype.cache_manager --clear`
- 交互式管理：`python -m mllmcelltype.cache_manager`

也可以在代码中使用：
```python
from mllmcelltype import get_cache_info, clear_cache

# 获取缓存信息
info = get_cache_info()
print(f"缓存文件数: {info['file_count']}")
print(f"缓存大小: {info['size_mb']:.2f} MB")

# 清理缓存
removed = clear_cache()
print(f"清理了 {removed} 个缓存文件")
```

## 使用建议

### 1. 更新代码后清理缓存
```bash
python -m mllmcelltype.cache_manager --clear
```

### 2. 正确指定模型
```python
# OpenRouter 模型
models = [
    {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
    {"provider": "openrouter", "model": "anthropic/claude-3-opus"},
]

# 或者让系统自动检测（推荐）
models = [
    "openai/gpt-4o-mini",  # 自动检测为 openrouter
    "anthropic/claude-3-opus",  # 自动检测为 openrouter
    "gpt-4o",  # 自动检测为 openai
    "claude-3-opus",  # 自动检测为 anthropic
]
```

### 3. 临时禁用缓存（如果需要）
```python
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="Hepatocellular carcinoma",
    models=models,
    use_cache=False  # 临时禁用缓存
)
```

## 技术细节

### 缓存键组成
- Provider（标准化后的提供商名称）
- Model（标准化后的模型名称）
- Prompt（用户提供的提示）

### 特殊处理
- 包含 "/" 的模型名称自动归类为 OpenRouter
- 所有输入都进行小写和空格标准化
- 使用 SHA256 生成唯一缓存键

### 缓存存储位置
- 默认：`~/.llmcelltype/cache/`
- 格式：JSON 文件，包含版本号和时间戳

## 监控和维护

### 查看缓存统计
```python
from mllmcelltype.utils import get_cache_stats
stats = get_cache_stats()
print(f"缓存文件数: {stats['count']}")
print(f"缓存大小: {stats['size_readable']}")
```

### 定期清理旧缓存
```python
from mllmcelltype.utils import clear_cache
# 清理超过 7 天的缓存
removed = clear_cache(older_than=7*24*60*60)
print(f"清理了 {removed} 个缓存文件")
```

## 验证修复效果

运行测试套件确认修复有效：
```bash
cd tests
python test_cache_system.py
```

预期输出：
```
✅ ALL TESTS PASSED!
The cache fix is working correctly.
```

## 后续建议

1. **版本控制**：在未来的版本中考虑缓存版本控制，便于升级时自动清理不兼容的缓存
2. **缓存过期**：考虑添加缓存过期时间设置
3. **缓存大小限制**：考虑添加缓存大小限制，自动清理最旧的缓存

## 相关文件

- `/mllmcelltype/utils.py` - 包含修复后的缓存键生成逻辑
- `/mllmcelltype/cache_manager.py` - 缓存管理模块
- `/tests/test_cache_system.py` - 完整的测试套件
- `/docs/CACHE_SYSTEM.md` - 本文档

## 问题追踪

GitHub Issue: https://github.com/cafferychen777/mLLMCelltype/issues/65