# Python版本 base_url 支持完整实现方案

## 📋 概述

本文档详细描述了为 mLLMCelltype Python 版本添加 per-provider base_url 支持的完整实现方案，以实现与 R 版本功能对等。

## 🔍 当前架构分析

### 核心结构
- **主要入口函数**: `annotate_clusters()`, `interactive_consensus_annotation()`
- **Provider架构**: 每个API提供商有独立的处理函数 (`process_openai`, `process_qwen`等)
- **函数映射**: `PROVIDER_FUNCTIONS` 字典映射provider到处理函数
- **当前版本**: v1.2.10

### 当前API调用方式
```python
# 例如 OpenAI provider
url = "https://api.openai.com/v1/chat/completions"  # 硬编码URL
response = requests.post(url=url, headers=headers, data=json.dumps(body))
```

### 存在的问题
1. **硬编码URL**: 所有API端点都是硬编码的
2. **无法自定义**: 用户无法使用代理或替代端点
3. **中国用户限制**: 无法通过代理访问国际API
4. **企业限制**: 无法使用内部API网关

## 🎯 实现方案

### 方案1: 最小侵入式修改（推荐）

#### 优势
- ✅ **最小侵入**: 保持现有架构不变
- ✅ **向后兼容**: 不破坏现有API
- ✅ **快速实现**: 可以在短时间内完成
- ✅ **与R版本一致**: 保持两个版本的功能对等

#### 实施步骤

##### 1. 创建URL工具函数

**新文件**: `python/mllmcelltype/url_utils.py`

```python
"""URL utilities for base URL resolution."""

from typing import Optional, Union


def resolve_provider_base_url(provider: str, base_urls: Union[str, dict, None]) -> Optional[str]:
    """解析provider特定的base URL
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        base_urls: User-provided base URLs (string or dict)
        
    Returns:
        Resolved base URL or None
    """
    if base_urls is None:
        return None
    
    if isinstance(base_urls, str):
        return base_urls  # 单一URL应用于所有provider
    
    if isinstance(base_urls, dict) and provider in base_urls:
        return base_urls[provider]  # Provider特定URL
    
    return None


def get_default_api_url(provider: str) -> str:
    """获取默认API URL
    
    Args:
        provider: Provider name
        
    Returns:
        Default API URL for the provider
    """
    default_urls = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "grok": "https://api.x.ai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "stepfun": "https://api.stepfun.com/v1/chat/completions",
        "minimax": "https://api.minimax.chat/v1/text/chatcompletion_v2"
    }
    return default_urls.get(provider, "")


def validate_base_url(url: str) -> bool:
    """验证base URL格式
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False
    
    # 基本URL格式检查
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    return True
```

##### 2. 修改Provider函数签名

**所有provider文件需要修改** (`providers/*.py`):

```python
# 当前签名
def process_openai(prompt: str, model: str, api_key: str) -> list[str]:

# 新签名
def process_openai(prompt: str, model: str, api_key: str, base_url: Optional[str] = None) -> list[str]:
```

##### 3. 更新Provider实现

**示例**: `providers/openai.py`

```python
def process_openai(prompt: str, model: str, api_key: str, base_url: Optional[str] = None) -> list[str]:
    """Process request using OpenAI models.
    
    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gpt-4o', 'o1')
        api_key: OpenAI API key
        base_url: Optional custom base URL
        
    Returns:
        List[str]: Processed responses, one per cluster
    """
    from ..url_utils import get_default_api_url, validate_base_url
    
    write_log(f"Starting OpenAI API request with model: {model}")
    
    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "OpenAI API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # 使用自定义URL或默认URL
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_default_api_url("openai")
        write_log(f"Using default URL: {url}")
    
    # 其余逻辑保持不变...
    response = requests.post(url=url, headers=headers, data=json.dumps(body))
```

##### 4. 修改主要入口函数

**文件**: `annotate.py`

```python
def annotate_clusters(
    marker_genes: Union[dict[str, list[str]], pd.DataFrame],
    species: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    base_urls: Optional[Union[str, dict[str, str]]] = None,  # 新增参数
) -> dict[str, str]:
    """Annotate cell clusters using LLM.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o', 'claude-3-opus-20240229')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs (e.g., {'openai': 'https://proxy.com/v1'})
                  
    Returns:
        dict[str, str]: Dictionary mapping cluster names to cell type annotations
    """
    from .url_utils import resolve_provider_base_url
    
    # 解析base URL
    base_url = resolve_provider_base_url(provider, base_urls)
    
    # 调用provider函数时传递base_url
    provider_func = PROVIDER_FUNCTIONS[provider]
    result = provider_func(prompt, model, api_key, base_url)
    
    # 其余逻辑保持不变...
```

**文件**: `consensus.py`

```python
def interactive_consensus_annotation(
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[Union[str, dict[str, str]]] = None,
    api_keys: Optional[dict[str, str]] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    max_discussion_rounds: int = 3,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    consensus_model: Optional[Union[str, dict[str, str]]] = None,
    base_urls: Optional[Union[str, dict[str, str]]] = None,  # 新增参数
) -> dict[str, Any]:
    """Perform consensus annotation using multiple LLMs.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        models: List of models to use for annotation
        api_keys: Dictionary mapping provider names to API keys
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        consensus_threshold: Agreement threshold below which a cluster is controversial
        entropy_threshold: Entropy threshold above which a cluster is controversial
        max_discussion_rounds: Maximum number of discussion rounds
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        verbose: Whether to print detailed logs
        consensus_model: Optional model for consensus checking and discussion
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs
                  
    Returns:
        dict[str, Any]: Dictionary containing consensus results and metadata
    """
    from .url_utils import resolve_provider_base_url
    
    # 在调用各个模型时传递相应的base_url
    for model_item in models:
        provider = get_provider(model_item)
        base_url = resolve_provider_base_url(provider, base_urls)
        
        # 调用annotate_clusters时传递base_urls参数
        result = annotate_clusters(
            marker_genes=marker_genes,
            species=species,
            provider=provider,
            model=model_item,
            api_key=api_keys.get(provider),
            tissue=tissue,
            additional_context=additional_context,
            use_cache=use_cache,
            cache_dir=cache_dir,
            base_urls={provider: base_url} if base_url else None
        )
    
    # 其余逻辑保持不变...
```

##### 5. 特殊功能实现

**Qwen智能端点选择** (类似R版本):

```python
# 在 providers/qwen.py 中添加
import time
import requests

def test_endpoint_connectivity(endpoint: str, api_key: str, timeout: int = 5) -> bool:
    """测试端点连通性

    Args:
        endpoint: API endpoint URL
        api_key: API key for authentication
        timeout: Timeout in seconds

    Returns:
        True if endpoint is accessible, False otherwise
    """
    try:
        # 发送简单的测试请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        test_body = {
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=test_body,
            timeout=timeout
        )

        # 如果返回200或者是认证/模型相关错误（说明端点可达），则认为连通
        return response.status_code in [200, 400, 401, 403]

    except Exception:
        return False


def get_working_qwen_endpoint(api_key: str) -> str:
    """智能选择Qwen端点

    Args:
        api_key: Qwen API key

    Returns:
        Working endpoint URL
    """
    endpoints = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",  # 国际版
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"       # 国内版
    ]

    write_log("Testing Qwen endpoint connectivity...")

    for i, endpoint in enumerate(endpoints):
        endpoint_type = "international" if i == 0 else "domestic"
        write_log(f"Testing {endpoint_type} endpoint: {endpoint}")

        if test_endpoint_connectivity(endpoint, api_key):
            write_log(f"✅ {endpoint_type} endpoint is accessible")
            return endpoint
        else:
            write_log(f"❌ {endpoint_type} endpoint is not accessible")

    # 如果都不可达，返回国际版作为默认
    write_log("No endpoints accessible, using international endpoint as fallback")
    return endpoints[0]


def process_qwen(prompt: str, model: str, api_key: str, base_url: Optional[str] = None) -> list[str]:
    """Process request using Alibaba Qwen models with smart endpoint selection.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'qwen-plus', 'qwen-max-2025-01-25')
        api_key: DashScope API key
        base_url: Optional custom base URL (overrides smart selection)

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Qwen API request with model: {model}")

    if not api_key:
        error_msg = "DashScope API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # 使用自定义URL或智能选择
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_working_qwen_endpoint(api_key)
        write_log(f"Using smart-selected endpoint: {url}")

    # 其余逻辑保持不变...
```

##### 6. 更新函数导出

**文件**: `__init__.py`

```python
# 添加新的导出
from .url_utils import (
    resolve_provider_base_url,
    get_default_api_url,
    validate_base_url,
)

__all__ = [
    # 现有导出...

    # URL utilities
    "resolve_provider_base_url",
    "get_default_api_url",
    "validate_base_url",
]
```

### 方案2: 面向对象重构（备选方案）

#### 概述
创建BaseProvider类和具体Provider子类，实现更清晰的架构。

#### 优势
- ✅ **更清晰的架构**: 面向对象设计
- ✅ **更好的扩展性**: 易于添加新provider
- ✅ **更强的类型安全**: 更好的IDE支持

#### 劣势
- ❌ **工作量大**: 需要重构大量代码
- ❌ **破坏性变更**: 可能影响现有用户
- ❌ **实施周期长**: 需要1-2周时间

#### 实施概要

```python
# 新文件: python/mllmcelltype/base_provider.py
from abc import ABC, abstractmethod
from typing import Optional

class BaseProvider(ABC):
    def __init__(self, provider_name: str, base_url: Optional[str] = None):
        self.provider_name = provider_name
        self.base_url = base_url or self.get_default_url()

    @abstractmethod
    def get_default_url(self) -> str:
        """获取默认API URL"""
        pass

    @abstractmethod
    def process(self, prompt: str, model: str, api_key: str) -> list[str]:
        """处理API请求"""
        pass

class OpenAIProvider(BaseProvider):
    def get_default_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def process(self, prompt: str, model: str, api_key: str) -> list[str]:
        # 使用 self.base_url 进行API调用
        pass
```

## 📊 工作量评估

### 方案1 (推荐)
- **文件修改**: ~15个文件
- **新增文件**: 1个 (`url_utils.py`)
- **工作量**: 中等 (2-3天)
- **风险**: 低 (向后兼容)
- **测试工作**: 中等

### 方案2
- **文件修改**: ~20个文件
- **新增文件**: 多个
- **工作量**: 大 (1-2周)
- **风险**: 中等 (架构变更)
- **测试工作**: 大

## 🧪 测试计划

### 单元测试
```python
# tests/test_url_utils.py
import pytest
from mllmcelltype.url_utils import resolve_provider_base_url, validate_base_url

def test_resolve_provider_base_url_string():
    """测试单一URL字符串"""
    result = resolve_provider_base_url("openai", "https://proxy.com/v1")
    assert result == "https://proxy.com/v1"

def test_resolve_provider_base_url_dict():
    """测试provider特定URL字典"""
    base_urls = {
        "openai": "https://openai-proxy.com/v1",
        "anthropic": "https://anthropic-proxy.com/v1"
    }
    result = resolve_provider_base_url("openai", base_urls)
    assert result == "https://openai-proxy.com/v1"

def test_validate_base_url():
    """测试URL验证"""
    assert validate_base_url("https://api.example.com/v1") == True
    assert validate_base_url("http://localhost:8080") == True
    assert validate_base_url("invalid-url") == False
    assert validate_base_url("") == False
```

### 集成测试
```python
# tests/test_base_url_integration.py
def test_annotate_clusters_with_base_url():
    """测试annotate_clusters的base_url功能"""
    # 模拟测试...

def test_consensus_annotation_with_base_urls():
    """测试共识注释的base_urls功能"""
    # 模拟测试...
```

## 📝 使用示例

### 基本用法
```python
import os
from mllmcelltype import annotate_clusters, interactive_consensus_annotation

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# 示例1: 单一代理URL
result = annotate_clusters(
    marker_genes=marker_genes,
    species="human",
    provider="openai",
    model="gpt-4o",
    base_urls="https://api.your-proxy.com/v1"  # 所有provider使用相同代理
)

# 示例2: Provider特定URL
result = annotate_clusters(
    marker_genes=marker_genes,
    species="human",
    provider="openai",
    model="gpt-4o",
    base_urls={
        "openai": "https://openai-proxy.com/v1",
        "anthropic": "https://anthropic-proxy.com/v1"
    }
)

# 示例3: 共识注释中使用代理
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=["gpt-4o", "claude-3-opus", "qwen-max"],
    api_keys={
        "openai": "your-openai-key",
        "anthropic": "your-anthropic-key",
        "qwen": "your-qwen-key"
    },
    base_urls={
        "openai": "https://openai-proxy.com/v1",
        "anthropic": "https://anthropic-proxy.com/v1"
        # qwen 使用默认端点（智能选择）
    }
)
```

### 中国用户配置示例
```python
# 中国用户推荐配置
china_base_urls = {
    # 国际API使用代理
    "openai": "https://your-openai-proxy.com/v1",
    "anthropic": "https://your-anthropic-proxy.com/v1",
    "gemini": "https://your-gemini-proxy.com/v1beta/models",

    # 国内API使用默认端点
    # "qwen", "deepseek", "zhipu" 等不需要配置
}

# 混合使用国内外模型
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=[
        "gpt-4o",           # 通过代理访问
        "claude-3-opus",    # 通过代理访问
        "qwen-max",         # 直接访问（智能端点选择）
        "deepseek-chat",    # 直接访问
        "glm-4-plus"        # 直接访问
    ],
    api_keys=your_api_keys,
    base_urls=china_base_urls
)
```

## 🚀 实施时间表

### 第1周
- [ ] 创建 `url_utils.py` 工具函数
- [ ] 修改所有provider函数签名
- [ ] 更新 `annotate_clusters` 函数
- [ ] 基本功能测试

### 第2周
- [ ] 更新 `interactive_consensus_annotation` 函数
- [ ] 实现Qwen智能端点选择
- [ ] 更新所有provider实现
- [ ] 编写单元测试

### 第3周
- [ ] 集成测试
- [ ] 文档更新
- [ ] 示例代码
- [ ] 版本发布准备

## 🎯 最终建议

**强烈推荐使用方案1**，原因：

1. **快速交付**: 可以在2-3周内完成
2. **风险可控**: 向后兼容，不破坏现有功能
3. **功能对等**: 与R版本保持一致的功能
4. **用户友好**: 满足中国用户和企业用户需求

实施完成后，Python版本将具备：
- ✅ **完整的base_url支持**
- ✅ **Qwen智能端点选择**
- ✅ **代理服务器支持**
- ✅ **企业API网关支持**
- ✅ **与R版本功能对等**

这将使mLLMCelltype成为真正的跨平台、全功能的细胞类型注释工具！

## 📋 需要修改的文件清单

### 新增文件
1. `python/mllmcelltype/url_utils.py` - URL工具函数

### 修改文件
1. `python/mllmcelltype/__init__.py` - 添加新函数导出
2. `python/mllmcelltype/annotate.py` - 添加base_urls参数
3. `python/mllmcelltype/consensus.py` - 添加base_urls参数
4. `python/mllmcelltype/providers/openai.py` - 添加base_url支持
5. `python/mllmcelltype/providers/anthropic.py` - 添加base_url支持
6. `python/mllmcelltype/providers/qwen.py` - 添加base_url支持和智能端点选择
7. `python/mllmcelltype/providers/deepseek.py` - 添加base_url支持
8. `python/mllmcelltype/providers/gemini.py` - 添加base_url支持
9. `python/mllmcelltype/providers/zhipu.py` - 添加base_url支持
10. `python/mllmcelltype/providers/grok.py` - 添加base_url支持
11. `python/mllmcelltype/providers/openrouter.py` - 添加base_url支持
12. `python/mllmcelltype/providers/stepfun.py` - 添加base_url支持
13. `python/mllmcelltype/providers/minimax.py` - 添加base_url支持
14. `python/setup.py` - 更新版本号到v1.3.0
15. `python/README.md` - 更新文档

### 测试文件
1. `python/tests/test_url_utils.py` - URL工具函数测试
2. `python/tests/test_base_url_integration.py` - 集成测试

**总计**: 1个新增文件 + 15个修改文件 + 2个测试文件 = 18个文件

这个方案提供了完整的实施路径，确保Python版本能够快速获得与R版本相同的base_url支持功能！
