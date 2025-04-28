<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - 单细胞 RNA 测序数据的多大语言模型细胞类型注释框架" width="200"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=推荐使用%20mLLMCelltype%3A%20一个用于单细胞%20RNA%20测序数据细胞类型注释的多模型共识框架%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="分享到推特"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="星标数"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="分支数"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-加入聊天-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="许可证">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="最近提交">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="问题">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="发布版本">
</div>

# mLLMCelltype: 多大语言模型共识框架用于细胞类型注释

mLLMCelltype是一个先进的迭代式多大语言模型（Multi-LLM）共识框架，专为单细胞RNA测序（scRNA-seq）数据的精确可靠的细胞类型注释而设计。通过利用多种大语言模型（包括OpenAI GPT-4o/4.1、Anthropic Claude-3.7/3.5、Google Gemini-2.0、X.AI Grok-3、DeepSeek-V3、阿里云 Qwen2.5、智谱 GLM-4、MiniMax、Stepfun、和 OpenRouter）的集体智慧，该框架显著提高了注释准确性，同时为生物信息学和计算生物学研究提供透明的不确定性量化。

## 摘要

mLLMCelltype 利用多个大语言模型在结构化的讨论过程中准确地从基因表达数据中识别细胞类型。与依赖参考数据集或单一 AI 模型的传统注释方法不同，mLLMCelltype 实现了一种新颜的共识方法，减少幻觉、提高准确性并提供透明的不确定性指标。该框架设计为与流行的单细胞分析平台（如 Scanpy 和 Seurat）无缝集成，使其对生物信息学研究社区的研究人员广泛可用。

## 目录
- [新闻](#新闻)
- [主要特点](#主要特点)
- [最新更新](#最新更新)
- [目录结构](#目录结构)
- [安装](#安装)
- [使用示例](#使用示例)
- [可视化示例](#可视化示例)
- [引用](#引用)
- [贡献](#贡献)

## 新闻

🎉 **2025年4月**：我们非常高兴地宣布，在预印本发布不到一周的时间里，mLLMCelltype已经突破了100个GitHub星标！我们也看到了来自各种媒体和内容创作者的大量报道。我们向所有通过星标、分享和贡献支持这个项目的人表示衷心的感谢。您的热情推动着我们继续开发和改进mLLMCelltype。

## 主要特点

- **多LLM共识架构**：汇集多种大语言模型的集体智慧，克服单一模型的局限性和偏见
- **结构化讨论过程**：使大语言模型能够通过多轮协作讨论分享推理、评估证据并改进注释
- **透明的不确定性量化**：提供定量指标（共识比例和香农熵）来识别需要专家审查的模糊细胞群体
- **幻觉减少**：跨模型讨论通过批判性评估主动抑制不准确或无支持的预测
- **对输入噪声的鲁棒性**：通过集体错误修正，即使在标记基因列表不完美的情况下也能保持高准确性
- **层次注释支持**：可选扩展，用于具有父子一致性的多分辨率分析
- **无需参考数据集**：无需预训练或参考数据即可进行准确注释
- **完整的推理链**：记录完整的讨论过程，实现透明的决策
- **无缝集成**：直接与标准Scanpy/Seurat工作流和标记基因输出配合使用
- **模块化设计**：随着新LLM的可用性，可轻松整合

## 目录结构

- `R/`：R语言接口和实现
- `python/`：Python接口和实现

## 安装

### R版本

```r
# 从GitHub安装
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```



### Python版本

```bash
# 从PyPI安装
pip install mllmcelltype

# 或从GitHub安装
pip install git+https://github.com/cafferychen777/mLLMCelltype.git
```

### 支持的模型

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([API Key](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([API Key](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([API Key](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API Key](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API Key](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API Key](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API Key](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API Key](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([API Key](https://accounts.x.ai/))
- **OpenRouter**: 通过单一API访问多种模型 ([API Key](https://openrouter.ai/keys))
  - 支持来自OpenAI、Anthropic、Meta、Google、Mistral等多家提供商的模型
  - 格式: 'provider/model-name'（例如：'openai/gpt-4o'、'anthropic/claude-3-opus'）
  - 提供免费模型，使用`:free`后缀（例如：'meta-llama/llama-4-maverick:free'、'nvidia/llama-3.1-nemotron-ultra-253b-v1:free'）

## 中国大陆用户指南

为了帮助中国大陆用户更好地使用 mLLMCelltype，我们提供以下特别指南，解决可能遇到的网络限制和 API 访问问题。

### 模型可访问性

在中国大陆地区，不同的 LLM 模型有不同的可访问性：

#### 直接可用的模型

以下模型在中国大陆可以直接访问，无需特殊网络设置：

- **DeepSeek**：DeepSeek-V3、DeepSeek-R1 等系列模型
- **阿里云 Qwen**：Qwen2.5-Max 等千问系列模型
- **智谱 AI**：GLM-4-Plus、GLM-3-Turbo 等系列模型
- **MiniMax**：MiniMax-Text-01 等系列模型
- **Stepfun**：Step-2-16K、Step-2-Mini、Step-1-8K 等系列模型

这些模型是中国大陆用户的首选，因为它们提供稳定的 API 访问，且不需要特殊网络配置。

#### 需要特殊网络设置的模型

以下模型可能需要特殊网络设置才能在中国大陆访问：

- **OpenAI**：GPT-4o、GPT-4.1 等系列模型
- **Anthropic**：Claude-3.7-Sonnet、Claude-3.5-Haiku 等系列模型
- **Google**：Gemini-2.0-Pro、Gemini-1.5-Flash 等系列模型
- **X.AI**：Grok-3、Grok-3-Mini 等系列模型

### 推荐使用方案

#### 方案一：使用国内可直接访问的模型

这是最简单的解决方案，使用中国大陆可直接访问的模型进行细胞类型注释：

```python
# Python 示例 - 使用国内大语言模型进行单细胞 RNA-seq 数据的细胞类型注释
import os
from mllmcelltype import interactive_consensus_annotation

# 配置国内大语言模型的 API 密钥，用于多模型共识注释
os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"  # DeepSeek-V3 模型 API 密钥，用于高精度生物信息学分析
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # 阿里云千问 2.5 API 密钥，提供中文生物医学专业知识
os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"      # 智谱 GLM-4 API 密钥，具有强大的中文生物学能力

# 执行多模型共识细胞类型注释，通过国内大语言模型的协作讨论提高准确性
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,      # 各细胞类群的标记基因字典，用于细胞类型识别
    species="human",               # 指定物种信息，提高注释准确性
    tissue="blood",               # 指定组织类型，提供重要的生物学背景
    models=["deepseek-chat", "qwen-max-2025-01-25", "glm-4-plus"],  # 使用国内三个顶级模型进行共识
    consensus_threshold=0.7        # 设置共识阈值，平衡准确性和覆盖率
```

```r
# R 示例
library(mLLMCelltype)

# 使用国内模型进行共识注释
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = c(
    "deepseek-chat",        # DeepSeek
    "qwen-max-2025-01-25",  # 阿里云千问
    "glm-4-plus"            # 智谱 GLM
  ),
  api_keys = list(
    deepseek = "your-deepseek-key",
    qwen = "your-qwen-key",
    zhipu = "your-zhipu-key"
  )
)
```

#### 方案二：使用 OpenRouter 作为统一接入点

OpenRouter 提供了一个统一的 API 接口，可以访问多种模型，包括部分在中国大陆难以直接访问的模型。通过 OpenRouter，您只需要一个 API 密钥即可使用多种模型：

```python
# Python 示例
import os
from mllmcelltype import interactive_consensus_annotation

# 设置 OpenRouter API 密钥
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 使用 OpenRouter 访问多种模型
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        "openai/gpt-4o",                  # 通过 OpenRouter 访问 OpenAI（付费）
        "anthropic/claude-3-opus",        # 通过 OpenRouter 访问 Anthropic（付费）
        "deepseek/deepseek-chat"          # 通过 OpenRouter 访问 DeepSeek（付费）
    ],
    consensus_threshold=0.7
)

# 使用 OpenRouter 免费模型（无需消耗积分）
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        "meta-llama/llama-4-maverick:free",                # Meta Llama 4 Maverick（免费）
        "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",    # NVIDIA Nemotron Ultra 253B（免费）
        "deepseek/deepseek-chat-v3-0324:free",             # DeepSeek Chat v3（免费）
        "microsoft/mai-ds-r1:free"                         # Microsoft MAI-DS-R1（免费）
    ],
    consensus_threshold=0.7
)
```

```r
# R 示例
library(mLLMCelltype)

# 使用 OpenRouter 进行共识注释（付费模型）
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = c(
    "openai/gpt-4o",              # 通过 OpenRouter 访问 OpenAI（付费）
    "anthropic/claude-3-opus",    # 通过 OpenRouter 访问 Anthropic（付费）
    "deepseek/deepseek-chat"      # 通过 OpenRouter 访问 DeepSeek（付费）
  ),
  api_keys = list(
    openrouter = "your-openrouter-key"  # 单一 API 密钥访问多种模型
  )
)

# 使用 OpenRouter 免费模型进行共识注释（无需消耗积分）
free_consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = c(
    "meta-llama/llama-4-maverick:free",                # Meta Llama 4 Maverick（免费）
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",    # NVIDIA Nemotron Ultra 253B（免费）
    "deepseek/deepseek-chat-v3-0324:free",             # DeepSeek Chat v3（免费）
    "microsoft/mai-ds-r1:free"                         # Microsoft MAI-DS-R1（免费）
  ),
  api_keys = list(
    openrouter = "your-openrouter-key"  # 单一 API 密钥访问多种模型
  )
)
```

您可以在 [OpenRouter 官网](https://openrouter.ai/keys) 申请 API 密钥。

#### 方案三：使用 Azure OpenAI 服务

Microsoft 的 Azure OpenAI 服务在中国大陆有数据中心，可以提供更稳定的访问体验：

```python
# Python 示例
import os
from mllmcelltype import annotate_cell_types

# 设置 Azure OpenAI 的凭证
# 方式一：使用环境变量
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"

# 方式二：直接在函数中提供密钥和端点
results = annotate_cell_types(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    model="azure-gpt-4o",  # 注意模型名称前缀为 'azure-'
    api_key="your-azure-openai-key|https://your-resource.openai.azure.com/"
)
```

```r
# R 示例
library(mLLMCelltype)

# 方式一：使用环境变量
Sys.setenv(AZURE_OPENAI_API_KEY = "your-azure-openai-key")
Sys.setenv(AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/")

# 方式二：直接在函数中提供密钥和端点
results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "azure-gpt-4o",  # 注意模型名称前缀为 'azure-'
  api_key = "your-azure-openai-key|https://your-resource.openai.azure.com/"
)
```

要使用 Azure OpenAI，您需要：
1. 创建 Azure 账户并申请 Azure OpenAI 服务
2. 创建部署，选择您需要的模型（如 gpt-4o）
3. 获取 API 密钥和端点 URL

### 网络配置建议

如果您需要访问国际模型，且遇到网络连接问题，以下是一些配置建议：

#### 自定义 base_url 功能（即将推出）

我们正在开发对 mLLMCelltype 的自定义 base_url 功能支持，这将对中国大陆用户特别有用，可以连接到替代 API 端点。该功能将在下一版本中推出，实现后将支持以下用法：

```r
# R 中使用自定义 base_url（即将推出的功能）
library(mLLMCelltype)

# 使用替代端点进行注释
results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "gpt-4o",
  api_key = "your-openai-key",
  base_url = "https://your-alternative-endpoint.com/v1"  # 替代 API 端点
)
```

```python
# Python 中使用自定义 base_url（即将推出的功能）
from mllmcelltype import annotate_cell_types

# 使用替代端点进行注释
results = annotate_cell_types(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    model="gpt-4o",
    api_key="your-openai-key",
    base_url="https://your-alternative-endpoint.com/v1"  # 替代 API 端点
)
```

这个即将推出的功能将允许您使用各种兼容 OpenAI API 的替代服务，如 ellmer.tidyverse.org 等。我们正在积极开发这一功能，以便中国大陆用户能够更灵活地访问各种 LLM 服务。

#### R 中设置代理

```r
# 在 R 中设置 HTTP 代理
Sys.setenv(http_proxy = "http://proxy-server:port")
Sys.setenv(https_proxy = "http://proxy-server:port")

# 或者在 httr 请求中指定代理
library(httr)
set_config(use_proxy(url = "http://proxy-server:port"))
```

#### Python 中设置代理

```python
# 在 Python 中设置环境变量
import os
os.environ["HTTP_PROXY"] = "http://proxy-server:port"
os.environ["HTTPS_PROXY"] = "http://proxy-server:port"

# 或者在请求中指定代理
import requests
proxies = {
    "http": "http://proxy-server:port",
    "https": "http://proxy-server:port"
}
response = requests.get("https://api.example.com", proxies=proxies)
```

#### 超时和重试策略

当访问国际 API 时，设置合理的超时时间和重试策略可以提高成功率：

```r
# R 中的超时和重试设置
library(httr)
config <- httr::config(timeout = 60)  # 设置 60 秒超时

# 手动实现重试
max_retries <- 3
for (i in 1:max_retries) {
  tryCatch({
    response <- httr::POST(url, config = config, ...)
    break  # 成功则退出循环
  }, error = function(e) {
    if (i == max_retries) stop(e)
    Sys.sleep(2^i)  # 指数退避策略
  })
}
```

```python
# Python 中的超时和重试设置
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置重试策略
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# 使用设置的会话发送请求
response = http.get("https://api.example.com", timeout=30)
```

### 故障排除与常见问题

以下是中国大陆用户可能遇到的常见问题及解决方案：

#### 1. API 连接超时

**问题：** 连接到国际 API 时出现超时错误。

**解决方案：**
- 尝试使用上面提到的代理设置
- 增加超时时间（如设置为 60 秒或更长）
- 尝试使用 Azure OpenAI 或 OpenRouter 等替代服务
- 在非高峰时段尝试访问

#### 2. 模型响应慢

**问题：** API 请求成功但响应时间过长。

**解决方案：**
- 减少输入标记基因的数量（使用 `top_gene_count` 参数）
- 使用响应更快的模型（如 Claude-3.5-Haiku 或 GPT-3.5-Turbo）
- 尝试国内模型，如 DeepSeek 或 Qwen
- 将请求拆分为多个小批次处理

#### 3. API 调用失败

**问题：** API 调用返回错误代码或失败。

**解决方案：**
- 检查 API 密钥是否有效且正确输入
- 确认模型名称是否正确（注意大小写和特殊字符）
- 检查是否超出 API 调用限制或配额
- 尝试使用不同的模型或提供商

#### 4. 环境变量问题

**问题：** 程序无法读取环境变量中的 API 密钥。

**解决方案：**
- 在 R 中使用 `Sys.setenv()` 而非 `Sys.setenv`
- 确保环境变量名称正确（注意大小写）
- 直接在函数调用中提供 API 密钥参数
- 在 Python 中使用 `.env` 文件和 python-dotenv 包

#### 5. 代理设置无效

**问题：** 设置了代理但仍然无法连接。

**解决方案：**
- 确认代理服务器是否正常运行
- 检查代理格式是否正确（应为 `http://host:port` 或 `socks5://host:port`）
- 尝试在代码级别而非环境变量级别设置代理
- 尝试使用 VPN 而非 HTTP 代理

## 使用示例

### Python

```python
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging, interactive_consensus_annotation
import os

# 设置日志
setup_logging()

# 加载数据
adata = sc.read_h5ad('your_data.h5ad')

# 检查是否已计算leiden聚类，如果没有，则计算
if 'leiden' not in adata.obs.columns:
    print("计算leiden聚类...")
    # 确保数据已预处理（标准化、对数转换等）
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 如果尚未计算PCA，则计算
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.pca(adata, use_highly_variable=True)

    # 计算邻居图和leiden聚类
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.leiden(adata, resolution=0.8)
    print(f"leiden聚类完成，共有{len(adata.obs['leiden'].cat.categories)}个聚类")

# 运行差异表达分析获取标记基因
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# 为每个聚类提取标记基因
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # 为每个聚类提取前10个基因
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# 重要提示：确保使用基因符号（如KCNJ8, PDGFRA）而不是Ensembl ID（如ENSG00000176771）
# 如果您的AnnData对象存储的是Ensembl ID，请先将其转换为基因符号：
# 示例：
# if 'Gene' in adata.var.columns:  # 检查var数据框中是否有基因符号
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# 设置您想要使用的提供商的API密钥
# 您至少需要一个与计划使用的模型相对应的API密钥
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # GPT模型所需
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Claude模型所需
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # Gemini模型所需
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # 通义千问模型所需
# 其他可选模型
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # DeepSeek模型所需
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # 智谱GLM模型所需
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # Step模型所需
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # MiniMax模型所需

# 使用多个模型运行共识注释
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro", "qwen-max-2025-01-25"],
    consensus_threshold=0.7,  # 调整共识一致性阈值
    max_discussion_rounds=3   # 模型间讨论的最大轮数
)

# 从字典中获取最终共识注释
final_annotations = consensus_results["consensus"]

# 将共识注释添加到AnnData对象
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# 将不确定性指标添加到AnnData对象
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])

# 重要提示：确保在可视化前已计算UMAP坐标
# 如果您的AnnData对象中没有UMAP坐标，请计算：
if 'X_umap' not in adata.obsm:
    print("计算UMAP坐标...")
    # 确保已计算邻居图
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("UMAP坐标计算完成")

# 使用增强美学效果可视化结果
# 基础可视化
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='right', frameon=True, title='mLLMCelltype共识注释')

# 更多自定义可视化
import matplotlib.pyplot as plt

# 设置图形尺寸和样式
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# 创建更适合发表的UMAP图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data',
         frameon=True, title='mLLMCelltype共识注释',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# 可视化不确定性指标
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sc.pl.umap(adata, color='consensus_proportion', ax=ax1, title='共识比例',
         cmap='viridis', vmin=0, vmax=1, size=30)
sc.pl.umap(adata, color='entropy', ax=ax2, title='注释不确定性（香农熵）',
         cmap='magma', vmin=0, size=30)
plt.tight_layout()
```

### R

```r
# 加载所需包
library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot) # 添加用于 plot_grid

# 加载预处理的Seurat对象
pbmc <- readRDS("your_seurat_object.rds")

# 为每个聚类寻找标记基因
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# 使用多个LLM模型运行LLMCelltype注释
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 提供组织上下文
  models = c(
    "claude-3-7-sonnet-20250219",  # Anthropic
    "gpt-4o",                   # OpenAI
    "gemini-1.5-pro",           # Google
    "qwen-max-2025-01-25"       # Alibaba
  ),
  api_keys = list(
    anthropic = "your-anthropic-key",
    openai = "your-openai-key",
    gemini = "your-google-key",
    qwen = "your-qwen-key"
  ),
  top_gene_count = 10,
  controversy_threshold = 0.7
)

# 将注释添加到Seurat对象
# 从 consensus_results$final_annotations 获取细胞类型注释
cluster_to_celltype_map <- consensus_results$final_annotations

# 创建新的细胞类型标识符列
cell_types <- as.character(Idents(pbmc))
for (cluster_id in names(cluster_to_celltype_map)) {
  cell_types[cell_types == cluster_id] <- cluster_to_celltype_map[[cluster_id]]
}

# 将细胞类型注释添加到Seurat对象
pbmc$cell_type <- cell_types

# 添加不确定性指标
# 提取包含指标的详细共识结果
consensus_details <- consensus_results$initial_results$consensus_results

# 为每个聚类创建包含指标的数据框
uncertainty_metrics <- data.frame(
  cluster_id = names(consensus_details),
  consensus_proportion = sapply(consensus_details, function(res) res$consensus_proportion),
  entropy = sapply(consensus_details, function(res) res$entropy)
)

# 为每个细胞添加不确定性指标
# 使用seurat_clusters来匹配每个细胞与其对应的聚类指标
current_clusters <- pbmc$seurat_clusters
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# 使用SCpubr进行出版级可视化
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)

# 基础UMAP可视化（默认设置）
pdf("pbmc_basic_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "right") +
  ggtitle("mLLMCelltype共识注释")
dev.off()

# 更多自定义可视化（增强样式）
pdf("pbmc_custom_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  label.box = TRUE,
                  legend.position = "right",
                  pt.size = 1.0,
                  border.size = 1,
                  font.size = 12) +
  ggtitle("mLLMCelltype共识注释") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

# 使用增强型SCpubr图表可视化不确定性指标
# 获取细胞类型并创建命名的颜色调色板
cell_types <- unique(pbmc$cell_type)
color_palette <- viridis::viridis(length(cell_types))
names(color_palette) <- cell_types

# 使用SCpubr的细胞类型注释
p1 <- SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "bottom",  # 将图例放在底部
                  pt.size = 1.0,
                  label.size = 4,  # 较小的标签字体大小
                  label.box = TRUE,  # 为标签添加背景框以提高可读性
                  repel = TRUE,  # 使标签相互排斥以避免重叠
                  colors.use = color_palette,
                  plot.title = "Cell Type") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            legend.text = element_text(size = 8),
            legend.key.size = unit(0.3, "cm"),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 使用SCpubr的共识比例特征图
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "consensus_proportion",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Consensus",
                       plot.title = "Consensus Proportion",
                       sequential.palette = "YlGnBu",  # 使用黄-绿-蓝渐变色，符合Nature Methods标准
                       sequential.direction = 1,  # 从浅到深方向
                       min.cutoff = min(pbmc$consensus_proportion),  # 设置最小值
                       max.cutoff = max(pbmc$consensus_proportion),  # 设置最大值
                       na.value = "lightgrey") +  # 缺失值的颜色
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 使用SCpubr的Shannon熵特征图
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "entropy",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Entropy",
                       plot.title = "Shannon Entropy",
                       sequential.palette = "OrRd",  # 使用橙-红渐变色，符合Nature Methods标准
                       sequential.direction = -1,  # 从深到浅方向（颠倒）
                       min.cutoff = min(pbmc$entropy),  # 设置最小值
                       max.cutoff = max(pbmc$entropy),  # 设置最大值
                       na.value = "lightgrey") +  # 缺失值的颜色
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 使用相等宽度组合图表
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

#### CSV输入示例

您也可以直接使用CSV文件而不需要Seurat，这对于已经有CSV格式标记基因的情况非常有用：

```r
# 安装最新版本的mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# 加载必要的包
library(mLLMCelltype)

# 创建缓存和日志目录
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# 读取CSV文件内容
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# 跳过标题行
data_lines <- file_content[-1]

# 将数据转换为list格式，使用数字索引作为键
marker_genes_list <- list()
cluster_names <- c()

# 首先收集所有的集群名称
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# 然后创建带数字索引的marker_genes_list
for(i in 1:length(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # 第一部分是cluster名称
  cluster_name <- parts[1]

  # 使用索引作为键 (0-based索引，与Seurat兼容)
  cluster_id <- as.character(i - 1)

  # 其余部分是基因
  genes <- parts[-1]

  # 过滤掉NA和空字符串
  genes <- genes[!is.na(genes) & genes != ""]

  # 添加到marker_genes_list
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# 设置API密钥
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# 运行consensus annotation
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # 例如："human heart"
    models = c("gemini-2.0-flash",
              "gemini-1.5-pro",
              "qwen-max-2025-01-25",
              "grok-3-latest",
              "anthropic/claude-3-7-sonnet-20250219",
              "openai/gpt-4o"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir,
    log_dir = log_dir
  )

# 保存结果
saveRDS(consensus_results, "your_results.rds")

# 打印结果摘要
cat("\n结果摘要:\n")
cat("可用字段:", paste(names(consensus_results), collapse=", "), "\n\n")

# 打印最终注释
cat("最终细胞类型注释:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**CSV格式说明**：
- CSV文件第一列可以是任何值（如集群名称、数字序列如0,1,2,3或1,2,3,4等），这些值将用作索引
- 第一列的值仅用于参考，不会传递给LLM模型
- 随后的列应该包含每个集群的标记基因
- 包内已包含猫心脏组织的示例CSV文件: `inst/extdata/Cat_Heart_markers.csv`

CSV文件结构示例：
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

您可以在R脚本中使用以下代码访问示例数据：
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### 使用单个 LLM 模型

如果您只有一个 API 密钥或偏好使用特定的 LLM 模型，可以使用 `annotate_cell_types()` 函数：

```r
# 加载预处理好的 Seurat 对象
pbmc <- readRDS("your_seurat_object.rds")

# 为每个聚类找到标记基因
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# 从任何支持的提供商中选择一个模型
# 支持的模型包括：
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 通过单一API访问多种模型。格式: 'provider/model-name'
#   - OpenAI模型: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic模型: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Meta模型: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google模型: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral模型: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - 其他模型: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# 使用单个 LLM 模型进行细胞类型注释
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 提供组织上下文
  model = "claude-3-7-sonnet-20250219",  # 指定单个模型
  api_key = "your-anthropic-key",  # 直接提供 API 密钥
  top_gene_count = 10
)

# 使用 OpenRouter 免费模型
free_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "meta-llama/llama-4-maverick:free",  # 免费模型，使用:free后缀
  api_key = "your-openrouter-key",
  provider = "openrouter",  # 指定 OpenRouter 作为提供商
  top_gene_count = 10
)

# 打印结果
print(single_model_results)

# 将注释添加到 Seurat 对象
# single_model_results 是一个字符向量，每个聚类对应一个注释
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# 可视化结果
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("使用单个 LLM 模型注释的细胞类型")
```

#### 比较不同的模型

您也可以通过多次使用不同模型运行 `annotate_cell_types()` 来比较不同模型的注释结果：

```r
# 使用不同的模型进行注释
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# 为每个模型创建一个列
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )

  # 创建基于模型的列名
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))

  # 将注释添加到 Seurat 对象
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# 可视化不同模型的结果
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# 组合图表
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## 可视化示例

以下是使用mLLMCelltype和SCpubr创建的出版级可视化示例，展示了细胞类型注释和不确定性指标（共识比例和Shannon熵）：

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype可视化" width="900"/>
</div>

*图示：左图显示UMAP投影上的细胞类型注释。中图使用黄-绿-蓝渐变色显示共识比例（深蓝色表示LLM之间的一致性更强）。右图使用橙-红渐变色显示Shannon熵（深红色表示不确定性较低，浅橙色表示不确定性较高）。*

## 许可证

MIT

## 引用

如果您在研究中使用了mLLMCelltype，请引用：

```bibtex
@article{Yang2025.04.10.647852,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data},
  elocation-id = {2025.04.10.647852},
  year = {2025},
  doi = {10.1101/2025.04.10.647852},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/04/17/2025.04.10.647852},
  journal = {bioRxiv}
}
```

您也可以使用纯文本格式引用：

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852

## 贡献

我们欢迎并感谢来自社区的贡献！您可以通过多种方式为 mLLMCelltype 做出贡献：

### 报告问题

如果您遇到任何 bug、有功能请求或对使用 mLLMCelltype 有疑问，请在我们的 GitHub 仓库上[提交 issue](https://github.com/cafferychen777/mLLMCelltype/issues)。在报告 bug 时，请包括：

- 问题的清晰描述
- 复现问题的步骤
- 预期行为与实际行为
- 您的操作系统和软件包版本信息
- 任何相关的代码片段或错误信息

### Pull Requests

我们鼓励您通过 pull requests 贡献代码改进或新功能：

1. Fork 仓库
2. 为您的功能创建一个新分支（`git checkout -b feature/amazing-feature`）
3. 提交您的更改（`git commit -m '添加一些很棒的功能'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 提交 Pull Request

### 贡献领域

以下是特别欢迎贡献的领域：

- 添加对新 LLM 模型的支持
- 改进文档和示例
- 优化性能
- 添加新的可视化选项
- 扩展对特定细胞类型或组织的功能
- 将文档翻译成不同语言

### 代码风格

请遵循仓库中现有的代码风格。对于 R 代码，我们通常遵循 [tidyverse 风格指南](https://style.tidyverse.org/)。对于 Python 代码，我们遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/)。

感谢您帮助改进 mLLMCelltype！
