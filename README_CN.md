<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype标志 - 领先的多大语言模型共识框架，用于单细胞RNA测序细胞类型注释，具有95%准确率和540+GitHub星标" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=推荐使用%20mLLMCelltype%3A%20一个用于单细胞%20RNA%20测序数据细胞类型注释的多模型共识框架%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="在推特分享mLLMCelltype - 多大语言模型共识细胞注释框架"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="mLLMCelltype GitHub星标 - 540+社区成员支持的领先共识框架"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="mLLMCelltype GitHub分支 - 社区对共识细胞注释的贡献"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-加入聊天-7289da?logo=discord&logoColor=white" alt="加入mLLMCelltype Discord社区 - 获得多大语言模型共识注释支持"></a>
</div>

<div align="center">
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://www.r-pkg.org/badges/version/mLLMCelltype" alt="CRAN版本"></a>
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://cranlogs.r-pkg.org/badges/grand-total/mLLMCelltype" alt="CRAN下载量"></a>
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="许可证">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="最近提交">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="问题">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="发布版本">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
  <a href="https://www.mllmcelltype.com/"><img src="https://img.shields.io/badge/Try%20Online-mLLMCelltype-brightgreen" alt="Try Online"></a>
  <a href="https://colab.research.google.com/github/cafferychen777/mLLMCelltype/blob/main/notebooks/mLLMCelltype_Tutorial.ipynb"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Open In Colab"></a>
</div>

# mLLMCelltype: 多大语言模型共识框架用于细胞类型注释

mLLMCelltype是一个迭代式多大语言模型（Multi-LLM）共识框架，专为单细胞RNA测序（scRNA-seq）数据的精确可靠的细胞类型注释而设计。通过利用多种大语言模型（包括OpenAI GPT-5/4.1、Anthropic Claude-4/3.7/3.5、Google Gemini-2.0、X.AI Grok-3、DeepSeek-V3、阿里云 Qwen2.5、智谱 GLM-4、MiniMax、Stepfun、和 OpenRouter）的集体智慧，该框架显著提高了注释准确性，同时为生物信息学和计算生物学研究提供透明的不确定性量化。

## 摘要

mLLMCelltype是一个用于单细胞转录组学分析的开源工具，它使用多个大语言模型从基因表达数据中识别细胞类型。该软件实现了一种共识方法，其中多个模型分析相同的数据，并将它们的预测结合起来，这有助于减少错误并提供不确定性度量。mLLMCelltype与流行的单细胞分析平台（如Scanpy和Seurat）集成，使研究人员能够将其纳入现有的生物信息学工作流程中。与某些传统方法不同，它不需要参考数据集进行注释。

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

<div align="center">
  <a href="https://star-history.com/#cafferychen777/mLLMCelltype">
    <img src="https://api.star-history.com/svg?repos=cafferychen777/mLLMCelltype&type=Date" alt="Star History Chart" width="600"/>
  </a>
  <p><em>图1：mLLMCelltype GitHub星标历史，展示社区采用和随时间的增长。</em></p>
</div>

## 新闻

🚀 **Web应用程序发布（2025-06-18）**

我们很高兴地宣布mLLMCelltype Web应用程序的发布！现在您可以直接通过Web浏览器访问mLLMCelltype强大的细胞类型注释功能，无需任何安装。

**✨ 主要特性：**
- **易于使用的界面**：上传您的scRNA-seq数据，几分钟内即可获得注释
- **多LLM共识**：从各种AI模型中选择，包括GPT-4、Claude、Gemini等
- **实时处理**：通过实时更新监控注释进度
- **多种导出格式**：以CSV、TSV、Excel或JSON格式下载结果
- **无需设置**：无需安装任何包即可立即开始注释

**🌐 访问Web应用**：[https://mllmcelltype.com](https://mllmcelltype.com)

**🇨🇳 中国大陆用户特别推荐**：我们强烈建议中国大陆用户使用网页版 mLLMCelltype！使用网页版可以完全避免网络访问限制问题，无需配置 VPN 或任何特殊网络设置。网页版提供与本地版本相同的强大功能，支持所有主流 LLM 模型，包括国内可直接访问的模型。这是中国大陆用户最便捷、最稳定的使用方式。

**⚠️ Beta测试阶段**：Web应用程序目前处于beta测试阶段。我们欢迎您的反馈和建议，以帮助我们改进平台。请通过我们的[GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues)或[Discord社区](https://discord.gg/pb2aZdG4)报告任何问题或分享您的体验。

**📢 重要提示：Gemini模型迁移（2025-06-02）**

Google已经停用了几个Gemini 1.5模型，并将在2025年9月24日停用更多模型：
- **已停用**：Gemini 1.5 Pro 001、Gemini 1.5 Flash 001
- **将于2025年9月24日停用**：Gemini 1.5 Pro 002、Gemini 1.5 Flash 002、Gemini 1.5 Flash-8B -001

**推荐迁移**：使用`gemini-2.0-flash`或`gemini-2.0-flash-lite`以获得更好的性能和持续支持。别名`gemini-1.5-pro`和`gemini-1.5-flash`将继续工作到2025年9月24日，因为它们指向-002版本。

**📢 重要提示：Claude模型弃用（2025-07-21）**

Anthropic将于2025年7月21日停用以下Claude模型：
- **Claude 2**（所有版本）
- **Claude 2.1**
- **Claude 3 Sonnet**（非版本化）
- **Claude 3 Opus**（非版本化）

**推荐迁移**：
- Claude 2/2.1 → 使用 `claude-sonnet-4-5-20250929`（最新）
- Claude 3 Sonnet → 使用 `claude-sonnet-4-5-20250929`（最新）
- Claude 3 Opus → 使用 `claude-opus-4-1-20250805`（最新）

请在2025年7月21日之前更新您的代码以避免服务中断。

🎉 **2025年4月**：我们非常高兴地宣布，在预印本发布仅两周的时间里，mLLMCelltype已经突破了200个GitHub星标！我们也看到了来自各种媒体和内容创作者的大量报道。我们向所有通过星标、分享和贡献支持这个项目的人表示衷心的感谢。您的热情推动着我们继续开发和改进mLLMCelltype。

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

## 最新更新

### v1.3.1 (2025-07-16)

#### 🎯 新功能：增强的集群分析控制

- **新参数 `clusters_to_analyze`**：精确控制要分析的集群
  - 无需手动过滤输入数据
  - 保持原始集群编号不变
  - 通过只分析相关集群来减少 API 调用和成本
  - 非常适合迭代细化和亚型分析工作流

- **新参数 `force_rerun`**：强制对有争议集群进行重新分析
  - 当需要使用不同上下文进行新分析时绕过缓存
  - 对于具有组织特异性上下文的亚型识别至关重要
  - 非争议集群仍然受益于缓存性能
  - 与 `clusters_to_analyze` 完美结合，实现有针对性的工作流

使用示例：
```r
# 只分析特定的 T 细胞集群
results <- interactive_consensus_annotation(
  input = data,
  tissue_name = "人类 T 细胞亚型",
  clusters_to_analyze = c(0, 1, 7),  # T 细胞集群
  force_rerun = TRUE  # 强制重新分析以获得亚型
)
```

### v1.3.0 (2025-07-15)

#### 🎉 重大新功能
- **Per-Provider Base URL 支持**：为每个 API 提供商配置自定义端点
- **Qwen 智能端点选择**：自动在国际版/国内版端点间选择最佳连接
- **代理支持**：通过 base_urls 参数实现完整的代理功能
- **企业级支持**：支持内部 API 网关和企业代理环境

#### 新增功能
- 添加了 `base_urls` 参数到 `annotate_cell_types()` 和 `interactive_consensus_annotation()` 函数
- 实现了 BaseAPIProcessor 架构，支持所有 10 个 API 提供商的自定义端点
- 添加了 URL 工具函数用于提供商特定的端点解析
- 修复了 Qwen API 格式问题（parameters 字段和响应解析）
- 增强了中文文档，包含详细的 base_url 使用示例

#### 特别适用场景
- 🇨🇳 **中国用户**：通过代理服务器访问国际 API
- 🏢 **企业用户**：使用内部 API 网关
- 🔧 **开发测试**：连接本地或测试环境的 API 端点
- 🌐 **网络受限环境**：自定义路由解决方案

### v1.2.3 (2025-05-10)

#### Bug修复
- 修复了当API响应为NULL或无效时共识检查中的错误处理
- 改进了OpenRouter API错误响应的错误日志记录
- 在check_consensus函数中添加了强大的NULL和类型检查

#### 改进
- 增强了OpenRouter API错误的错误诊断
- 添加了API错误消息和响应结构的详细日志记录
- 改进了处理意外API响应格式时的鲁棒性

### v1.2.2 (2025-05-09)

#### Bug修复
- 修复了处理API响应时出现的"非字符参数"错误
- 为所有模型提供商的API响应添加了强大的类型检查
- 改进了意外API响应格式的错误处理

#### 改进
- 为API响应问题添加了详细的错误日志记录
- 在所有API处理函数中实现了一致的错误处理模式
- 增强了响应验证，以确保在处理之前具有正确的结构

### v1.2.1 (2025-05-01)

#### 改进
- 添加了对OpenRouter API的支持
- 通过OpenRouter添加了对免费模型的支持
- 更新了文档，包含使用OpenRouter模型的示例

### v1.2.0 (2025-04-30)

#### 特性
- 添加了细胞类型注释结果的可视化功能
- 添加了对不确定性指标可视化的支持
- 实现了改进的共识构建算法

### v1.1.5 (2025-04-27)

#### Bug修复
- 修复了在处理某些CSV输入文件时导致错误的聚类索引验证问题
- 改进了负索引的错误处理，提供更清晰的错误消息

#### 改进
- 添加了基于CSV的注释工作流程的示例脚本（cat_heart_annotation.R）
- 通过更详细的诊断增强了输入验证
- 更新了文档以澄清CSV输入格式要求

查看[NEWS.md](R/NEWS.md)获取完整的更改日志。

## 目录结构

- `R/`：R语言接口和实现
- `python/`：Python接口和实现

## 安装

### R版本

```r
# 从CRAN安装（推荐）
install.packages("mLLMCelltype")

# 或从GitHub安装开发版本
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python版本

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZgmtlaORogSy0-QsaF0CHwFWOyOD26d2?usp=sharing)

**快速开始**：无需安装即可在 Google Colab 中立即体验 mLLMCelltype！点击上方徽章打开我们的交互式笔记本，包含示例和逐步指导。

**💡 中国大陆用户使用 Google Colab 的优势**：
- 虽然 Google Colab 也需要 VPN 访问，但对 VPN 环境的纯净性要求较低，对区域限制也更宽松
- 在 Colab 笔记本中运行代码本质上是在 Google 的机房中执行，这可以有效避开某些模型厂商对网络环境的严格要求
- 许多翻墙环境无法满足 OpenAI、Anthropic 等厂商的高标准网络要求，但在 Colab 中运行可以绕过这些限制
- 一旦成功访问 Colab，即可稳定使用各种国际主流 AI 模型，无需担心本地网络环境问题

```bash
# 从PyPI安装
pip install mllmcelltype

# 或从GitHub安装（注意子目录参数）
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### 关于依赖包的重要说明

mLLMCelltype采用模块化设计，不同的LLM提供商库作为可选依赖。根据您计划使用的模型，您需要安装相应的包：

```bash
# 使用OpenAI模型(GPT-5等)
pip install "mllmcelltype[openai]"

# 使用Anthropic模型(Claude)
pip install "mllmcelltype[anthropic]"

# 使用Google模型(Gemini)
pip install "mllmcelltype[gemini]"

# 一次性安装所有可选依赖
pip install "mllmcelltype[all]"
```

如果您遇到类似`ImportError: cannot import name 'genai' from 'google'`的错误，这意味着您需要安装相应的提供商包。例如：

```bash
# 对于Google Gemini模型
pip install google-genai
```

### 支持的模型

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-5 ([API Key](https://platform.openai.com/settings/organization/billing/overview))
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
  - 格式: 'provider/model-name'（例如：'openai/gpt-5'、'anthropic/claude-opus-4.1'）
  - 提供免费模型，使用`:free`后缀（例如：'microsoft/mai-ds-r1:free'、'deepseek/deepseek-chat:free'）

## 中国大陆用户指南

为了帮助中国大陆用户更好地使用 mLLMCelltype，我们提供以下特别指南，解决可能遇到的网络限制和 API 访问问题。

### 🌟 强烈推荐：使用网页版 mLLMCelltype

**对于中国大陆用户，我们强烈推荐直接使用网页版：[https://mllmcelltype.com](https://mllmcelltype.com)**

**为什么选择网页版？**
- ✅ **无需 VPN**：完全避免访问国外 API 的网络限制问题
- ✅ **简单配置**：只需配置您选择使用的模型 API 密钥，无需复杂的网络设置
- ✅ **即开即用**：上传数据即可开始分析，无需安装任何软件包
- ✅ **功能完整**：提供与本地版本相同的所有功能
- ✅ **模型丰富**：支持国内外所有主流模型，包括 DeepSeek、Qwen、GLM 等国内模型
- ✅ **稳定可靠**：服务器已优化网络连接，确保稳定访问各种 API
- ✅ **实时更新**：自动使用最新版本，无需手动更新

如果您因特殊需求必须使用本地版本，请继续阅读以下指南。

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

- **OpenAI**：GPT-5、GPT-4.1 等系列模型
- **Anthropic**：Claude-3.7-Sonnet、Claude-3.5-Haiku 等系列模型
- **Google**：Gemini-2.5-Pro、Gemini-2.0-Flash、Gemini-2.0-Flash-Lite 等系列模型
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
)
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
        "deepseek/deepseek-chat",                # 通过 OpenRouter 访问 DeepSeek（付费）
        "anthropic/claude-sonnet-4",  # 通过 OpenRouter 访问 Anthropic（付费）
        "google/gemini-2.5-pro-preview-03-25"    # 通过 OpenRouter 访问 Google（付费）
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
    "deepseek/deepseek-chat",                # 通过 OpenRouter 访问 DeepSeek（付费）
    "anthropic/claude-sonnet-4",  # 通过 OpenRouter 访问 Anthropic（付费）
    "google/gemini-2.5-pro-preview-03-25"    # 通过 OpenRouter 访问 Google（付费）
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
  ),
  consensus_check_model = "deepseek/deepseek-chat-v3-0324:free"  # 免费模型用于共识检查
)
```

您可以在 [OpenRouter 官网](https://openrouter.ai/keys) 申请 API 密钥。

> ⚠️ **重要警告** ⚠️
>
> **即使通过 OpenRouter 访问，OpenAI 的模型（如 gpt-5、gpt-4-turbo 等）和 X.AI 的 Grok 模型仍可能受到 IP 限制。**
>
> **如果您在中国大陆访问，这些模型可能无法正常工作。您有以下选择：**
>
> **方案1：使用替代模型**
> - 国内模型：DeepSeek、Qwen、GLM-4、MiniMax、Stepfun
> - 通过 OpenRouter 访问的其他国际模型：Anthropic Claude、Google Gemini、Meta Llama 4
> - OpenRouter 免费模型：`meta-llama/llama-4-maverick:free`、`nvidia/llama-3.1-nemotron-ultra-253b-v1:free`、`microsoft/mai-ds-r1:free`
>
> **方案2：使用干净 IP 的 VPN**
> - 根据用户反馈，使用具有"干净 IP"（未被 OpenAI 或 X.AI 封禁的 IP 地址）的 VPN 服务可能解决访问限制
> - 如果您尝试使用 VPN 但仍无法访问，可能需要尝试更换不同的 VPN 服务商或节点
> - 请注意，IP 地址的状态可能随时变化，某些之前可用的 IP 可能会被封禁
>
> **方案3：使用 Google Colab**
> - Google Colab 提供了干净的 IP 地址，可以成功访问 OpenAI 和 Grok 模型
> - 我们将在本周内发布专门为 Colab 环境优化的 Jupyter Notebook，您可以直接在 Colab 中运行
> - 这是一种免费且便捷的解决方案，特别适合没有可靠 VPN 的用户

### 网络配置建议

如果您需要访问国际模型，且遇到网络连接问题，以下是一些配置建议：

#### 🎉 自定义 base_url 功能（v1.3.0 新增）

**好消息！** mLLMCelltype v1.3.0 已经支持自定义 base_url 功能，这对中国大陆用户特别有用，可以连接到代理服务器或替代 API 端点。

```r
# R 中使用自定义 base_url
library(mLLMCelltype)

# 单个代理端点应用于所有提供商
results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "gpt-5",
  api_key = "your-openai-key",
  base_urls = "https://api.your-proxy.com/v1"  # 代理端点
)

# 为不同提供商指定不同的代理端点
results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "gpt-5",
  api_key = "your-openai-key",
  base_urls = list(
    openai = "https://openai-proxy.com/v1",
    anthropic = "https://anthropic-proxy.com/v1",
    qwen = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
  )
)

# 在共识注释中使用代理
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = c("gpt-5", "claude-3-opus", "qwen-max"),
  api_keys = list(
    openai = "your-openai-key",
    anthropic = "your-anthropic-key",
    qwen = "your-qwen-key"
  ),
  base_urls = list(
    openai = "https://openai-proxy.com/v1",
    anthropic = "https://anthropic-proxy.com/v1"
    # qwen 使用默认端点（智能选择国际版/国内版）
  )
)
```

```python
# Python 中使用自定义 base_url（计划在未来版本中支持）
# 目前 Python 版本尚未支持 base_urls 参数
# 请使用 R 版本获得完整的代理支持功能
```

**功能特点：**
- ✅ **灵活配置**：可以为每个 API 提供商设置不同的代理端点
- ✅ **智能回退**：Qwen 模型支持国际版/国内版端点自动选择
- ✅ **向后兼容**：不设置 base_urls 时使用默认官方端点
- ✅ **企业友好**：支持内部 API 网关和企业代理

这个功能允许您使用各种兼容的代理服务和替代端点，为中国大陆用户提供了更灵活的访问方式。

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
# 使用mLLMCelltype和Scanpy进行单细胞RNA-seq细胞类型注释的示例
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# 注意：导入mllmcelltype时会自动配置日志记录
# 如需自定义日志记录，可以使用logging模块

# 以AnnData格式加载您的单细胞RNA-seq数据集
adata = sc.read_h5ad('your_data.h5ad')  # 用您的scRNA-seq数据集路径替换

# 如果尚未进行Leiden聚类，则执行聚类以识别细胞群体
if 'leiden' not in adata.obs.columns:
    print("计算leiden聚类以识别细胞群体...")
    # 预处理单细胞数据：标准化计数并进行log转换以进行基因表达分析
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # 标准化到每个细胞10,000个计数
        sc.pp.log1p(adata)  # Log转换标准化计数

    # 降维：为scRNA-seq数据计算PCA
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # 选择信息性基因
        sc.pp.pca(adata, use_highly_variable=True)  # 计算主成分

    # 细胞聚类：计算邻域图并执行Leiden社区检测
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # 构建用于聚类的KNN图
    sc.tl.leiden(adata, resolution=0.8)  # 使用Leiden算法识别细胞群体
    print(f"Leiden聚类完成，识别出{len(adata.obs['leiden'].cat.categories)}个不同的细胞群体")

# 使用差异表达分析为每个细胞簇识别标记基因
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # 用于标记检测的Wilcoxon秩和检验

# 提取每个细胞簇的顶部标记基因用于细胞类型注释
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # 选择每个簇的前10个差异表达基因作为标记
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# 重要提示：mLLMCelltype需要基因符号（例如KCNJ8、PDGFRA）而不是Ensembl ID（例如ENSG00000176771）
# 如果您的AnnData对象使用Ensembl ID，请将其转换为基因符号以获得准确的注释：
# 示例转换代码：
# if 'Gene' in adata.var.columns:  # 检查元数据中是否有基因符号
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# 重要提示：mLLMCelltype需要数字聚类ID
# 'cluster'列必须包含数字值或可以转换为数字的值。
# 非数字聚类ID（例如"cluster_1"、"T_cells"、"7_0"）可能会导致错误或意外行为。
# 如果您的数据包含非数字聚类ID，请创建原始ID和数字ID之间的映射：
# 示例标准化代码：
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# 配置用于共识注释的大语言模型的API密钥
# 至少需要一个API密钥用于多LLM共识注释
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # 用于GPT-5/4.1模型（推荐）
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # 用于Claude-3.7/3.5模型
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # 用于Google Gemini-2.5模型
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # 用于阿里巴巴Qwen2.5模型
# 用于增强共识多样性的其他可选LLM提供商：
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # 用于DeepSeek-V3模型
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # 用于智谱GLM-4模型
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # 用于Stepfun模型
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # 用于MiniMax模型
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # 通过OpenRouter访问多种模型

# 使用迭代讨论执行多LLM共识细胞类型注释
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # 每个簇的标记基因字典
    species="human",            # 指定生物体以进行适当的细胞类型注释
    tissue="blood",            # 指定组织环境以获得更准确的注释
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "qwen-max-2025-01-25"],  # 用于共识的多个LLM
    consensus_threshold=1,     # 共识一致所需的最小比例
    max_discussion_rounds=3    # 模型之间进行改进的讨论轮数
)

# 或者，使用OpenRouter通过单一API访问多种模型
# 这对于使用:free后缀访问免费模型特别有用
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 使用免费OpenRouter模型的示例（无需积分）
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Meta Llama 4 Maverick（免费）
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"},  # NVIDIA Nemotron Ultra 253B（免费）
        {"provider": "openrouter", "model": "deepseek/deepseek-chat-v3-0324:free"},   # DeepSeek Chat v3（免费）
        {"provider": "openrouter", "model": "microsoft/mai-ds-r1:free"}               # Microsoft MAI-DS-R1（免费）
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# 从多LLM讨论中获取最终共识细胞类型注释
final_annotations = consensus_results["consensus"]

# 将共识细胞类型注释整合到原始AnnData对象中
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# 添加不确定性量化指标以评估注释置信度
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])  # 一致性水平
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])  # 注释不确定性

# 准备可视化：如果尚未计算UMAP嵌入，则计算
# UMAP提供细胞群体的2D表示用于可视化
if 'X_umap' not in adata.obsm:
    print("计算UMAP坐标...")
    # 确保先计算邻居
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

# 创建更适合发表的UMAP
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

### 使用单个免费OpenRouter模型

对于喜欢使用单个模型的更简单方法的用户，通过OpenRouter的Microsoft MAI-DS-R1免费模型提供了出色的结果：

```python
import os
from mllmcelltype import annotate_clusters

# 注意：自动配置日志记录

# 设置您的OpenRouter API密钥
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 定义每个簇的标记基因
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T细胞
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B细胞
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # 单核细胞
}

# 使用Microsoft MAI-DS-R1免费模型进行注释
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='microsoft/mai-ds-r1:free'  # 免费模型
)

# 打印注释
for cluster, annotation in annotations.items():
    print(f"簇 {cluster}: {annotation}")
```

这种方法快速、准确，不需要任何API积分，使其成为快速分析或API访问有限时的理想选择。

#### 从AnnData对象提取标记基因

如果您使用Scanpy与AnnData对象，您可以轻松地直接从`rank_genes_groups`结果中提取标记基因：

```python
import os
import scanpy as sc
from mllmcelltype import annotate_clusters

# 注意：自动配置日志记录

# 设置您的OpenRouter API密钥
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 加载并预处理您的数据
adata = sc.read_h5ad('your_data.h5ad')

# 如果尚未完成，执行预处理和聚类
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata)

# 为每个簇寻找标记基因
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# 为每个簇提取顶部标记基因
marker_genes = {
    cluster: adata.uns['rank_genes_groups']['names'][cluster][:10].tolist()
    for cluster in adata.obs['leiden'].cat.categories
}

# 使用Microsoft MAI-DS-R1免费模型进行注释
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',  # 根据您的组织类型调整
    provider='openrouter',
    model='microsoft/mai-ds-r1:free'  # 免费模型
)

# 将注释添加到AnnData对象
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# 可视化结果
sc.pl.umap(adata, color='cell_type', legend_loc='on data',
           frameon=True, title='MAI-DS-R1注释的细胞类型')
```

此方法自动从`rank_genes_groups`结果中提取每个簇的顶部差异表达基因，使其易于将mLLMCelltype整合到您的Scanpy工作流程中。

### R

> **注意**：有关更详细的R教程和文档，请访问[mLLMCelltype文档网站](https://cafferyang.com/mLLMCelltype/)。

#### 使用Seurat对象

```r
# 加载所需包
library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot) # 添加用于 plot_grid

# 加载预处理的Seurat对象
pbmc <- readRDS("your_seurat_object.rds")

# 如果从原始数据开始，执行预处理步骤
# pbmc <- NormalizeData(pbmc)
# pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
# pbmc <- ScaleData(pbmc)
# pbmc <- RunPCA(pbmc)
# pbmc <- FindNeighbors(pbmc, dims = 1:10)
# pbmc <- FindClusters(pbmc, resolution = 0.5)
# pbmc <- RunUMAP(pbmc, dims = 1:10)

# 为每个聚类寻找标记基因
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# 设置缓存目录以加快处理速度
cache_dir <- "./mllmcelltype_cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# 从任何支持的提供商中选择一个模型
# 支持的模型包括：
# - OpenAI: 'gpt-5', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 通过单一API访问多个提供商的模型。格式：'provider/model-name'
#   - OpenAI模型：'openai/gpt-5', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic模型：'anthropic/claude-sonnet-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4', 'anthropic/claude-opus-4.1'
#   - Meta模型：'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google模型：'google/gemini-2.5-pro', 'google/gemini-2.5-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral模型：'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - 其他模型：'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# 使用多个LLM模型运行LLMCelltype注释
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 提供组织上下文
  models = c(
    "claude-sonnet-4-5-20250929",  # Anthropic
    "gpt-5",                   # OpenAI
    "gemini-2.5-pro",           # Google
    "qwen-max-2025-01-25"       # Alibaba
  ),
  api_keys = list(
    anthropic = "your-anthropic-key",
    openai = "your-openai-key",
    gemini = "your-google-key",
    qwen = "your-qwen-key"
  ),
  top_gene_count = 10,
  controversy_threshold = 1.0,
  entropy_threshold = 1.0,
  cache_dir = cache_dir
)

# 打印结果结构以了解数据
print("consensus_results中的可用字段：")
print(names(consensus_results))

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
# 注意：seurat_clusters是FindClusters()函数自动创建的元数据列
# 它包含聚类期间分配给每个细胞的聚类ID
# 这里我们使用它将聚类级别的指标（consensus_proportion和entropy）映射到各个细胞

# 如果您没有seurat_clusters列（例如，如果您使用了不同的聚类方法），
# 您可以使用活动标识（Idents）或元数据中包含聚类ID的任何其他列：
# 选项1：使用活动标识
# current_clusters <- as.character(Idents(pbmc))
# 选项2：使用包含聚类ID的另一个元数据列
# current_clusters <- pbmc$your_cluster_column

# 对于这个示例，我们使用标准的seurat_clusters列：
current_clusters <- pbmc$seurat_clusters  # 获取每个细胞的聚类ID

# 将每个细胞的聚类ID与uncertainty_metrics中的相应指标匹配
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# 保存结果以备将来使用
saveRDS(consensus_results, "pbmc_mLLMCelltype_results.rds")
saveRDS(pbmc, "pbmc_annotated.rds")

# 使用SCpubr进行出版级可视化
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)
library(viridis)  # 用于调色板

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
                       sequential.palette = "YlGnBu",  # 黄-绿-蓝渐变，遵循Nature Methods标准
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
                       sequential.palette = "OrRd",  # 橙-红渐变，遵循Nature Methods标准
                       sequential.direction = -1,  # 从深到浅方向（反转）
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

#### 使用CSV输入

您也可以直接使用CSV文件而不需要Seurat，这对于已经有CSV格式标记基因的情况非常有用：

```r
# 安装最新版本的mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# 加载必要的包
library(mLLMCelltype)

# 配置统一日志记录（可选 - 如果未指定则使用默认值）
configure_logger(level = "INFO", console_output = TRUE, json_format = TRUE)

# 创建缓存目录
cache_dir <- "path/to/your/cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

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

# 使用付费模型运行共识注释
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # 例如："human heart"
    models = c("gemini-2.5-pro",
              "gemini-2.5-flash",
              "qwen-max-2025-01-25",
              "grok-3-latest",
              "claude-sonnet-4-5-20250929",
              "gpt-5"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir
  )

# 或者，使用免费OpenRouter模型（无需积分）
# 将OpenRouter API密钥添加到api_keys列表
api_keys$openrouter <- "your-openrouter-api-key"

# 使用免费模型运行共识注释
free_consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # 例如："human heart"
    models = c(
      "meta-llama/llama-4-maverick:free",      # Meta Llama 4 Maverick（免费）
      "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",  # NVIDIA Nemotron Ultra 253B（免费）
      "deepseek/deepseek-chat-v3-0324:free",   # DeepSeek Chat v3（免费）
      "microsoft/mai-ds-r1:free"               # Microsoft MAI-DS-R1（免费）
    ),
    api_keys = api_keys,
    consensus_check_model = "deepseek/deepseek-chat-v3-0324:free",  # 用于共识检查的免费模型
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 2,
    cache_dir = cache_dir
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
- CSV文件的第一列中的值将用作索引（这些可以是聚类名称、像0,1,2,3或1,2,3,4等数字）
- 第一列中的值仅用于参考，不会传递给LLM
- 后续列应包含每个聚类的标记基因
- 软件包的`inst/extdata/Cat_Heart_markers.csv`中包含了猫心脏组织的示例CSV文件

CSV结构示例：
```
cluster,gene
0,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
1,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
2,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
3,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

您可以在R脚本中使用以下代码访问示例数据：
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### 使用单个 LLM 模型

如果您只想使用单个LLM模型而不是共识方法，请使用`annotate_cell_types()`函数。当您只能访问一个API密钥或偏好特定模型时，这很有用：

```r
# 加载所需包
library(mLLMCelltype)
library(Seurat)

# 加载预处理的Seurat对象
pbmc <- readRDS("your_seurat_object.rds")

# 为每个聚类寻找标记基因
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# 从任何支持的提供商中选择一个模型
# 支持的模型包括：
# - OpenAI: 'gpt-5', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 通过单一API访问多个提供商的模型。格式：'provider/model-name'
#   - OpenAI模型：'openai/gpt-5', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic模型：'anthropic/claude-sonnet-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4', 'anthropic/claude-opus-4.1'
#   - Meta模型：'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google模型：'google/gemini-2.5-pro', 'google/gemini-2.5-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral模型：'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - 其他模型：'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# 使用单个LLM模型运行细胞类型注释
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 提供组织上下文
  model = "claude-sonnet-4-5-20250929",  # 指定单个模型
  api_key = "your-anthropic-key",  # 直接提供API密钥
  top_gene_count = 10
)

# 使用免费OpenRouter模型
free_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "meta-llama/llama-4-maverick:free",  # 带有:free后缀的免费模型
  api_key = "your-openrouter-key",
  top_gene_count = 10
)

# 打印结果
print(single_model_results)

# 将注释添加到Seurat对象
# single_model_results是一个字符向量，每个聚类对应一个注释
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# 可视化结果
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("使用单个LLM模型注释的细胞类型")
```

#### 比较不同的模型

您也可以通过多次使用不同模型运行`annotate_cell_types()`来比较不同模型的注释：

```r
# 定义要测试的模型
models_to_test <- c(
  "claude-sonnet-4-5-20250929",  # Anthropic
  "gpt-5",                      # OpenAI
  "gemini-2.5-pro",              # Google
  "qwen-max-2025-01-25"          # Alibaba
)

# 不同提供商的API密钥
api_keys <- list(
  anthropic = "your-anthropic-key",
  openai = "your-openai-key",
  gemini = "your-gemini-key",
  qwen = "your-qwen-key"
)

# 测试每个模型并存储结果
results <- list()
for (model in models_to_test) {
  provider <- get_provider(model)
  api_key <- api_keys[[provider]]

  # 运行注释
  results[[model]] <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = model,
    api_key = api_key,
    top_gene_count = 10
  )

  # 添加到Seurat对象
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", model))
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results[[model]])-1)),
    to = results[[model]]
  )
}
```

### 高级共识配置：指定共识检查模型

`consensus_check_model`参数（R）/`consensus_model`参数（Python）允许您指定用于共识检查和讨论调节的LLM模型。这个参数对共识注释的准确性**至关重要**，因为共识检查模型：

1. 评估不同细胞类型注释之间的语义相似性
2. 计算共识指标（比例和熵）
3. 调节和综合有争议聚类的模型之间的讨论
4. 在模型不同意时做出最终决定

**⚠️ 重要提示：我们强烈建议使用可用的最有能力的模型进行共识检查，因为这直接影响注释质量。**

#### 推荐的共识检查模型（按性能排序）

1. **Anthropic Claude模型**（最高推荐）
   - `claude-sonnet-4-5-20250929` - **总体性能最佳**（最新最智能）
   - `claude-opus-4-1-20250805` - 适合复杂推理任务

2. **OpenAI模型**
   - `o1` / `o1-pro` - 高级推理能力
   - `gpt-5` - 在各种细胞类型中的强大性能
   - `gpt-4.1` - 最新的GPT-4变体

3. **Google Gemini模型**
   - `gemini-2.5-pro` - 顶级性能，增强推理能力
   - `gemini-2.5-flash` - 性能与速度的绝佳平衡
   - `gemini-2.0-flash` - 良好的性能，处理速度更快

4. **其他高性能模型**
   - `deepseek-r1` / `deepseek-reasoner` - 强大的推理能力
   - `qwen-max-2025-01-25` - 在科学背景下表现出色
   - `grok-3-latest` - 高级语言理解

#### R包使用

```r
# 示例1：使用最佳可用模型进行共识检查（推荐）
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human brain",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # 使用最有能力的模型
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# 示例2：当Claude Opus不可用时使用高性能模型
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "mouse liver",
  models = c("gpt-5", "gemini-2.5-pro", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # 替代高性能模型
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# 示例3：为复杂情况使用OpenAI的推理模型
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human immune cells",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro"),
  api_keys = api_keys,
  consensus_check_model = "o1",  # OpenAI的高级推理模型
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# ⚠️ 不推荐：避免使用能力较弱或免费模型进行共识检查
# 因为这可能会显著降低注释准确性
```

#### 自定义API端点（base_urls参数）

**新功能**：mLLMCelltype现在支持为每个模型提供商配置自定义API端点，这对中国大陆用户和企业用户特别有用。

##### 使用场景

1. **中国大陆用户**：通过代理服务器访问国际API
2. **企业用户**：使用内部API网关
3. **开发测试**：连接本地或测试环境的API端点
4. **Qwen模型**：选择国际版或国内版API端点

##### 基本用法

```r
# 单个注释示例 - 使用代理端点
result <- annotate_cell_types(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  model = "gpt-5",
  api_key = "your-api-key",
  base_urls = "https://api.openai-proxy.com/v1"  # 所有提供商使用相同代理
)

# 为不同提供商指定不同的端点
result <- annotate_cell_types(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  model = "gpt-5",
  api_key = "your-api-key",
  base_urls = list(
    openai = "https://openai-proxy.com/v1",
    anthropic = "https://anthropic-proxy.com/v1",
    qwen = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"  # 国际版
  )
)
```

##### 共识注释中的使用

```r
# 在共识注释中使用自定义端点
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  models = c("gpt-5", "claude-3-opus", "qwen-max-2025-01-25"),
  api_keys = list(
    openai = Sys.getenv("OPENAI_API_KEY"),
    anthropic = Sys.getenv("ANTHROPIC_API_KEY"),
    qwen = Sys.getenv("QWEN_API_KEY")
  ),
  base_urls = list(
    openai = "https://openai-proxy.com/v1",
    anthropic = "https://anthropic-proxy.com/v1",
    qwen = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
  )
)
```

##### Qwen模型的智能端点选择

**新功能**：Qwen模型现在支持智能端点选择，自动检测最佳可用端点！

Qwen有两个API端点：
- **国际版**：`https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
- **国内版**：`https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`

**智能选择机制**：
1. 🔍 首先尝试国际版端点
2. 🔄 如果国际版无法访问，自动切换到国内版
3. 💾 缓存工作的端点，避免重复检测
4. ⚡ 后续调用直接使用已验证的端点

```r
# 🚀 推荐用法：让系统自动选择最佳端点
result <- annotate_cell_types(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  model = "qwen-max-2025-01-25",
  api_key = "your-qwen-api-key"
  # 无需配置base_urls，系统自动选择最佳端点
)

# 🎯 共识注释中的智能端点选择
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  models = c("gpt-5", "claude-3-opus", "qwen-max-2025-01-25"),
  api_keys = list(
    openai = "your-openai-key",
    anthropic = "your-anthropic-key",
    qwen = "your-qwen-key"
  )
  # Qwen会自动选择最佳端点，其他模型使用默认端点
)

# 🔧 手动指定端点（如果需要）
result <- annotate_cell_types(
  input = marker_genes_list,
  tissue_name = "human PBMC",
  model = "qwen-max-2025-01-25",
  api_key = "your-qwen-api-key",
  base_urls = list(qwen = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")  # 强制使用国内版
)
```

**优势**：
- ✅ **零配置**：无需手动选择端点
- ✅ **自动适应**：根据网络环境自动选择
- ✅ **高效缓存**：避免重复检测
- ✅ **向后兼容**：支持手动指定端点

##### 常用代理配置示例

```r
# 中国用户常用代理配置
china_proxy_urls <- list(
  openai = "https://api.openai-proxy.com/v1",
  anthropic = "https://api.anthropic-proxy.com/v1",
  deepseek = "https://api.deepseek.com/v1",  # DeepSeek通常可直接访问
  gemini = "https://generativelanguage-proxy.googleapis.com/v1beta/models"
  # qwen 默认使用国际版，通常无需代理
)

# 企业内部网关配置
enterprise_urls <- list(
  openai = "https://internal-gateway.company.com/openai/v1",
  anthropic = "https://internal-gateway.company.com/anthropic/v1"
)
```

#### Python包使用

```python
# 示例1：使用最佳可用模型进行共识检查（推荐）
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "qwen-max-2025-01-25"],
    consensus_model="claude-sonnet-4-5-20250929",  # 使用最有能力的模型
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 示例2：使用字典格式与高性能模型
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5", "gemini-2.5-pro", "qwen-max-2025-01-25"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 示例3：使用Google的最新模型进行共识
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "qwen-max-2025-01-25"],
    consensus_model={"provider": "google", "model": "gemini-2.5-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 示例4：默认行为（使用Qwen与高性能回退）
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro"],
    # 如果未指定，默认为qwen-max-2025-01-25（高性能模型）
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### 共识模型选择的最佳实践

1. **优先考虑准确性而非成本**：共识检查模型在确定最终注释中起着至关重要的作用。在这里使用能力较弱的模型可能会损害整个注释过程。

2. **模型可用性**：确保您有API访问权限以使用您选择的共识模型。如果主要选择不可用，系统将使用回退模型。

3. **一致性**：在项目中的所有共识检查中使用相同的高性能模型，以确保一致的评估标准。

4. **复杂组织**：对于具有挑战性的组织（例如，大脑、免疫系统），考虑使用最先进的模型，如Claude Opus、O1或Gemini 2.5 Pro。

5. **默认行为**：
   - R：如果未指定，则使用`models`列表中的第一个模型
   - Python：默认为`qwen-max-2025-01-25`（高性能模型），以`claude-sonnet-4-5-20250929`作为回退

#### 为什么模型质量对共识检查很重要

共识检查模型必须：
- 准确评估不同细胞类型名称之间的语义相似性（例如，识别"T淋巴细胞"和"T细胞"指的是同一细胞类型）
- 理解生物学背景和层次关系
- 综合来自多个模型的讨论以得出准确的结论
- 为下游分析提供可靠的置信度指标

为这些关键任务使用能力较弱的模型可能导致：
- 对有争议聚类的误识别
- 不正确的共识计算
- 模型之间分歧的解决不当
- 最终，细胞类型注释准确性降低

## 可视化示例

### 细胞类型注释可视化

以下是使用mLLMCelltype和SCpubr创建的出版级可视化示例，展示了细胞类型注释和不确定性指标（共识比例和Shannon熵）：

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype可视化" width="900"/>
</div>

*图示：左图显示UMAP投影上的细胞类型注释。中图使用黄-绿-蓝渐变色显示共识比例（深蓝色表示LLM之间的一致性更强）。右图使用橙-红渐变色显示Shannon熵（深红色表示不确定性较低，浅橙色表示不确定性较高）。*

### 标记基因可视化

mLLMCelltype现在包括增强的标记基因可视化功能，可与共识注释工作流程无缝集成：

```r
# 加载所需库
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# 运行共识注释后
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-sonnet-4.5", "openai/gpt-5"),
  api_keys = list(openrouter = "your_api_key")
)

# 使用Seurat创建标记基因可视化
# 将共识注释添加到Seurat对象
cluster_ids <- as.character(Idents(pbmc_data))
cell_type_annotations <- consensus_results$final_annotations[cluster_ids]

# 处理任何缺失的注释
if (any(is.na(cell_type_annotations))) {
  na_mask <- is.na(cell_type_annotations)
  cell_type_annotations[na_mask] <- paste("Cluster", cluster_ids[na_mask])
}

# 添加到Seurat对象
pbmc_data@meta.data$cell_type_consensus <- cell_type_annotations

# 创建标记基因点图
DotPlot(pbmc_data, 
        features = top_markers,
        group.by = "cell_type_consensus") + 
  RotatedAxis()

# 创建标记基因热图
DoHeatmap(pbmc_data, 
          features = top_markers,
          group.by = "cell_type_consensus")
```

**标记基因可视化的主要特性：**

- **点图（DotPlot）**：显示表达每个基因的细胞百分比（点大小）和平均表达水平（颜色强度）
- **热图（Heatmap）**：显示缩放的表达值，可对基因和细胞类型进行聚类
- **无缝集成**：直接将共识注释结果添加到Seurat对象中使用
- **标准Seurat函数**：使用熟悉的Seurat可视化函数，保持一致性

有关详细说明和高级自定义选项，请参阅[可视化指南](R/vignettes/06-visualization-guide.html)。

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

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. [在bioRxiv上阅读我们的完整研究论文](https://doi.org/10.1101/2025.04.10.647852)

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

### 社区

加入我们的[Discord社区](https://discord.gg/pb2aZdG4)，获取有关mLLMCelltype的实时更新、提问、分享您的经验，或与其他用户和开发人员合作。这是与团队和其他从事单细胞RNA-seq分析的用户联系的好地方。

感谢您帮助改进 mLLMCelltype！