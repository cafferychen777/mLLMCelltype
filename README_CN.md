# mLLMCelltype

mLLMCelltype是一个迭代式多大语言模型（Multi-LLM）共识框架，专为单细胞RNA测序数据的细胞类型注释而设计。通过利用多种大语言模型（如GPT、Claude、Gemini、Qwen等）的互补优势，该框架显著提高了注释准确性，同时提供透明的不确定性量化。

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

# 运行差异表达分析获取标记基因
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# 为每个聚类提取标记基因
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # 为每个聚类提取前10个基因
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# 设置不同提供商的API密钥
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"

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
# 创建映射字典，正确映射聚类ID到细胞类型
cluster_to_celltype_map <- setNames(
  unlist(consensus_results$final_annotations),
  names(consensus_results$final_annotations)
)

# 获取每个细胞的当前聚类ID
current_clusters <- as.character(Idents(pbmc))

# 将细胞类型注释添加到Seurat对象
pbmc$cell_type <- cluster_to_celltype_map[current_clusters]

# 添加不确定性指标
pbmc$consensus_proportion <- consensus_results$consensus_results[current_clusters]$consensus_proportion
pbmc$entropy <- consensus_results$consensus_results[current_clusters]$entropy

# 使用SCpubr进行出版级可视化
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)

# 基础UMAP可视化
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "right") +
  ggtitle("mLLMCelltype共识注释")

# 更多自定义可视化
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  label.box = FALSE,
                  legend.position = "right",
                  pt.size = 1.2,
                  border.size = 1,
                  font.size = 14) +
  ggtitle("mLLMCelltype共识注释")
```

## 许可证

MIT

## 引用

如果您在研究中使用了mLLMCelltype，请引用：

```bibtex
@software{mllmcelltype2025,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {mLLMCelltype: An iterative multi-LLM consensus framework for cell type annotation},
  url = {https://github.com/cafferychen777/mLLMCelltype},
  version = {1.0.2},
  year = {2025}
}
```
