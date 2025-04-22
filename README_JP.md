<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltypeは、単一細胞RNAシーケンシングデータにおける細胞タイプアノテーションのための反復型マルチLLMコンセンサスフレームワークです。複数の大規模言語モデル（GPT、Claude、Gemini、Grok、DeepSeek、Qwenなど）の相補的な強みを活用することで、このフレームワークはアノテーションの精度を大幅に向上させながら、透明性のある不確実性の定量化を提供します。

## 主な特徴

- **マルチLLMコンセンサスアーキテクチャ**: 多様なLLMからの集合知を活用し、単一モデルの限界とバイアスを克服
- **構造化された審議プロセス**: LLMが複数回の協調的な議論を通じて推論を共有し、証拠を評価し、アノテーションを改善することを可能に
- **透明な不確実性の定量化**: 専門家のレビューを必要とするあいまいな細胞集団を特定するための定量的指標（コンセンサス比率とシャノンエントロピー）を提供
- **ハルシネーション（幻覚）の削減**: モデル間の審議が批判的評価を通じて不正確または裏付けのない予測を積極的に抑制
- **入力ノイズに対する堅牢性**: 集合的なエラー修正により、不完全なマーカー遺伝子リストでも高い精度を維持
- **階層的アノテーションのサポート**: 親子の一貫性を持つマルチ解像度分析のためのオプション拡張
- **参照データセット不要**: 事前トレーニングや参照データなしで正確なアノテーションを実行
- **完全な推論チェーン**: 透明な意思決定のために審議プロセス全体を文書化
- **シームレスな統合**: 標準的なScanpy/Seuratワークフローとマーカー遺伝子出力と直接連携
- **モジュラー設計**: 新しいLLMが利用可能になった時に簡単に組み込み可能

### サポートされているモデル

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([APIキー](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([APIキー](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([APIキー](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([APIキー](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([APIキー](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([APIキー](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([APIキー](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([APIキー](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([APIキー](https://accounts.x.ai/))
- **OpenRouter**: 単一APIで複数のモデルにアクセス ([APIキー](https://openrouter.ai/keys))
  - OpenAI、Anthropic、Meta、Google、Mistralなどのモデルをサポート
  - 形式: 'provider/model-name'（例: 'openai/gpt-4o'、'anthropic/claude-3-opus'）

## ディレクトリ構造

- `R/`: R言語インターフェースと実装
- `python/`: Pythonインターフェースと実装

## インストール

### Rバージョン

```r
# GitHubからインストール
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Pythonバージョン

```bash
pip install mllmcelltype
```

## クイックスタート

### Rでの使用例

```r
library(mLLMCelltype)
library(Seurat)

# マーカー遺伝子リストを準備
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# 細胞タイプアノテーションを実行
consensus_results <- interactive_consensus_annotation(
  input = markers,
  tissue_name = "human PBMC",
  models = c("gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.0-pro"),
  api_keys = list(
    openai = "your_openai_api_key",
    anthropic = "your_anthropic_api_key",
    gemini = "your_gemini_api_key"
  ),
  top_gene_count = 10
)

# 結果を確認
print(consensus_results$final_annotations)

# Seuratオブジェクトにアノテーションを追加
current_clusters <- as.character(Idents(seurat_obj))
cell_types <- as.character(current_clusters)
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
seurat_obj$cell_type <- cell_types
```

### Pythonでの使用例

```python
import scanpy as sc
import mllmcelltype as mct

# AnnDataオブジェクトを読み込み
adata = sc.read_h5ad("your_data.h5ad")

# クラスタリングを実行（既に実行済みの場合はスキップ）
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# マーカー遺伝子を特定
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# マーカー遺伝子をmLLMCelltypeの入力形式に変換
markers_dict = mct.utils.convert_scanpy_markers(adata)

# 細胞タイプアノテーションを実行
consensus_results = mct.annotate.interactive_consensus_annotation(
    input=markers_dict,
    tissue_name="human PBMC",
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro"],
    api_keys={
        "openai": "your_openai_api_key",
        "anthropic": "your_anthropic_api_key",
        "gemini": "your_gemini_api_key"
    },
    top_gene_count=10
)

# AnnDataオブジェクトにアノテーションを追加
adata.obs["cell_type"] = adata.obs["leiden"].map(
    lambda x: consensus_results["final_annotations"].get(x, "Unknown")
)
```

## 不確実性の可視化

mLLMCelltypeは、アノテーションの不確実性を定量化するための2つのメトリクスを提供します：

1. **コンセンサス比率**: 特定の予測に同意するモデルの割合
2. **シャノンエントロピー**: 予測分布の不確実性を測定

これらのメトリクスを可視化するには：

```r
library(Seurat)
library(ggplot2)
library(cowplot)
library(SCpubr)

# 不確実性メトリクスを計算
uncertainty_metrics <- calculate_uncertainty_metrics(consensus_results)

# Seuratオブジェクトに不確実性メトリクスを追加
current_clusters <- as.character(Idents(pbmc))
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# 細胞タイプアノテーションを可視化
p1 <- SCpubr::do_DimPlot(sample = pbmc, 
                       group.by = "cell_type",
                       label = TRUE, 
                       repel = TRUE,
                       pt.size = 0.1) +
      ggtitle("Cell Type Annotations") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# コンセンサス比率を可視化
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "consensus_proportion",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("yellow", "green", "blue"),
                         limits = c(min(pbmc$consensus_proportion),  # 最小値を設定
                                   max(pbmc$consensus_proportion)),  # 最大値を設定
                         na.value = "lightgrey") +  # 欠損値の色
      ggtitle("Consensus Proportion") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# エントロピーを可視化
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "entropy",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("darkred", "red", "orange"),
                         limits = c(min(pbmc$entropy),  # 最小値を設定
                                   max(pbmc$entropy)),  # 最大値を設定
                         na.value = "lightgrey") +  # 欠損値の色
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 等幅でプロットを結合
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

### 単一LLMモデルの使用

APIキーが1つしかない場合や特定のLLMモデルを使用したい場合は、`annotate_cell_types()`関数を使用できます：

```r
# 前処理済みのSeuratオブジェクトを読み込む
pbmc <- readRDS("your_seurat_object.rds")

# 各クラスターのマーカー遣伏子を見つける
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# サポートされている任意のプロバイダーからモデルを選択
# サポートされているモデルの例：
# - Anthropic: "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-latest"
# - OpenAI: "gpt-4o"
# - Google: "gemini-1.5-pro", "gemini-2.0-flash"
# - DeepSeek: "deepseek-chat", "deepseek-reasoner"
# - Alibaba: "qwen-max-2025-01-25"
# - X.AI: "grok-3", "grok-3-mini"
# - Zhipu: "glm-4-plus", "glm-3-turbo"
# - MiniMax: "minimax-text-01"
# - Stepfun: "step-2-16k", "step-2-mini"

# 単一LLMモデルで細胞タイプアノテーションを実行
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 組織のコンテキストを提供
  model = "claude-3-7-sonnet-20250219",  # 単一モデルを指定
  api_key = "your-anthropic-key",  # APIキーを直接提供
  top_gene_count = 10
)

# 結果を表示
print(single_model_results)

# Seuratオブジェクトにアノテーションを追加
# single_model_resultsは各クラスターに対して1つのアノテーションを持つ文字ベクトル
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# 結果を可視化
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("単一LLMモデルによる細胞タイプアノテーション")
```

#### 異なるモデルの比較

異なるモデルで`annotate_cell_types()`を複数回実行することで、異なるモデルのアノテーションを比較することもできます：

```r
# 異なるモデルを使用してアノテーションを行う
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# 各モデルの結果を格納する列を作成
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )
  
  # モデルに基づいた列名を作成
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))
  
  # Seuratオブジェクトにアノテーションを追加
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# 異なるモデルの結果を可視化
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# プロットを結合
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## 可視化の例

以下は、mLLMCelltypeとSCpubrを使用して作成された出版品質の可視化の例で、細胞タイプアノテーションと不確実性メトリクス（コンセンサス比率とシャノンエントロピー）を示しています：

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*図：左パネルはUMAP投影上の細胞タイプアノテーションを示しています。中央パネルは黄色-緑-青のグラデーションを使用してコンセンサス比率を表示しています（濃い青はLLM間の強い合意を示します）。右パネルはオレンジ-赤のグラデーションを使用してシャノンエントロピーを示しています（濃い赤は低い不確実性、明るいオレンジは高い不確実性を示します）。*

## 引用

研究でmLLMCelltypeを使用する場合は、以下を引用してください：

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

プレーンテキスト形式での引用：

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852