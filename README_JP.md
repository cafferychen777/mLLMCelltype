<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - 単一細胞 RNA シーケンシングデータのマルチ大規模言語モデル細胞タイプアノテーションフレームワーク" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=mLLMCelltype%3A%20単一細胞%20RNA%20シーケンシングデータの細胞タイプアノテーションのためのマルチLLMコンセンサスフレームワーク%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-チャットに参加-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
</div>

# mLLMCelltype: 単一細胞 RNA シーケンシングのためのマルチ大規模言語モデルコンセンサスフレームワーク

mLLMCelltypeは、単一細胞RNAシーケンシング(scRNA-seq)データにおける正確で信頼性の高い細胞タイプアノテーションのための高度な反復型マルチLLMコンセンサスフレームワークです。複数の大規模言語モデル（OpenAI GPT-4o/4.1、Anthropic Claude-3.7/3.5、Google Gemini-2.0、X.AI Grok-3、DeepSeek-V3、Alibaba Qwen2.5、Zhipu GLM-4、MiniMax、Stepfun、OpenRouterなど）の集合知能を活用することで、このフレームワークはアノテーションの精度を大幅に向上させながら、バイオインフォマティクスおよび計算生物学研究のための透明性のある不確実性の定量化を提供します。

## 概要

mLLMCelltypeは、単一細胞トランスクリプトーム解析のためのオープンソースツールで、複数の大規模言語モデルを使用して適切に選択された違いに発現するマーカー選択から細胞タイプを同定します。このソフトウェアは、複数のモデルが同じデータを分析し、その予測を組み合わせるコンセンサスアプローチを実装しており、これによりエラーの削減と不確実性指標の提供が可能になります。mLLMCelltypeはScanpyやSeuratなどの人気のある単一細胞解析プラットフォームと統合され、研究者が既存のバイオインフォマティクスワークフローに組み込むことができます。一部の従来の方法とは異なり、アノテーションに参照データセットを必要としません。

## 目次
- [ニュース](#ニュース)
- [主な特徴](#主な特徴)
- [最新の更新](#最新の更新)
- [ディレクトリ構造](#ディレクトリ構造)
- [インストール](#インストール)
- [使用例](#使用例)
- [可視化例](#可視化例)
- [引用](#引用)
- [コントリビューション](#コントリビューション)

## ニュース

🎉 **2025年4月**: 私たちは、プレプリント公開からわずか2週間でmLLMCelltypeがGitHubで200スターを超えたことを発表できることを大変喫しく思います！また、様々なメディアやコンテンツクリエイターからの大きな反響もいただいています。スター、シェア、貢献を通じてこのプロジェクトを支援してくださった全ての方々に心から感謝いたします。皆様の熱意が、mLLMCelltypeの継続的な開発と改善の原動力となっています。

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
# PyPIからインストール
pip install mllmcelltype

# またはGitHubからインストール（subdirectoryパラメータに注意）
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

## クイックスタート

### Rでの使用例

> **注意**: より詳細なRチュートリアルとドキュメントについては、[mLLMCelltype ドキュメントサイト](https://cafferyang.com/mLLMCelltype/)をご覧ください。

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

#### CSV入力例

Seuratを使用せずに直接CSVファイルからmLLMCelltypeを使用することもできます。これはマーカー遣伝子がCSV形式で既に利用可能な場合に便利です：

```r
# mLLMCelltypeの最新バージョンをインストール
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# 必要なパッケージを読み込む
library(mLLMCelltype)

# キャッシュとログディレクトリを作成
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# CSVファイルの内容を読み込む
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# ヘッダー行をスキップ
data_lines <- file_content[-1]

# データをリスト形式に変換、数値インデックスをキーとして使用
marker_genes_list <- list()
cluster_names <- c()

# まず、すべてのクラスター名を収集
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# 次に、数値インデックスを持つmarker_genes_listを作成
for(i in seq_along(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # 最初の部分はクラスター名
  cluster_name <- parts[1]

  # インデックスをキーとして使用（0ベースインデックス、Seuratと互換性あり）
  cluster_id <- as.character(i - 1)

  # 残りは遣伝子
  genes <- parts[-1]

  # NAと空の文字列をフィルタリング
  genes <- genes[!is.na(genes) & genes != ""]

  # marker_genes_listに追加
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# APIキーを設定
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# コンセンサスアノテーションを実行
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # 例："human heart"
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

# 結果を保存
saveRDS(consensus_results, "your_results.rds")

# 結果の概要を出力
cat("\n結果の概要:\n")
cat("利用可能なフィールド:", paste(names(consensus_results), collapse=", "), "\n\n")

# 最終アノテーションを出力
cat("最終細胞タイプアノテーション:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**CSV形式に関する注意点**:
- CSVファイルの最初の列には、任意の値（クラスター名、0,1,2,3や1,2,3,4などの数値シーケンスなど）を使用でき、これらはインデックスとして使用されます
- 最初の列の値は参照用としてのみ使用され、LLMモデルには渡されません
- 次の列には、各クラスターのマーカー遣伝子を含める必要があります
- パッケージには猫の心臓組織用のサンプルCSVファイルが含まれています：`inst/extdata/Cat_Heart_markers.csv`

CSV構造の例：
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

Rスクリプトでサンプルデータにアクセスするには、次のコードを使用します：
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### Pythonでの使用例

```python
# 単一細胞 RNA-seq データにおける mLLMCelltype を用いた細胞タイプアノテーションの例
import scanpy as sc
import mllmcelltype as mct

# 単一細胞 RNA-seq データを AnnData 形式で読み込み
adata = sc.read_h5ad("your_data.h5ad")  # あなたの scRNA-seq データに置き換えてください

# 細胞集団の特定のためのネットワーク構築とLeidenクラスタリングの実行
sc.pp.neighbors(adata)  # 類似細胞を接続するKNNグラフの構築
sc.tl.leiden(adata)     # コミュニティ検出アルゴリズムを使用した細胞集団の特定

# 各クラスターのマーカー遺伝子を特定するための差別発現分析の実行
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")  # Wilcoxon順位和検定を用いたマーカー検出

# Scanpyのマーカー遺伝子結果をmLLMCelltypeで使用可能な形式に変換
markers_dict = mct.utils.convert_scanpy_markers(adata)  # マーカー遺伝子辞書形式に変換

# 複数の大規模言語モデルを使用した細胞タイプアノテーションの実行
consensus_results = mct.annotate.interactive_consensus_annotation(
    input=markers_dict,                                             # マーカー遺伝子辞書
    tissue_name="human PBMC",                                      # 組織情報の指定（アノテーション精度向上）
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro"],  # 複数の大規模言語モデルの使用
    api_keys={                                                     # 各LLMプロバイダーのAPIキー設定
        "openai": "your_openai_api_key",                             # OpenAI GPT-4o APIキー
        "anthropic": "your_anthropic_api_key",                       # Anthropic Claude APIキー
        "gemini": "your_gemini_api_key"                              # Google Gemini APIキー
    },
    top_gene_count=10                                              # 各クラスターごとに使用する上位マーカー遺伝子数
)

# アノテーション結果をAnnDataオブジェクトに追加し、後続の分析と可視化の準備
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
# サポートされているモデルには以下が含まれます：
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 単一APIで複数のモデルにアクセス。形式: 'provider/model-name'
#   - OpenAIモデル: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropicモデル: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Metaモデル: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Googleモデル: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistralモデル: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - その他のモデル: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

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

## コントリビューション

コミュニティからの貢献を歓迎し、感謝しています！mLLMCelltypeに貢献する方法は多数あります：

### 問題の報告

バグ、機能リクエスト、またはmLLMCelltypeの使用に関する質問がある場合は、GitHubリポジトリで[イシューを開いて](https://github.com/cafferychen777/mLLMCelltype/issues)ください。バグを報告する際には、以下の情報を含めてください：

- 問題の明確な説明
- 問題を再現する手順
- 期待される動作と実際の動作
- オペレーティングシステムとパッケージバージョン情報
- 関連するコードスニペットやエラーメッセージ

### プルリクエスト

プルリクエストを通じてコードの改善や新機能の貢献を奥励します：

1. リポジトリをフォークする
2. 機能用の新しいブランチを作成する (`git checkout -b feature/amazing-feature`)
3. 変更をコミットする (`git commit -m '素晴らしい機能を追加'`)
4. ブランチにプッシュする (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

### 貢献の分野

特に価値のある貢献の分野は以下の通りです：

- 新しいLLMモデルのサポートの追加
- ドキュメントと例の改善
- パフォーマンスの最適化
- 新しい視覚化オプションの追加
- 特定の細胞タイプや組織の機能拡張
- ドキュメントの異なる言語への翻訳

### コードスタイル

リポジトリ内の既存のコードスタイルに従ってください。Rコードについては、一般的に[tidyverseスタイルガイド](https://style.tidyverse.org/)に従います。Pythonコードについては、[PEP 8](https://www.python.org/dev/peps/pep-0008/)に従います。

mLLMCelltypeの改善にご協力いただき、ありがとうございます！