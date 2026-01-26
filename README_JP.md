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
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://www.r-pkg.org/badges/version/mLLMCelltype" alt="CRANバージョン"></a>
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://cranlogs.r-pkg.org/badges/grand-total/mLLMCelltype" alt="CRANダウンロード数"></a>
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
  <a href="https://www.mllmcelltype.com/"><img src="https://img.shields.io/badge/Try%20Online-mLLMCelltype-brightgreen" alt="Try Online"></a>
  <a href="https://colab.research.google.com/github/cafferychen777/mLLMCelltype/blob/main/notebooks/mLLMCelltype_Tutorial.ipynb"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Open In Colab"></a>
</div>

# mLLMCelltype: 単一細胞 RNA シーケンシングのためのマルチ大規模言語モデルコンセンサスフレームワーク

mLLMCelltypeは、単一細胞RNAシーケンシング(scRNA-seq)データにおける正確で信頼性の高い細胞タイプアノテーションのための高度な反復型マルチLLMコンセンサスフレームワークです。複数の大規模言語モデル（OpenAI GPT-5/4.1、Anthropic Claude-4/3.7/3.5、Google Gemini-2.0、X.AI Grok-3、DeepSeek-V3、Alibaba Qwen2.5、Zhipu GLM-4、MiniMax、Stepfun、OpenRouterなど）の集合知能を活用することで、このフレームワークはアノテーションの精度を大幅に向上させながら、バイオインフォマティクスおよび計算生物学研究のための透明性のある不確実性の定量化を提供します。

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

🚀 **Webアプリケーションリリース（2025-06-18）**

mLLMCelltype Webアプリケーションのリリースを発表いたします！これで、インストール不要でWebブラウザから直接mLLMCelltypeの強力な細胞タイプアノテーション機能にアクセスできるようになりました。

**✨ 主な機能：**
- **使いやすいインターフェース**：scRNA-seqデータをアップロードして数分でアノテーションを取得
- **マルチLLMコンセンサス**：GPT-4、Claude、Geminiなど様々なAIモデルから選択
- **リアルタイム処理**：ライブ更新でアノテーションの進行状況を監視
- **複数のエクスポート形式**：CSV、TSV、Excel、JSON形式で結果をダウンロード
- **セットアップ不要**：パッケージのインストールなしで即座にアノテーション開始

**🌐 Webアプリにアクセス**：[https://mllmcelltype.com](https://mllmcelltype.com)

**⚠️ ベータテスト段階**：Webアプリケーションは現在ベータテスト中です。プラットフォームの改善にご協力いただくため、フィードバックとご提案をお待ちしております。問題の報告や体験の共有は、[GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues)または[Discordコミュニティ](https://discord.gg/pb2aZdG4)を通じてお願いします。

**📢 重要：Geminiモデル移行（2025-06-02）**

Googleは複数のGemini 1.5モデルを廃止し、2025年9月24日にさらなるモデルを廃止予定です：
- **既に廃止済み**：Gemini 1.5 Pro 001、Gemini 1.5 Flash 001
- **2025年9月24日廃止予定**：Gemini 1.5 Pro 002、Gemini 1.5 Flash 002、Gemini 1.5 Flash-8B -001

**推奨移行**：より良いパフォーマンスと強化された推論能力のために`gemini-3-pro`または`gemini-3-flash`をご使用ください。エイリアス`gemini-1.5-pro`と`gemini-1.5-flash`は-002バージョンを指すため、2025年9月24日まで動作を継続します。

**📢 重要：Claudeモデルの廃止（2025年7月21日）**

Anthropicは2025年7月21日に複数のClaudeモデルを廃止します：
- **廃止されるモデル**: Claude 2、Claude 2.1、Claude 3 Sonnet（バージョンなし）、Claude 3 Opus（バージョンなし）

**推奨される移行**:
- Claude 2/2.1 → `claude-sonnet-4-5-20250929`または`claude-3-5-sonnet-20241022`
- Claude 3 Sonnet → `claude-sonnet-4-5-20250929`または`claude-3-7-sonnet-20250219`
- Claude 3 Opus → `claude-sonnet-4-5-20250929`または`claude-3-opus-20240229`

サービスの中断を避けるため、2025年7月21日までにモデルを更新してください。

🎉 **2025年4月**: 私たちは、プレプリント公開からわずか2週間でmLLMCelltypeがGitHubで200スターを超えたことを発表できることを大変嬉しく思います！また、様々なメディアやコンテンツクリエイターからの大きな反響もいただいています。スター、シェア、貢献を通じてこのプロジェクトを支援してくださった全ての方々に心から感謝いたします。皆様の熱意が、mLLMCelltypeの継続的な開発と改善の原動力となっています。

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

## 最新の更新

### v1.2.3 (2025-05-10)

#### バグ修正
- API応答がNULLまたは無効な場合のコンセンサスチェックでのエラーハンドリングを修正
- OpenRouter APIエラー応答のエラーログを改善
- check_consensus関数に堅牢なNULLと型チェックを追加

#### 改善
- OpenRouter APIエラーの拡張エラー診断
- APIエラーメッセージと応答構造の詳細ログを追加
- 予期しないAPI応答形式処理時の堅牢性を向上

### v1.2.2 (2025-05-09)

#### バグ修正
- API応答処理時に発生した「非文字引数」エラーを修正
- 全モデルプロバイダーでAPI応答の堅牢な型チェックを追加
- 予期しないAPI応答形式のエラーハンドリングを改善

#### 改善
- API応答問題の詳細エラーログを追加
- 全API処理関数で一貫したエラーハンドリングパターンを実装
- 処理前の適切な構造確保のための応答検証を強化

### v1.2.1 (2025-05-01)

#### 改善
- OpenRouter APIサポートを追加
- OpenRouter経由の無料モデルサポートを追加
- OpenRouterモデル使用例でドキュメントを更新

### v1.2.0 (2025-04-30)

#### 機能
- 細胞タイプアノテーション結果の可視化関数を追加
- 不確実性メトリクス可視化サポートを追加
- 改善されたコンセンサス構築アルゴリズムを実装

### v1.1.5 (2025-04-27)

#### バグ修正
- 特定のCSV入力ファイル処理時にエラーを引き起こすクラスターインデックス検証の問題を修正
- より明確なエラーメッセージで負のインデックスのエラーハンドリングを改善

#### 改善
- CSVベースアノテーションワークフローのサンプルスクリプトを追加（cat_heart_annotation.R）
- より詳細な診断で入力検証を強化
- CSV入力形式要件を明確化するためドキュメントを更新

完全な変更履歴については[NEWS.md](R/NEWS.md)をご覧ください。

### サポートされているモデル

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-5 ([APIキー](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([APIキー](https://console.anthropic.com/))
- **Google**: Gemini-2.5-Pro/Gemini-2.5-Flash ([APIキー](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([APIキー](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([APIキー](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([APIキー](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([APIキー](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([APIキー](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([APIキー](https://accounts.x.ai/))
- **OpenRouter**: 単一APIで複数のモデルにアクセス ([APIキー](https://openrouter.ai/keys))
  - OpenAI、Anthropic、Meta、Google、Mistralなどのモデルをサポート
  - 形式: 'provider/model-name'（例: 'openai/gpt-5'、'anthropic/claude-opus-4.1'）
  - `:free`接尾辞付きの無料モデルが利用可能（例: 'deepseek/deepseek-r1:free'、'deepseek/deepseek-chat:free'）

## ディレクトリ構造

- `R/`: R言語インターフェースと実装
- `python/`: Pythonインターフェースと実装

## インストール

### Rバージョン

```r
# CRANからインストール（推奨）
install.packages("mLLMCelltype")

# または開発版をGitHubからインストール
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Pythonバージョン

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZgmtlaORogSy0-QsaF0CHwFWOyOD26d2?usp=sharing)

**クイックスタート**: インストール不要でGoogle ColabでmLLMCelltypeをすぐにお試しいただけます！上記のバッジをクリックして、例題とステップバイステップガイド付きのインタラクティブなノートブックを開いてください。

```bash
# PyPIからインストール
pip install mllmcelltype

# またはGitHubからインストール（subdirectoryパラメータに注意）
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### 依存関係に関する重要な注意事項

mLLMCelltypeはモジュラー設計を採用しており、異なるLLMプロバイダーライブラリはオプションの依存関係となっています。使用予定のモデルに応じて、対応するパッケージをインストールする必要があります：

```bash
# OpenAIモデル（GPT-5など）を使用する場合
pip install "mllmcelltype[openai]"

# Anthropicモデル（Claude）を使用する場合
pip install "mllmcelltype[anthropic]"

# Googleモデル（Gemini）を使用する場合
pip install "mllmcelltype[gemini]"

# すべてのオプション依存関係を一度にインストール
pip install "mllmcelltype[all]"
```

`ImportError: cannot import name 'genai' from 'google'`のようなエラーが発生した場合、対応するプロバイダーパッケージをインストールする必要があります。例えば：

```bash
# Google Geminiモデルの場合
pip install google-genai
```

## 使用例

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
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"),
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
    models = c("gemini-3-flash",
              "gemini-3-pro",
              "qwen-max-2025-01-25",
              "grok-3-latest",
              "anthropic/claude-sonnet-4",
              "openai/gpt-5"),
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

### Python

```python
# Scanpyを使ったscRNA-seqデータでのmLLMCelltype細胞タイプアノテーションの例
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# 注意：mllmcelltypeインポート時にログが自動設定されます
# 必要に応じてloggingモジュールを使用してログをカスタマイズできます

# あなたのscRNA-seqデータセットをAnnData形式で読み込み
adata = sc.read_h5ad('your_data.h5ad')  # あなたのscRNA-seqデータセットのパスに置き換えてください

# 細胞集団の識別のためのLeidenクラスタリングを実行（まだ行っていない場合）
if 'leiden' not in adata.obs.columns:
    print("細胞集団識別のためのleidenクラスタリングを計算中...")
    # 単一細胞データの前処理：遺伝子発現分析のためのカウント正規化とlog変換
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # 細胞あたり10,000カウントに正規化
        sc.pp.log1p(adata)  # 正規化カウントをlog変換

    # 次元削減：scRNA-seqデータのPCA計算
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # 情報量の多い遺伝子を選択
        sc.pp.pca(adata, use_highly_variable=True)  # 主成分を計算

    # 細胞クラスタリング：近傍グラフを計算してLeidenコミュニティ検出を実行
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # クラスタリング用のKNNグラフを構築
    sc.tl.leiden(adata, resolution=0.8)  # Leidenアルゴリズムを使用して細胞集団を識別
    print(f"Leidenクラスタリング完了、{len(adata.obs['leiden'].cat.categories)}の異なる細胞集団を識別")

# 差分発現解析を使用して各細胞クラスターのマーカー遺伝子を識別
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # マーカー検出のためのWilcoxon順位和検定

# 細胞タイプアノテーションで使用する各細胞クラスターの主要マーカー遺伝子を抽出
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # 各クラスターのマーカーとして上位10の差分発現遺伝子を選択
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# 重要：mLLMCelltypeには遺伝子シンボル（例：KCNJ8、PDGFRA）が必要で、Ensembl ID（例：ENSG00000176771）ではありません
# AnnDataオブジェクトがEnsembl IDを使用している場合、正確なアノテーションのために遺伝子シンボルに変換してください：
# 変換例コード：
# if 'Gene' in adata.var.columns:  # メタデータで遺伝子シンボルが利用可能かチェック
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# 重要：mLLMCelltypeには数値クラスターIDが必要です
# 'cluster'列には数値または数値に変換可能な値が含まれている必要があります。
# 非数値クラスターID（例："cluster_1"、"T_cells"、"7_0"）はエラーまたは予期しない動作を引き起こす可能性があります。
# データに非数値クラスターIDが含まれている場合、元のIDと数値IDの間のマッピングを作成してください：
# 標準化例コード：
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# コンセンサスアノテーションで使用される大規模言語モデルのAPIキーを設定
# マルチLLMコンセンサスアノテーションには少なくとも1つのAPIキーが必要
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # GPT-5/4.1モデル用（推奨）
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Claude-3.7/3.5モデル用
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # Google Gemini-2.5モデル用
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # Alibaba Qwen2.5モデル用
# コンセンサスの多様性を向上させるための追加のオプションLLMプロバイダー：
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # DeepSeek-V3モデル用
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # Zhipu GLM-4モデル用
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # Stepfunモデル用
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # MiniMaxモデル用
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # OpenRouter経由で複数モデルにアクセス

# 反復的な議論を伴うマルチLLMコンセンサス細胞タイプアノテーションを実行
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # 各クラスターのマーカー遺伝子辞書
    species="human",            # 適切な細胞タイプアノテーションのために生物種を指定
    tissue="blood",            # より正確なアノテーションのために組織コンテキストを指定
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"],  # コンセンサス用の複数LLM
    consensus_threshold=1,     # コンセンサス合意に必要な最小比率
    max_discussion_rounds=3    # 改善のためのモデル間議論ラウンド数
)

# または、単一APIで複数モデルにアクセスするためにOpenRouterを使用
# これは:free接尾辞付きの無料モデルにアクセスする場合に特に有用
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# OpenRouter無料モデルを使用した例（クレジット不要）
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Meta Llama 4 Maverick（無料）
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"},  # NVIDIA Nemotron Ultra 253B（無料）
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},   # DeepSeek Chat v3（無料）
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"}               # Microsoft MAI-DS-R1（無料）
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# マルチLLM議論から最終コンセンサス細胞タイプアノテーションを取得
final_annotations = consensus_results["consensus"]

# コンセンサス細胞タイプアノテーションを元のAnnDataオブジェクトに統合
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# アノテーション信頼度を評価するための不確実性定量化メトリクスを追加
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])  # 合意レベル
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])  # アノテーション不確実性

# 可視化の準備：UMAPエンベディングが利用可能でない場合は計算
# UMAPは可視化のための細胞集団の2D表現を提供
if 'X_umap' not in adata.obsm:
    print("UMAP座標を計算中...")
    # 最初に近傍が計算されていることを確認
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("UMAP座標計算完了")

# 拡張された美的観点で結果を可視化
# 基本的な可視化
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='right', frameon=True, title='mLLMCelltypeコンセンサスアノテーション')

# よりカスタマイズされた可視化
import matplotlib.pyplot as plt

# 図のサイズとスタイルを設定
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# より出版準備の整ったUMAPを作成
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data',
         frameon=True, title='mLLMCelltypeコンセンサスアノテーション',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# 不確実性メトリクスを可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sc.pl.umap(adata, color='consensus_proportion', ax=ax1, title='コンセンサス比率',
         cmap='viridis', vmin=0, vmax=1, size=30)
sc.pl.umap(adata, color='entropy', ax=ax2, title='アノテーション不確実性（シャノンエントロピー）',
         cmap='magma', vmin=0, size=30)
plt.tight_layout()
```

### 単一のOpenRouter無料モデルの使用

1つのモデルでよりシンプルなアプローチを好むユーザーには、OpenRouter経由のMicrosoft MAI-DS-R1無料モデルが優れた結果を提供します：

```python
import os
from mllmcelltype import annotate_clusters

# 注意：ログは自動的に設定されます

# OpenRouter APIキーを設定
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 各クラスターのマーカー遺伝子を定義
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T細胞
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B細胞
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # 単球
}

# Microsoft MAI-DS-R1無料モデルを使用してアノテーション
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # 無料モデル
)

# アノテーションを表示
for cluster, annotation in annotations.items():
    print(f"クラスター {cluster}: {annotation}")
```

このアプローチは高速で正確、APIクレジット不要で、迅速な分析やAPIアクセスが限られている場合に理想的です。

#### AnnDataオブジェクトからのマーカー遺伝子抽出

ScanpyでAnnDataオブジェクトを使用している場合、`rank_genes_groups`結果から直接マーカー遺伝子を簡単に抽出できます：

```python
import os
import scanpy as sc
from mllmcelltype import annotate_clusters

# 注意：ログは自動的に設定されます

# OpenRouter APIキーを設定
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# データを読み込んで前処理
adata = sc.read_h5ad('your_data.h5ad')

# 前処理とクラスタリングがまだ行われていない場合は実行
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata)

# 各クラスターのマーカー遺伝子を見つける
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# 各クラスターの主要マーカー遺伝子を抽出
marker_genes = {
    cluster: adata.uns['rank_genes_groups']['names'][cluster][:10].tolist()
    for cluster in adata.obs['leiden'].cat.categories
}

# Microsoft MAI-DS-R1無料モデルを使用してアノテーション
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',  # 組織タイプに応じて調整
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # 無料モデル
)

# AnnDataオブジェクトにアノテーションを追加
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# 結果を可視化
sc.pl.umap(adata, color='cell_type', legend_loc='on data',
           frameon=True, title='MAI-DS-R1でアノテーションされた細胞タイプ')
```

この方法は`rank_genes_groups`結果から各クラスターの主要差分発現遺伝子を自動的に抽出し、mLLMCelltypeをScanpyワークフローに統合しやすくします。

### R

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
# - OpenAI: 'gpt-5', 'gpt-5-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 単一APIで複数のモデルにアクセス。形式: 'provider/model-name'
#   - OpenAIモデル: 'openai/gpt-5', 'openai/gpt-5-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropicモデル: 'anthropic/claude-sonnet-4', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-opus-4.1'
#   - Metaモデル: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Googleモデル: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistralモデル: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - その他のモデル: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# 単一LLMモデルで細胞タイプアノテーションを実行
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 組織のコンテキストを提供
  model = "claude-sonnet-4-5-20250929",  # 単一モデルを指定
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
models <- c("claude-sonnet-4-5-20250929", "gpt-5", "gemini-3-pro", "qwen-max-2025-01-25", "grok-3")
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
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 4")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-5")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# プロットを結合
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

### 高度なコンセンサス設定：コンセンサスチェックモデルの指定

`consensus_check_model`パラメータ（R）/ `consensus_model`パラメータ（Python）では、コンセンサスチェックと議論の司会に使用するLLMモデルを指定できます。このパラメータは、コンセンサスチェックモデルが以下を行うため、コンセンサスアノテーションの精度において**重要**です：

1. 異なる細胞タイプアノテーション間の意味的類似性を評価
2. コンセンサスメトリクス（比率とエントロピー）を計算
3. 議論の多いクラスターに対してモデル間の議論を司会し合成
4. モデルが意見を異にする際の最終決定を下す

**⚠️ 重要：コンセンサスチェックには利用可能な最も有能なモデルを使用することを強く推奨します。これはアノテーション品質に直接影響するためです。**

#### コンセンサスチェック推奨モデル（パフォーマンス順）

1. **Anthropic Claudeモデル**（最高推奨）
   - `claude-sonnet-4-5-20250929` - **全体的に最高**（最新かつ最も知的）
   - `claude-opus-4-1-20250805` - 複雑な推論タスクに優れる

2. **OpenAIモデル**
   - `o1` / `o1-pro` - 高度な推論能力
   - `gpt-5` - 様々な細胞タイプでの強固なパフォーマンス
   - `gpt-4.1` - 最新のGPT-4バリアント

3. **Google Geminiモデル**
   - `gemini-3-pro` - 強化された推論による最上級パフォーマンス
   - `gemini-3-flash` - より高速な処理での良好なパフォーマンス

4. **その他の高パフォーマンスモデル**
   - `deepseek-r1` / `deepseek-reasoner` - 強力な推論能力
   - `qwen-max-2025-01-25` - 科学的コンテキストに優秀
   - `grok-3-latest` - 高度な言語理解

#### Rパッケージの使用

```r
# 例1：コンセンサスチェックに最高のモデルを使用（推奨）
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human brain",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # 最も有能なモデルを使用
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# 例2：Claude Opusが利用できない場合の高パフォーマンスモデル使用
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "mouse liver",
  models = c("gpt-5", "gemini-3-flash", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # 代替高パフォーマンスモデル
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# 例3：複雑なケースでOpenAIの推論モデルを使用
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human immune cells",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash"),
  api_keys = api_keys,
  consensus_check_model = "o1",  # OpenAIの高度推論モデル
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# ⚠️ 非推奨：コンセンサスチェックに能力の低いモデルや無料モデルの使用は避ける
# アノテーション精度を大幅に下げる可能性があります
```

#### Pythonパッケージの使用

```python
# 例1：コンセンサスチェックに最高のモデルを使用（推奨）
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen-max-2025-01-25"],
    consensus_model="claude-sonnet-4-5-20250929",  # 最も有能なモデルを使用
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 例2：高パフォーマンスモデルでの辞書形式使用
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5", "gemini-3-flash", "qwen-max-2025-01-25"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 例3：コンセンサスにGoogleの最新モデルを使用
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "qwen-max-2025-01-25"],
    consensus_model={"provider": "google", "model": "gemini-3-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 例4：デフォルト動作（高パフォーマンスフォールバック付きQwenを使用）
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash"],
    # 指定しない場合、デフォルトでqwen-max-2025-01-25（高パフォーマンスモデル）を使用
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### コンセンサスモデル選択のベストプラクティス

1. **コストより精度を優先**：コンセンサスチェックモデルは最終アノテーションの決定に重要な役割を果たします。ここで能力の低いモデルを使用すると、アノテーションプロセス全体が損なわれる可能性があります。

2. **モデルの可用性**：選択したコンセンサスモデルへのAPIアクセスがあることを確認してください。主要選択が利用できない場合、システムはフォールバックモデルを使用します。

3. **一貫性**：プロジェクト内のすべてのコンセンサスチェックに同じ高パフォーマンスモデルを使用して、一貫した評価基準を確保してください。

4. **複雑な組織**：困難な組織（例：脳、免疫系）には、Claude Opus、O1、Gemini 2.5 Proなどの最も高度なモデルの使用を検討してください。

5. **デフォルト動作**：
   - R：指定されていない場合は`models`リストの最初のモデルを使用
   - Python：デフォルトで`qwen-max-2025-01-25`（高パフォーマンスモデル）を使用、`claude-3-5-sonnet-latest`をフォールバックとして使用

#### コンセンサスチェックでのモデル品質の重要性

コンセンサスチェックモデルは以下を行う必要があります：
- 異なる細胞タイプ名間の意味的類似性を正確に評価（例：「Tリンパ球」と「T細胞」が同じ細胞タイプを指すことを認識）
- 生物学的コンテキストと階層関係の理解
- 複数モデルからの議論を合成して正確な結論に到達
- 下流解析のための信頼性の高い信頼度メトリクスの提供

これらの重要なタスクに能力の低いモデルを使用すると以下につながる可能性があります：
- 議論の多いクラスターの誤識別
- 不正確なコンセンサス計算
- モデル間の意見の相違の不適切な解決
- 最終的に、より正確でない細胞タイプアノテーション

## 可視化の例

以下は、mLLMCelltypeとSCpubrを使用して作成された出版品質の可視化の例で、細胞タイプアノテーションと不確実性メトリクス（コンセンサス比率とシャノンエントロピー）を示しています：

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*図：左パネルはUMAP投影上の細胞タイプアノテーションを示しています。中央パネルは黄色-緑-青のグラデーションを使用してコンセンサス比率を表示しています（濃い青はLLM間の強い合意を示します）。右パネルはオレンジ-赤のグラデーションを使用してシャノンエントロピーを示しています（濃い赤は低い不確実性、明るいオレンジは高い不確実性を示します）。*

### マーカー遺伝子の可視化

mLLMCelltypeには、コンセンサスアノテーションワークフローとシームレスに統合される拡張マーカー遺伝子可視化機能が含まれています：

```r
# 必要なライブラリの読み込み
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# コンセンサスアノテーション実行後
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-sonnet-4.5", "openai/gpt-5"),
  api_keys = list(openrouter = "your_api_key")
)

# Seuratを使用したマーカー遺伝子の可視化作成
# Seuratオブジェクトにコンセンサスアノテーションを追加
pbmc_data$cell_type_consensus <- consensus_results$final_annotations[Idents(pbmc_data)]

# マーカー遺伝子のドットプロット作成
DotPlot(pbmc_data, 
        features = top_markers,
        group.by = "cell_type_consensus") + 
  RotatedAxis()

# マーカー遺伝子のヒートマップ作成
DoHeatmap(pbmc_data, 
          features = top_markers,
          group.by = "cell_type_consensus")
```

**マーカー遺伝子可視化の主な特徴：**

- **バブルプロット**: 各遺伝子を発現する細胞の割合（バブルサイズ）と平均発現レベル（色の強度）の両方を表示
- **ヒートマップ**: 遺伝子の階層クラスタリングを伴うスケール化された発現値を表示
- **出版対応**: viridisカラーパレットを使用したカスタマイズ可能な美観の高品質プロット
- **シームレスな統合**: コンセンサスアノテーション結果とSeuratオブジェクトと直接連携

詳細な手順と高度なカスタマイゼーションオプションについては、[可視化ガイド](R/vignettes/06-visualization-guide.html)を参照してください。

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