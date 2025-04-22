<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltype ist ein iteratives Multi-LLM-Konsensus-Framework für die Zelltyp-Annotation in Einzelzell-RNA-Sequenzierungsdaten. Durch die Nutzung der komplementären Stärken mehrerer großer Sprachmodelle (GPT, Claude, Gemini, Grok, DeepSeek, Qwen usw.) verbessert dieses Framework die Annotationsgenauigkeit erheblich und bietet gleichzeitig eine transparente Quantifizierung der Unsicherheit.

## Hauptmerkmale

- **Multi-LLM-Konsensus-Architektur**: Nutzt die kollektive Intelligenz verschiedener LLMs, um Einschränkungen und Verzerrungen einzelner Modelle zu überwinden
- **Strukturierter Beratungsprozess**: Ermöglicht LLMs, Argumentationen zu teilen, Beweise zu bewerten und Annotationen durch mehrere Runden kollaborativer Diskussion zu verfeinern
- **Transparente Unsicherheitsquantifizierung**: Bietet quantitative Metriken (Konsensusanteil und Shannon-Entropie), um mehrdeutige Zellpopulationen zu identifizieren, die eine Expertenüberprüfung erfordern
- **Halluzinationsreduktion**: Modellübergreifende Beratung unterdrückt aktiv ungenaue oder unbegründete Vorhersagen durch kritische Bewertung
- **Robust gegenüber Eingaberauschen**: Behält hohe Genauigkeit auch bei unvollkommenen Markergen-Listen durch kollektive Fehlerkorrektur
- **Unterstützung für hierarchische Annotation**: Optionale Erweiterung für Multiresolutions-Analyse mit Eltern-Kind-Konsistenz
- **Kein Referenzdatensatz erforderlich**: Führt genaue Annotation ohne Vortraining oder Referenzdaten durch
- **Vollständige Argumentationsketten**: Dokumentiert den gesamten Beratungsprozess für transparente Entscheidungsfindung
- **Nahtlose Integration**: Arbeitet direkt mit Standard-Scanpy/Seurat-Workflows und Markergen-Outputs
- **Modulares Design**: Einfache Integration neuer LLMs, sobald diese verfügbar werden

### Unterstützte Modelle

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([API-Schlüssel](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([API-Schlüssel](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([API-Schlüssel](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API-Schlüssel](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API-Schlüssel](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API-Schlüssel](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API-Schlüssel](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API-Schlüssel](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([API-Schlüssel](https://accounts.x.ai/))
- **OpenRouter**: Zugriff auf mehrere Modelle über eine einzige API ([API-Schlüssel](https://openrouter.ai/keys))
  - Unterstützt Modelle von OpenAI, Anthropic, Meta, Google, Mistral und mehr
  - Format: 'provider/model-name' (z.B. 'openai/gpt-4o', 'anthropic/claude-3-opus')

## Verzeichnisstruktur

- `R/`: R-Sprach-Interface und Implementierung
- `python/`: Python-Interface und Implementierung

## Installation

### R-Version

```r
# Installation von GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python-Version

```bash
pip install mllmcelltype
```

## Schnellstart

### Verwendungsbeispiel in R

```r
library(mLLMCelltype)
library(Seurat)

# Markergenliste vorbereiten
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# Zelltyp-Annotation durchführen
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

# Ergebnisse überprüfen
print(consensus_results$final_annotations)

# Annotationen zum Seurat-Objekt hinzufügen
current_clusters <- as.character(Idents(seurat_obj))
cell_types <- as.character(current_clusters)
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
seurat_obj$cell_type <- cell_types
```

### Verwendungsbeispiel in Python

```python
import scanpy as sc
import mllmcelltype as mct

# AnnData-Objekt laden
adata = sc.read_h5ad("your_data.h5ad")

# Clustering durchführen (überspringen, wenn bereits durchgeführt)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Markergene identifizieren
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# Markergene in das mLLMCelltype-Eingabeformat konvertieren
markers_dict = mct.utils.convert_scanpy_markers(adata)

# Zelltyp-Annotation durchführen
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

# Annotationen zum AnnData-Objekt hinzufügen
adata.obs["cell_type"] = adata.obs["leiden"].map(
    lambda x: consensus_results["final_annotations"].get(x, "Unknown")
)
```

## Unsicherheitsvisualisierung

mLLMCelltype bietet zwei Metriken zur Quantifizierung der Annotationsunsicherheit:

1. **Konsensusanteil**: Der Anteil der Modelle, die einer bestimmten Vorhersage zustimmen
2. **Shannon-Entropie**: Misst die Unsicherheit in der Vorhersageverteilung

Um diese Metriken zu visualisieren:

```r
library(Seurat)
library(ggplot2)
library(cowplot)
library(SCpubr)

# Unsicherheitsmetriken berechnen
uncertainty_metrics <- calculate_uncertainty_metrics(consensus_results)

# Unsicherheitsmetriken zum Seurat-Objekt hinzufügen
current_clusters <- as.character(Idents(pbmc))
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# Zelltyp-Annotationen visualisieren
p1 <- SCpubr::do_DimPlot(sample = pbmc, 
                       group.by = "cell_type",
                       label = TRUE, 
                       repel = TRUE,
                       pt.size = 0.1) +
      ggtitle("Cell Type Annotations") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Konsensusanteil visualisieren
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "consensus_proportion",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("yellow", "green", "blue"),
                         limits = c(min(pbmc$consensus_proportion),  # Minimalwert setzen
                                   max(pbmc$consensus_proportion)),  # Maximalwert setzen
                         na.value = "lightgrey") +  # Farbe für fehlende Werte
      ggtitle("Consensus Proportion") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Entropie visualisieren
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "entropy",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("darkred", "red", "orange"),
                         limits = c(min(pbmc$entropy),  # Minimalwert setzen
                                   max(pbmc$entropy)),  # Maximalwert setzen
                         na.value = "lightgrey") +  # Farbe für fehlende Werte
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Plots mit gleichen Breiten kombinieren
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

### Verwendung eines einzelnen LLM-Modells

Wenn Sie nur einen API-Schlüssel haben oder ein bestimmtes LLM-Modell bevorzugen, können Sie die Funktion `annotate_cell_types()` verwenden:

```r
# Vorverarbeitetes Seurat-Objekt laden
pbmc <- readRDS("your_seurat_object.rds")

# Markergene für jeden Cluster finden
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Ein Modell von einem beliebigen unterstützten Anbieter auswählen
# Beispiele für unterstützte Modelle:
# - Anthropic: "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-latest"
# - OpenAI: "gpt-4o"
# - Google: "gemini-1.5-pro", "gemini-2.0-flash"
# - DeepSeek: "deepseek-chat", "deepseek-reasoner"
# - Alibaba: "qwen-max-2025-01-25"
# - X.AI: "grok-3", "grok-3-mini"
# - Zhipu: "glm-4-plus", "glm-3-turbo"
# - MiniMax: "minimax-text-01"
# - Stepfun: "step-2-16k", "step-2-mini"

# Zelltyp-Annotation mit einem einzelnen LLM-Modell durchführen
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # Gewebekontext angeben
  model = "claude-3-7-sonnet-20250219",  # ein einzelnes Modell spezifizieren
  api_key = "your-anthropic-key",  # API-Schlüssel direkt angeben
  top_gene_count = 10
)

# Ergebnisse ausgeben
print(single_model_results)

# Annotationen zum Seurat-Objekt hinzufügen
# single_model_results ist ein Zeichenvektor mit einer Annotation pro Cluster
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# Ergebnisse visualisieren
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("Mit einem einzelnen LLM-Modell annotierte Zelltypen")
```

#### Vergleich verschiedener Modelle

Sie können auch Annotationen von verschiedenen Modellen vergleichen, indem Sie `annotate_cell_types()` mehrmals mit unterschiedlichen Modellen ausführen:

```r
# Verschiedene Modelle für die Annotation verwenden
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# Eine Spalte für jedes Modell erstellen
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )
  
  # Spaltennamen basierend auf dem Modell erstellen
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))
  
  # Annotationen zum Seurat-Objekt hinzufügen
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# Ergebnisse verschiedener Modelle visualisieren
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# Grafiken kombinieren
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## Visualisierungsbeispiel

Nachfolgend ein Beispiel für eine publikationsreife Visualisierung, die mit mLLMCelltype und SCpubr erstellt wurde und Zelltyp-Annotationen zusammen mit Unsicherheitsmetriken (Konsensusanteil und Shannon-Entropie) zeigt:

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Abbildung: Das linke Panel zeigt Zelltyp-Annotationen auf der UMAP-Projektion. Das mittlere Panel zeigt den Konsensusanteil mit einem Gelb-Grün-Blau-Farbverlauf (tieferes Blau zeigt stärkere Übereinstimmung zwischen den LLMs an). Das rechte Panel zeigt die Shannon-Entropie mit einem Orange-Rot-Farbverlauf (tieferes Rot zeigt geringere Unsicherheit, helleres Orange zeigt höhere Unsicherheit an).*

## Zitierung

Wenn Sie mLLMCelltype in Ihrer Forschung verwenden, zitieren Sie bitte:

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

Sie können auch im Klartext-Format zitieren:

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852