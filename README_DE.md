<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - Multi-Sprachmodell-Konsensus-Framework für die Zelltyp-Annotation in Einzelzell-RNA-Sequenzierungsdaten" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=Entdecken%20Sie%20mLLMCelltype%3A%20Ein%20Multi-LLM-Konsensus-Framework%20f%C3%BCr%20die%20Zelltyp-Annotation%20in%20scRNA-seq-Daten%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Chat%20beitreten-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
</div>

# mLLMCelltype: Multi-Sprachmodell-Konsensus-Framework für die Zelltyp-Annotation

mLLMCelltype ist ein fortschrittliches iteratives Multi-LLM-Konsensus-Framework für die präzise und zuverlässige Zelltyp-Annotation in Einzelzell-RNA-Sequenzierungsdaten (scRNA-seq). Durch die Nutzung der kollektiven Intelligenz mehrerer großer Sprachmodelle (OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, und OpenRouter) verbessert dieses Framework die Annotationsgenauigkeit erheblich und bietet gleichzeitig eine transparente Quantifizierung der Unsicherheit für die Forschung in Bioinformatik und Computational Biology.

## Zusammenfassung

mLLMCelltype ist ein Open-Source-Tool für die Einzelzell-Transkriptomanalyse, das mehrere große Sprachmodelle nutzt, um Zelltypen anhand von Genexpressionsdaten zu identifizieren. Die Software implementiert einen Konsensusansatz, bei dem mehrere Modelle dieselben Daten analysieren und ihre Vorhersagen kombiniert werden, was dazu beiträgt, Fehler zu reduzieren und Unsicherheitsmetriken bereitzustellen. mLLMCelltype integriert sich in beliebte Einzelzellanalyse-Plattformen wie Scanpy und Seurat und ermöglicht es Forschern, es in bestehende bioinformatische Arbeitsabläufe einzubinden. Im Gegensatz zu einigen herkömmlichen Methoden erfordert es keine Referenzdatensätze für die Annotation.

## Inhaltsverzeichnis
- [Neuigkeiten](#neuigkeiten)
- [Hauptmerkmale](#hauptmerkmale)
- [Aktuelle Updates](#aktuelle-updates)
- [Verzeichnisstruktur](#verzeichnisstruktur)
- [Installation](#installation)
- [Nutzungsbeispiele](#nutzungsbeispiele)
- [Visualisierungsbeispiel](#visualisierungsbeispiel)
- [Zitierung](#zitierung)
- [Mitwirken](#mitwirken)

## Neuigkeiten

🎉 **April 2025**: Wir freuen uns, bekannt zu geben, dass mLLMCelltype nur zwei Wochen nach der Veröffentlichung unseres Preprints bereits über 200 GitHub-Sterne erreicht hat! Wir haben auch eine beeindruckende Berichterstattung von verschiedenen Medien und Content-Erstellern gesehen. Wir möchten allen, die dieses Projekt durch Sterne, Teilen und Beiträge unterstützt haben, unseren herzlichen Dank aussprechen. Ihre Begeisterung treibt unsere kontinuierliche Entwicklung und Verbesserung von mLLMCelltype voran.

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
# Installation von PyPI
pip install mllmcelltype

# Oder Installation von GitHub (beachten Sie den subdirectory-Parameter)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

## Schnellstart

### Verwendungsbeispiel in R

> **Hinweis**: Für ausführlichere R-Tutorials und Dokumentation besuchen Sie bitte die [mLLMCelltype Dokumentationswebsite](https://cafferyang.com/mLLMCelltype/).

```r
# Erforderliche Bibliotheken für Einzelzellanalyse und Zelltyp-Annotation laden
library(mLLMCelltype)  # Multi-LLM-Konsensus-Framework für Zelltyp-Annotation
library(Seurat)        # Beliebte Plattform für Einzelzellanalyse

# Differentiell exprimierte Markergene für jeden Zellcluster identifizieren
# only.pos = TRUE: Nur hochregulierte Gene in jedem Cluster auswählen
# min.pct = 0.25: Gen muss in mindestens 25% der Zellen im Cluster exprimiert sein
# logfc.threshold = 0.25: Minimale Expressionsdifferenz zwischen Clustern (log-fold change)
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# Multi-LLM-Konsensus-Zelltyp-Annotationsprozess starten
# Diese Funktion koordiniert den Beratungsprozess zwischen mehreren LLMs zur Bestimmung der Zelltypen
consensus_results <- interactive_consensus_annotation(
  input = markers,  # Dataframe mit Markergenen aus Seurat
  tissue_name = "human PBMC",  # Gewebetyp angeben, um biologischen Kontext für die LLMs bereitzustellen
  models = c("gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.5-pro"),  # Neueste LLM-Modelle verwenden
  api_keys = list(
    openai = "your_openai_api_key",     # API-Schlüssel für OpenAI-Dienste
    anthropic = "your_anthropic_api_key", # API-Schlüssel für Anthropic-Dienste
    gemini = "your_gemini_api_key"       # API-Schlüssel für Google Gemini-Dienste
  ),
  top_gene_count = 10  # Anzahl der Top-Markergene, die für jeden Cluster berücksichtigt werden sollen
)

# Finale Konsensus-Zelltypannotationen anzeigen, die durch LLM-Beratung erzielt wurden
print(consensus_results$final_annotations)

# Konsensus-Zelltypannotationen in das Seurat-Objekt integrieren für weitere Visualisierung und Analyse
current_clusters <- as.character(Idents(seurat_obj))  # Aktuelle Cluster-IDs abrufen
cell_types <- as.character(current_clusters)          # Kopie für die Modifikation erstellen
# Cluster-IDs durch entsprechende Zelltyp-Annotationen ersetzen
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
# Zelltypannotationen als neue Spalte in den Metadaten des Seurat-Objekts speichern
seurat_obj$cell_type <- cell_types
```

#### CSV-Eingabebeispiel

Sie können mLLMCelltype auch direkt mit CSV-Dateien ohne Seurat verwenden, was nützlich ist, wenn Ihre Markergene bereits im CSV-Format vorliegen:

```r
# Neueste Version von mLLMCelltype installieren
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# Erforderliche Pakete laden
library(mLLMCelltype)

# Cache- und Log-Verzeichnisse erstellen
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# CSV-Dateiinhalt lesen
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# Kopfzeile überspringen
data_lines <- file_content[-1]

# Daten in Listenformat konvertieren, mit numerischen Indizes als Schlüssel
marker_genes_list <- list()
cluster_names <- c()

# Zuerst alle Clusternamen sammeln
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# Dann marker_genes_list mit numerischen Indizes erstellen
for(i in seq_along(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # Erster Teil ist der Clustername
  cluster_name <- parts[1]

  # Index als Schlüssel verwenden (0-basierter Index, kompatibel mit Seurat)
  cluster_id <- as.character(i - 1)

  # Rest sind Gene
  genes <- parts[-1]

  # NA und leere Strings filtern
  genes <- genes[!is.na(genes) & genes != ""]

  # Zu marker_genes_list hinzufügen
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# API-Schlüssel festlegen
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Konsensus-Annotation ausführen
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # z.B.: "human heart"
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

# Ergebnisse speichern
saveRDS(consensus_results, "your_results.rds")

# Ergebniszusammenfassung ausgeben
cat("\nErgebniszusammenfassung:\n")
cat("Verfügbare Felder:", paste(names(consensus_results), collapse=", "), "\n\n")

# Endgültige Annotationen ausgeben
cat("Endgültige Zelltyp-Annotationen:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**Hinweise zum CSV-Format**:
- Die erste Spalte der CSV-Datei kann beliebige Werte enthalten (wie Clusternamen, Zahlensequenzen wie 0,1,2,3 oder 1,2,3,4 usw.), die als Indizes verwendet werden
- Die Werte in der ersten Spalte dienen nur als Referenz und werden nicht an die LLM-Modelle übergeben
- Die folgenden Spalten sollten Markergene für jeden Cluster enthalten
- Ein Beispiel-CSV für Katzenherz-Gewebe ist im Paket enthalten: `inst/extdata/Cat_Heart_markers.csv`

Beispiel für die CSV-Struktur:
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

Sie können auf die Beispieldaten in Ihrem R-Skript zugreifen mit:
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### Verwendungsbeispiel in Python

```python
import scanpy as sc
import mllmcelltype as mct

# AnnData-Objekt mit Einzelzell-Genexpressionsdaten laden
adata = sc.read_h5ad("your_data.h5ad")

# Nachbarschaftsgraphen erstellen und Clustering durchführen
# Dieser Schritt konstruiert einen Konnektivitätsgraphen zwischen Zellen basierend auf Genexpressionsmustern
sc.pp.neighbors(adata)
# Leiden-Algorithmus anwenden, um distinkte Zellgemeinschaften zu identifizieren
sc.tl.leiden(adata)

# Differentiell exprimierte Markergene für jeden Cluster identifizieren
# Wilcoxon-Ranksum-Test wird verwendet, um signifikante Marker für jeden Zelltyp zu finden
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# Scanpy-Markergene in das Dictionary-Format konvertieren, das von mLLMCelltype benötigt wird
# Diese Funktion extrahiert die signifikantesten Gene für jeden Cluster
markers_dict = mct.utils.convert_scanpy_markers(adata)

# Multi-LLM-Konsensus-Zelltyp-Annotationsprozess starten
# Diese Funktion koordiniert den Beratungsprozess zwischen mehreren LLMs zur Bestimmung der Zelltypen
consensus_results = mct.annotate.interactive_consensus_annotation(
    input=markers_dict,  # Wörterbuch mit Markergenen für jeden Cluster
    tissue_name="human PBMC",  # Gewebetyp angeben, um biologischen Kontext für die LLMs bereitzustellen
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.5-pro"],  # Neueste LLM-Modelle verwenden
    api_keys={
        "openai": "your_openai_api_key",  # API-Schlüssel für OpenAI-Dienste
        "anthropic": "your_anthropic_api_key",  # API-Schlüssel für Anthropic-Dienste
        "gemini": "your_gemini_api_key"  # API-Schlüssel für Google Gemini-Dienste
    },
    top_gene_count=10  # Anzahl der Top-Markergene, die für jeden Cluster berücksichtigt werden sollen
)

# Konsensus-Zelltypannotationen in das AnnData-Objekt integrieren für weitere Visualisierung und Analyse
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
# Unterstützte Modelle umfassen:
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Zugriff auf mehrere Modelle über eine einzige API. Format: 'provider/model-name'
#   - OpenAI Modelle: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic Modelle: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Meta Modelle: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google Modelle: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral Modelle: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Andere Modelle: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

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

## Mitwirken

Wir begrüßen und schätzen Beiträge aus der Community! Es gibt viele Möglichkeiten, wie Sie zu mLLMCelltype beitragen können:

### Probleme melden

Wenn Sie auf Fehler stoßen, Funktionswünsche haben oder Fragen zur Verwendung von mLLMCelltype haben, [eröffnen Sie bitte ein Issue](https://github.com/cafferychen777/mLLMCelltype/issues) in unserem GitHub-Repository. Bei der Meldung von Fehlern geben Sie bitte Folgendes an:

- Eine klare Beschreibung des Problems
- Schritte zur Reproduktion des Problems
- Erwartetes vs. tatsächliches Verhalten
- Informationen zu Ihrem Betriebssystem und Paketversionen
- Relevante Codeausschnitte oder Fehlermeldungen

### Pull Requests

Wir ermutigen Sie, Codeverbesserungen oder neue Funktionen über Pull Requests beizutragen:

1. Forken Sie das Repository
2. Erstellen Sie einen neuen Branch für Ihre Funktion (`git checkout -b feature/amazing-feature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'Eine tolle Funktion hinzufügen'`)
4. Pushen Sie zum Branch (`git push origin feature/amazing-feature`)
5. Eröffnen Sie einen Pull Request

### Bereiche für Beiträge

Hier sind einige Bereiche, in denen Beiträge besonders wertvoll wären:

- Unterstützung für neue LLM-Modelle hinzufügen
- Dokumentation und Beispiele verbessern
- Leistungsoptimierung
- Neue Visualisierungsoptionen hinzufügen
- Funktionalität für spezialisierte Zelltypen oder Gewebe erweitern
- Übersetzungen der Dokumentation in verschiedene Sprachen

### Code-Stil

Bitte folgen Sie dem bestehenden Code-Stil im Repository. Für R-Code folgen wir im Allgemeinen dem [tidyverse-Stilführer](https://style.tidyverse.org/). Für Python-Code folgen wir [PEP 8](https://www.python.org/dev/peps/pep-0008/).

Vielen Dank für Ihre Hilfe bei der Verbesserung von mLLMCelltype!