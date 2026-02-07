<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="GitHub forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Chat%20beitreten-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://www.r-pkg.org/badges/version/mLLMCelltype" alt="CRAN version"></a>
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://cranlogs.r-pkg.org/badges/grand-total/mLLMCelltype" alt="CRAN downloads"></a>
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
  <a href="https://pypi.org/project/mllmcelltype/"><img src="https://img.shields.io/pypi/v/mllmcelltype" alt="PyPI version"></a>
</div>

# mLLMCelltype: Multi-Sprachmodell-Konsensus-Framework für die Zelltyp-Annotation

mLLMCelltype ist ein iteratives Multi-LLM-Konsensus-Framework für die Zelltyp-Annotation in Einzelzell-RNA-Sequenzierungsdaten (scRNA-seq). Durch die Kombination der Vorhersagen mehrerer großer Sprachmodelle (OpenAI GPT-5.2/5, Anthropic Claude-4.6/4.5, Google Gemini-3, X.AI Grok-4, DeepSeek-V3, Alibaba Qwen3, Zhipu GLM-4, MiniMax, Stepfun, und OpenRouter) zielt dieses Framework darauf ab, die Annotationsgenauigkeit zu verbessern und bietet gleichzeitig eine transparente Quantifizierung der Unsicherheit für die Forschung in Bioinformatik und Computational Biology.

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

**Webanwendung-Launch (18.06.2025)**

Wir freuen uns, den Launch der mLLMCelltype-Webanwendung bekannt zu geben! Jetzt können Sie die leistungsstarken Funktionen zur Zelltyp-Annotation von mLLMCelltype direkt über Ihren Webbrowser nutzen, ohne Installation.

**✨ Hauptfunktionen:**
- **Benutzerfreundliche Oberfläche**: Laden Sie Ihre scRNA-seq-Daten hoch und erhalten Sie Annotationen in wenigen Minuten
- **Multi-LLM-Konsensus**: Wählen Sie aus verschiedenen KI-Modellen wie GPT-4, Claude, Gemini und mehr
- **Echtzeitverarbeitung**: Verfolgen Sie den Annotationsfortschritt mit Live-Updates
- **Mehrere Exportformate**: Laden Sie Ergebnisse im CSV-, TSV-, Excel- oder JSON-Format herunter
- **Keine Einrichtung erforderlich**: Beginnen Sie sofort mit der Annotation ohne Paketinstallation

**🌐 Zugriff auf die Web-App**: [https://mllmcelltype.com](https://mllmcelltype.com)

**⚠️ Beta-Testphase**: Die Webanwendung befindet sich derzeit in der Beta-Testphase. Wir freuen uns über Ihr Feedback und Ihre Vorschläge zur Verbesserung der Plattform. Bitte melden Sie Probleme oder teilen Sie Ihre Erfahrungen über unsere [GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues) oder [Discord-Community](https://discord.gg/pb2aZdG4).

**📢 Wichtig: Gemini-Modell-Migration (02.06.2025)**

Google hat mehrere Gemini 1.5-Modelle eingestellt und wird am 24. September 2025 weitere einstellen:
- **Bereits eingestellt**: Gemini 1.5 Pro 001, Gemini 1.5 Flash 001
- **Werden am 24. Sept. 2025 eingestellt**: Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, Gemini 1.5 Flash-8B -001

**Empfohlene Migration**: Verwenden Sie `gemini-3-pro` oder `gemini-3-flash` für bessere Leistung und verbesserte Argumentationsfähigkeiten. Die Aliase `gemini-1.5-pro` und `gemini-1.5-flash` funktionieren bis zum 24. September 2025 weiterhin, da sie auf die -002-Versionen verweisen.

**📢 Wichtig: Claude-Modell-Einstellung (21.07.2025)**

Anthropic wird am 21. Juli 2025 mehrere Claude-Modelle einstellen:
- **Einzustellende Modelle**: Claude 2, Claude 2.1, Claude 3 Sonnet (ohne Version), Claude 3 Opus (ohne Version)

**Empfohlene Migration**:
- Claude 2/2.1 → `claude-sonnet-4-5-20250929` oder `claude-3-5-sonnet-20241022`
- Claude 3 Sonnet → `claude-sonnet-4-5-20250929` oder `claude-3-7-sonnet-20250219`
- Claude 3 Opus → `claude-sonnet-4-5-20250929` oder `claude-3-opus-20240229`

Bitte aktualisieren Sie Ihre Modelle vor dem 21. Juli 2025, um Dienstunterbrechungen zu vermeiden.

## Hauptmerkmale

- **Multi-LLM-Konsensus-Architektur**: Kombiniert die Vorhersagen verschiedener LLMs, um Einschränkungen und Verzerrungen einzelner Modelle zu reduzieren
- **Strukturierter Beratungsprozess**: Ermöglicht LLMs, Argumentationen zu teilen, Beweise zu bewerten und Annotationen durch mehrere Runden kollaborativer Diskussion zu verfeinern
- **Transparente Unsicherheitsquantifizierung**: Bietet quantitative Metriken (Konsensusanteil und Shannon-Entropie), um mehrdeutige Zellpopulationen zu identifizieren, die eine Expertenüberprüfung erfordern
- **Halluzinationsreduktion**: Modellübergreifende Beratung hilft, ungenaue oder unbegründete Vorhersagen durch kritische Bewertung zu identifizieren
- **Robust gegenüber Eingaberauschen**: Behält hohe Genauigkeit auch bei unvollkommenen Markergen-Listen durch kollektive Fehlerkorrektur
- **Unterstützung für hierarchische Annotation**: Optionale Erweiterung für Multiresolutions-Analyse mit Eltern-Kind-Konsistenz
- **Kein Referenzdatensatz erforderlich**: Führt genaue Annotation ohne Vortraining oder Referenzdaten durch
- **Vollständige Argumentationsketten**: Dokumentiert den gesamten Beratungsprozess für transparente Entscheidungsfindung
- **Nahtlose Integration**: Arbeitet direkt mit Standard-Scanpy/Seurat-Workflows und Markergen-Outputs
- **Modulares Design**: Einfache Integration neuer LLMs, sobald diese verfügbar werden

## Aktuelle Updates

### v1.2.3 (10.05.2025)

#### Fehlerbehebungen
- Fehlerbehandlung bei der Konsensprüfung behoben, wenn API-Antworten NULL oder ungültig sind
- Verbesserte Fehlerprotokollierung für OpenRouter API-Fehlerantworten
- Robuste NULL- und Typprüfung in der check_consensus Funktion hinzugefügt

#### Verbesserungen
- Erweiterte Fehlerdiagnose für OpenRouter API-Fehler
- Detaillierte Protokollierung von API-Fehlermeldungen und Antwortstrukturen hinzugefügt
- Verbesserte Robustheit bei der Behandlung unerwarteter API-Antwortformate

### v1.2.2 (09.05.2025)

#### Fehlerbehebungen
- "Nicht-Zeichen-Argument"-Fehler behoben, der bei der Verarbeitung von API-Antworten auftrat
- Robuste Typprüfung für API-Antworten aller Modellanbieter hinzugefügt
- Verbesserte Fehlerbehandlung für unerwartete API-Antwortformate

#### Verbesserungen
- Detaillierte Fehlerprotokollierung für API-Antwortprobleme hinzugefügt
- Konsistente Fehlerbehandlungsmuster in allen API-Verarbeitungsfunktionen implementiert
- Erweiterte Antwortvalidierung, um eine ordnungsgemäße Struktur vor der Verarbeitung sicherzustellen

### v1.2.1 (01.05.2025)

#### Verbesserungen
- Unterstützung für OpenRouter API hinzugefügt
- Unterstützung für kostenlose Modelle über OpenRouter hinzugefügt
- Dokumentation mit Beispielen für die Verwendung von OpenRouter-Modellen aktualisiert

### v1.2.0 (30.04.2025)

#### Funktionen
- Visualisierungsfunktionen für Zelltyp-Annotationsergebnisse hinzugefügt
- Unterstützung für die Visualisierung von Unsicherheitsmetriken hinzugefügt
- Verbesserter Konsensbildungsalgorithmus implementiert

### v1.1.5 (27.04.2025)

#### Fehlerbehebungen
- Problem mit der Cluster-Index-Validierung behoben, das bei der Verarbeitung bestimmter CSV-Eingabedateien zu Fehlern führte
- Verbesserte Fehlerbehandlung für negative Indizes mit klareren Fehlermeldungen

#### Verbesserungen
- Beispielskript für CSV-basierten Annotations-Workflow hinzugefügt (cat_heart_annotation.R)
- Erweiterte Eingabevalidierung mit detaillierteren Diagnosen
- Dokumentation aktualisiert, um CSV-Eingabeformatanforderungen zu klären

Siehe [NEWS.md](R/NEWS.md) für ein vollständiges Änderungsprotokoll.

## Verzeichnisstruktur

- `R/`: R-Sprachschnittstelle und Implementierung
- `python/`: Python-Schnittstelle und Implementierung

## Installation

### R-Version

```r
# Von CRAN installieren (empfohlen)
install.packages("mLLMCelltype")

# Oder Entwicklungsversion von GitHub installieren
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python-Version

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZgmtlaORogSy0-QsaF0CHwFWOyOD26d2?usp=sharing)

**Schnellstart**: Probieren Sie mLLMCelltype sofort in Google Colab ohne Installation aus! Klicken Sie auf das obige Badge, um unser interaktives Notebook mit Beispielen und Schritt-für-Schritt-Anleitung zu öffnen.

```bash
# Von PyPI installieren
pip install mllmcelltype

# Oder von GitHub installieren (beachten Sie den Unterverzeichnis-Parameter)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### Wichtiger Hinweis zu Abhängigkeiten

mLLMCelltype verwendet ein modulares Design, bei dem verschiedene LLM-Anbieter-Bibliotheken optionale Abhängigkeiten sind. Je nachdem, welche Modelle Sie verwenden möchten, müssen Sie die entsprechenden Pakete installieren:

```bash
# Für die Verwendung von OpenAI-Modellen (GPT-5, etc.)
pip install "mllmcelltype[openai]"

# Für die Verwendung von Anthropic-Modellen (Claude)
pip install "mllmcelltype[anthropic]"

# Für die Verwendung von Google-Modellen (Gemini)
pip install "mllmcelltype[gemini]"

# Um alle optionalen Abhängigkeiten auf einmal zu installieren
pip install "mllmcelltype[all]"
```

Wenn Sie auf Fehler wie `ImportError: cannot import name 'genai' from 'google'` stoßen, bedeutet dies, dass Sie das entsprechende Anbieterpaket installieren müssen. Zum Beispiel:

```bash
# Für Google Gemini-Modelle
pip install google-genai
```

### Unterstützte Modelle

- **OpenAI**: GPT-5.2/GPT-5/GPT-4.1 ([API-Schlüssel](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-4.6-Opus/Claude-4.5-Sonnet/Claude-4.5-Haiku ([API-Schlüssel](https://console.anthropic.com/))
- **Google**: Gemini-3-Pro/Gemini-3-Flash ([API-Schlüssel](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen3-Max ([API-Schlüssel](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API-Schlüssel](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-M2.1 ([API-Schlüssel](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-3 ([API-Schlüssel](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4.7/GLM-4-Plus ([API-Schlüssel](https://bigmodel.cn/))
- **X.AI**: Grok-4/Grok-3 ([API-Schlüssel](https://accounts.x.ai/))
- **OpenRouter**: Zugriff auf mehrere Modelle über eine einzige API ([API-Schlüssel](https://openrouter.ai/keys))
  - Unterstützt Modelle von OpenAI, Anthropic, Meta, Google, Mistral und mehr
  - Format: 'provider/model-name' (z.B. 'openai/gpt-5.2', 'anthropic/claude-opus-4.5')
  - Kostenlose Modelle verfügbar mit `:free` Suffix (z.B. 'deepseek/deepseek-r1:free', 'deepseek/deepseek-chat:free')

## Nutzungsbeispiele

### Python

```python
# Beispiel für die Verwendung von mLLMCelltype zur Zelltyp-Annotation in Einzelzell-RNA-seq mit Scanpy
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# Hinweis: Die Protokollierung wird automatisch beim Import von mllmcelltype konfiguriert
# Sie können die Protokollierung bei Bedarf mit dem logging-Modul anpassen

# Laden Sie Ihren Einzelzell-RNA-seq-Datensatz im AnnData-Format
adata = sc.read_h5ad('your_data.h5ad')  # Ersetzen Sie durch Ihren scRNA-seq-Datensatzpfad

# Führen Sie Leiden-Clustering zur Identifizierung von Zellpopulationen durch, falls noch nicht geschehen
if 'leiden' not in adata.obs.columns:
    print("Berechne Leiden-Clustering zur Identifizierung von Zellpopulationen...")
    # Vorverarbeitung der Einzelzelldaten: Normalisierung und Log-Transformation für Genexpressionsanalyse
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # Normalisierung auf 10.000 Counts pro Zelle
        sc.pp.log1p(adata)  # Log-Transformation normalisierter Counts

    # Dimensionsreduktion: PCA für scRNA-seq-Daten berechnen
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # Informative Gene auswählen
        sc.pp.pca(adata, use_highly_variable=True)  # Hauptkomponenten berechnen

    # Zellclustering: Nachbarschaftsgraph berechnen und Leiden-Community-Erkennung durchführen
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # KNN-Graph für Clustering erstellen
    sc.tl.leiden(adata, resolution=0.8)  # Zellpopulationen mit Leiden-Algorithmus identifizieren
    print(f"Leiden-Clustering abgeschlossen, {len(adata.obs['leiden'].cat.categories)} verschiedene Zellpopulationen identifiziert")

# Markergene für jeden Zellcluster mittels differentieller Expressionsanalyse identifizieren
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # Wilcoxon-Rangsummentest zur Markererkennung

# Top-Markergene für jeden Zellcluster extrahieren zur Verwendung bei der Zelltyp-Annotation
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # Wähle die Top 10 differentiell exprimierten Gene als Marker für jeden Cluster
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# WICHTIG: mLLMCelltype benötigt Gensymbole (z.B. KCNJ8, PDGFRA) nicht Ensembl-IDs (z.B. ENSG00000176771)
# Wenn Ihr AnnData-Objekt Ensembl-IDs verwendet, konvertieren Sie diese in Gensymbole für eine genaue Annotation:
# Beispiel-Konvertierungscode:
# if 'Gene' in adata.var.columns:  # Prüfen, ob Gensymbole in den Metadaten verfügbar sind
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# WICHTIG: mLLMCelltype erfordert numerische Cluster-IDs
# Die 'cluster'-Spalte muss numerische Werte oder Werte enthalten, die in numerische konvertiert werden können.
# Nicht-numerische Cluster-IDs (z.B. "cluster_1", "T_cells", "7_0") können zu Fehlern oder unerwartetem Verhalten führen.
# Wenn Ihre Daten nicht-numerische Cluster-IDs enthalten, erstellen Sie eine Zuordnung zwischen Original-IDs und numerischen IDs:
# Beispiel-Standardisierungscode:
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# API-Schlüssel für die in der Konsensus-Annotation verwendeten großen Sprachmodelle konfigurieren
# Mindestens ein API-Schlüssel ist für die Multi-LLM-Konsensus-Annotation erforderlich
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # Für GPT-5.2/5-Modelle (empfohlen)
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Für Claude-4.6/4.5-Modelle
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # Für Google Gemini-3-Modelle
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # Für Alibaba Qwen3-Modelle
# Zusätzliche optionale LLM-Anbieter für erweiterte Konsensus-Diversität:
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # Für DeepSeek-V3-Modelle
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # Für Zhipu GLM-4-Modelle
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # Für Stepfun-Modelle
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # Für MiniMax-Modelle
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # Für Zugriff auf mehrere Modelle über OpenRouter

# Multi-LLM-Konsensus-Zelltyp-Annotation mit iterativer Beratung ausführen
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # Wörterbuch der Markergene für jeden Cluster
    species="human",            # Organismus für angemessene Zelltyp-Annotation angeben
    tissue="blood",            # Gewebekontext für genauere Annotation angeben
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen3-max"],  # Mehrere LLMs für Konsensus
    consensus_threshold=1,     # Minimaler Anteil für Konsensusübereinstimmung erforderlich
    max_discussion_rounds=3    # Anzahl der Beratungsrunden zwischen Modellen zur Verfeinerung
)

# Alternativ: OpenRouter für den Zugriff auf mehrere Modelle über eine einzige API verwenden
# Dies ist besonders nützlich für den Zugriff auf kostenlose Modelle mit dem :free Suffix
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Beispiel mit kostenlosen OpenRouter-Modellen (keine Credits erforderlich)
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Meta Llama 4 Maverick (kostenlos)
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"},  # NVIDIA Nemotron Ultra 253B (kostenlos)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},   # DeepSeek Chat v3 (kostenlos)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"}               # Microsoft MAI-DS-R1 (kostenlos)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# Finale Konsensus-Zelltyp-Annotationen aus der Multi-LLM-Beratung abrufen
final_annotations = consensus_results["consensus"]

# Konsensus-Zelltyp-Annotationen in das ursprüngliche AnnData-Objekt integrieren
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# Unsicherheitsquantifizierungsmetriken hinzufügen zur Bewertung des Annotationsvertrauens
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])  # Übereinstimmungsgrad
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])  # Annotationsunsicherheit

# Vorbereitung für Visualisierung: UMAP-Einbettungen berechnen, falls noch nicht vorhanden
# UMAP bietet eine 2D-Darstellung von Zellpopulationen für die Visualisierung
if 'X_umap' not in adata.obsm:
    print("Berechne UMAP-Koordinaten...")
    # Sicherstellen, dass Nachbarn zuerst berechnet werden
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("UMAP-Koordinaten berechnet")

# Ergebnisse mit verbesserter Ästhetik visualisieren
# Basis-Visualisierung
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='right', frameon=True, title='mLLMCelltype Konsensus-Annotationen')

# Weitere angepasste Visualisierung
import matplotlib.pyplot as plt

# Figurengröße und Stil festlegen
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Publikationsreifes UMAP erstellen
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data',
         frameon=True, title='mLLMCelltype Konsensus-Annotationen',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# Unsicherheitsmetriken visualisieren
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sc.pl.umap(adata, color='consensus_proportion', ax=ax1, title='Konsensusanteil',
         cmap='viridis', vmin=0, vmax=1, size=30)
sc.pl.umap(adata, color='entropy', ax=ax2, title='Annotationsunsicherheit (Shannon-Entropie)',
         cmap='magma', vmin=0, size=30)
plt.tight_layout()
```

### Verwendung eines einzelnen kostenlosen OpenRouter-Modells

Für Benutzer, die einen einfacheren Ansatz mit nur einem Modell bevorzugen, bietet das kostenlose Microsoft MAI-DS-R1-Modell über OpenRouter hervorragende Ergebnisse:

```python
import os
from mllmcelltype import annotate_clusters

# Hinweis: Protokollierung wird automatisch konfiguriert

# Setzen Sie Ihren OpenRouter API-Schlüssel
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Definieren Sie Markergene für jeden Cluster
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T-Zellen
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B-Zellen
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monozyten
}

# Annotieren mit kostenlosem Microsoft MAI-DS-R1 Modell
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Kostenloses Modell
)

# Annotationen ausgeben
for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

Dieser Ansatz ist schnell, genau und erfordert keine API-Credits, was ihn ideal für schnelle Analysen oder bei begrenztem API-Zugang macht.

#### Markergene aus AnnData-Objekten extrahieren

Wenn Sie Scanpy mit AnnData-Objekten verwenden, können Sie Markergene einfach direkt aus den `rank_genes_groups`-Ergebnissen extrahieren:

```python
import os
import scanpy as sc
from mllmcelltype import annotate_clusters

# Hinweis: Protokollierung wird automatisch konfiguriert

# Setzen Sie Ihren OpenRouter API-Schlüssel
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Laden und vorverarbeiten Sie Ihre Daten
adata = sc.read_h5ad('your_data.h5ad')

# Führen Sie Vorverarbeitung und Clustering durch, falls noch nicht geschehen
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata)

# Finden Sie Markergene für jeden Cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Extrahieren Sie Top-Markergene für jeden Cluster
marker_genes = {
    cluster: adata.uns['rank_genes_groups']['names'][cluster][:10].tolist()
    for cluster in adata.obs['leiden'].cat.categories
}

# Annotieren mit kostenlosem Microsoft MAI-DS-R1 Modell
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',  # an Ihren Gewebetyp anpassen
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Kostenloses Modell
)

# Fügen Sie Annotationen zum AnnData-Objekt hinzu
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Visualisieren Sie die Ergebnisse
sc.pl.umap(adata, color='cell_type', legend_loc='on data',
           frameon=True, title='Zelltypen annotiert mit MAI-DS-R1')
```

Diese Methode extrahiert automatisch die Top differentiell exprimierten Gene für jeden Cluster aus den `rank_genes_groups`-Ergebnissen, was die Integration von mLLMCelltype in Ihren Scanpy-Workflow erleichtert.

### R

> **Hinweis**: Für detailliertere R-Tutorials und Dokumentation besuchen Sie bitte die [mLLMCelltype-Dokumentationswebsite](https://cafferyang.com/mLLMCelltype/).

#### Verwendung eines Seurat-Objekts

```r
# Erforderliche Pakete laden
library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot) # Hinzugefügt für plot_grid

# Laden Sie Ihr vorverarbeitetes Seurat-Objekt
pbmc <- readRDS("your_seurat_object.rds")

# Falls Sie mit Rohdaten beginnen, führen Sie Vorverarbeitungsschritte durch
# pbmc <- NormalizeData(pbmc)
# pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
# pbmc <- ScaleData(pbmc)
# pbmc <- RunPCA(pbmc)
# pbmc <- FindNeighbors(pbmc, dims = 1:10)
# pbmc <- FindClusters(pbmc, resolution = 0.5)
# pbmc <- RunUMAP(pbmc, dims = 1:10)

# Markergene für jeden Cluster finden
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Cache-Verzeichnis einrichten, um die Verarbeitung zu beschleunigen
cache_dir <- "./mllmcelltype_cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# Wählen Sie ein Modell von einem unterstützten Anbieter
# Unterstützte Modelle umfassen:
# - OpenAI: 'gpt-5.2', 'gpt-5', 'gpt-4.1', 'o3-pro', 'o3', 'o4-mini', 'o1', 'o1-pro'
# - Anthropic: 'claude-opus-4-6-20260205', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'
# - Qwen: 'qwen3-max', 'qwen-max-2025-01-25'
# - Stepfun: 'step-3', 'step-2-16k', 'step-2-mini'
# - Zhipu: 'glm-4.7', 'glm-4-plus'
# - MiniMax: 'minimax-m2.1', 'minimax-m2'
# - Grok: 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast', 'grok-3-mini'
# - OpenRouter: Zugriff auf Modelle von mehreren Anbietern über eine einzige API. Format: 'provider/model-name'
#   - OpenAI-Modelle: 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro'
#   - Anthropic-Modelle: 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'
#   - Meta-Modelle: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google-Modelle: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral-Modelle: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Andere Modelle: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# LLMCelltype-Annotation mit mehreren LLM-Modellen ausführen
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # Gewebekontext bereitstellen
  models = c(
    "claude-sonnet-4-5-20250929",  # Anthropic
    "gpt-5.2",                   # OpenAI
    "gemini-3-pro",           # Google
    "qwen3-max"       # Alibaba
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

# Struktur der Ergebnisse ausgeben, um die Daten zu verstehen
print("Verfügbare Felder in consensus_results:")
print(names(consensus_results))

# Annotationen zum Seurat-Objekt hinzufügen
# Zelltyp-Annotationen aus consensus_results$final_annotations abrufen
cluster_to_celltype_map <- consensus_results$final_annotations

# Neue Zelltyp-Identifikatorspalte erstellen
cell_types <- as.character(Idents(pbmc))
for (cluster_id in names(cluster_to_celltype_map)) {
  cell_types[cell_types == cluster_id] <- cluster_to_celltype_map[[cluster_id]]
}

# Zelltyp-Annotationen zum Seurat-Objekt hinzufügen
pbmc$cell_type <- cell_types

# Unsicherheitsmetriken hinzufügen
# Detaillierte Konsensergebnisse mit Metriken extrahieren
consensus_details <- consensus_results$initial_results$consensus_results

# Datenrahmen mit Metriken für jeden Cluster erstellen
uncertainty_metrics <- data.frame(
  cluster_id = names(consensus_details),
  consensus_proportion = sapply(consensus_details, function(res) res$consensus_proportion),
  entropy = sapply(consensus_details, function(res) res$entropy)
)

# Unsicherheitsmetriken für jede Zelle hinzufügen
# Hinweis: seurat_clusters ist eine Metadatenspalte, die automatisch von FindClusters() erstellt wird
# Sie enthält die Cluster-ID, die jeder Zelle während des Clusterings zugewiesen wurde
# Hier verwenden wir sie, um Cluster-Level-Metriken (consensus_proportion und entropy) einzelnen Zellen zuzuordnen

# Falls Sie keine seurat_clusters-Spalte haben (z.B. wenn Sie eine andere Clustering-Methode verwendet haben),
# können Sie die aktive Identität (Idents) oder eine andere Metadatenspalte mit Cluster-IDs verwenden:
# Option 1: Aktive Identität verwenden
# current_clusters <- as.character(Idents(pbmc))
# Option 2: Eine andere Metadatenspalte verwenden, die Cluster-IDs enthält
# current_clusters <- pbmc$your_cluster_column

# Für dieses Beispiel verwenden wir die Standard-seurat_clusters-Spalte:
current_clusters <- pbmc$seurat_clusters  # Cluster-ID für jede Zelle abrufen

# Jede Zell-Cluster-ID mit den entsprechenden Metriken in uncertainty_metrics abgleichen
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# Ergebnisse für zukünftige Verwendung speichern
saveRDS(consensus_results, "pbmc_mLLMCelltype_results.rds")
saveRDS(pbmc, "pbmc_annotated.rds")

# Ergebnisse mit SCpubr für publikationsreife Plots visualisieren
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)
library(viridis)  # Für Farbpaletten

# Basis-UMAP-Visualisierung mit Standardeinstellungen
pdf("pbmc_basic_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "right") +
  ggtitle("mLLMCelltype Konsensus-Annotationen")
dev.off()

# Mehr angepasste Visualisierung mit verbessertem Styling
pdf("pbmc_custom_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  label.box = TRUE,
                  legend.position = "right",
                  pt.size = 1.0,
                  border.size = 1,
                  font.size = 12) +
  ggtitle("mLLMCelltype Konsensus-Annotationen") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

# Unsicherheitsmetriken mit erweiterten SCpubr-Plots visualisieren
# Zelltypen abrufen und benannte Farbpalette erstellen
cell_types <- unique(pbmc$cell_type)
color_palette <- viridis::viridis(length(cell_types))
names(color_palette) <- cell_types

# Zelltyp-Annotationen mit SCpubr
p1 <- SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "bottom",  # Legende unten platzieren
                  pt.size = 1.0,
                  label.size = 4,  # Kleinere Label-Schriftgröße
                  label.box = TRUE,  # Hintergrundbox zu Labels hinzufügen für bessere Lesbarkeit
                  repel = TRUE,  # Labels gegenseitig abstoßen lassen, um Überlappungen zu vermeiden
                  colors.use = color_palette,
                  plot.title = "Cell Type") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            legend.text = element_text(size = 8),
            legend.key.size = unit(0.3, "cm"),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Konsensusanteil-Feature-Plot mit SCpubr
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "consensus_proportion",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Consensus",
                       plot.title = "Consensus Proportion",
                       sequential.palette = "YlGnBu",  # Gelb-Grün-Blau-Gradient, folgt Nature Methods Standards
                       sequential.direction = 1,  # Hell-zu-Dunkel-Richtung
                       min.cutoff = min(pbmc$consensus_proportion),  # Minimalwert setzen
                       max.cutoff = max(pbmc$consensus_proportion),  # Maximalwert setzen
                       na.value = "lightgrey") +  # Farbe für fehlende Werte
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Shannon-Entropie-Feature-Plot mit SCpubr
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "entropy",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Entropy",
                       plot.title = "Shannon Entropy",
                       sequential.palette = "OrRd",  # Orange-Rot-Gradient, folgt Nature Methods Standards
                       sequential.direction = -1,  # Dunkel-zu-Hell-Richtung (umgekehrt)
                       min.cutoff = min(pbmc$entropy),  # Minimalwert setzen
                       max.cutoff = max(pbmc$entropy),  # Maximalwert setzen
                       na.value = "lightgrey") +  # Farbe für fehlende Werte
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Plots mit gleichen Breiten kombinieren
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

#### CSV-Eingabe verwenden

Sie können mLLMCelltype auch direkt mit CSV-Dateien ohne Seurat verwenden, was nützlich ist, wenn Sie bereits Markergene im CSV-Format haben:

```r
# Installieren Sie die neueste Version von mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# Notwendige Pakete laden
library(mLLMCelltype)

# Einheitliche Protokollierung konfigurieren (optional - verwendet Standardwerte, wenn nicht angegeben)
configure_logger(level = "INFO", console_output = TRUE, json_format = TRUE)

# Cache-Verzeichnis erstellen
cache_dir <- "path/to/your/cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

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
for(i in 1:length(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # Erster Teil ist der Clustername
  cluster_name <- parts[1]

  # Index als Schlüssel verwenden (0-basierter Index, kompatibel mit Seurat)
  cluster_id <- as.character(i - 1)

  # Restliche Teile sind Gene
  genes <- parts[-1]

  # NA und leere Strings herausfiltern
  genes <- genes[!is.na(genes) & genes != ""]

  # Zu marker_genes_list hinzufügen
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# API-Schlüssel setzen
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Konsensus-Annotation mit kostenpflichtigen Modellen ausführen
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # z.B. "human heart"
    models = c("gemini-3-flash",
              "gemini-3-pro",
              "qwen3-max",
              "grok-4",
              "claude-sonnet-4-5-20250929",
              "gpt-5.2"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir
  )

# Alternativ: Kostenlose OpenRouter-Modelle verwenden (keine Credits erforderlich)
# OpenRouter API-Schlüssel zur api_keys-Liste hinzufügen
api_keys$openrouter <- "your-openrouter-api-key"

# Konsensus-Annotation mit kostenlosen Modellen ausführen
free_consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # z.B. "human heart"
    models = c(
      "meta-llama/llama-4-maverick:free",      # Meta Llama 4 Maverick (kostenlos)
      "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",  # NVIDIA Nemotron Ultra 253B (kostenlos)
      "deepseek/deepseek-r1:free",   # DeepSeek Chat v3 (kostenlos)
      "deepseek/deepseek-r1:free"               # Microsoft MAI-DS-R1 (kostenlos)
    ),
    api_keys = api_keys,
    consensus_check_model = "deepseek/deepseek-r1:free",  # Kostenloses Modell für Konsensprüfung
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 2,
    cache_dir = cache_dir
  )

# Ergebnisse speichern
saveRDS(consensus_results, "your_results.rds")

# Ergebniszusammenfassung ausgeben
cat("\nErgebniszusammenfassung:\n")
cat("Verfügbare Felder:", paste(names(consensus_results), collapse=", "), "\n\n")

# Finale Annotationen ausgeben
cat("Finale Zelltyp-Annotationen:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**Hinweise zum CSV-Format**:
- Die CSV-Datei sollte Werte in der ersten Spalte haben, die als Indizes verwendet werden (diese können Clusternamen, Zahlen wie 0,1,2,3 oder 1,2,3,4 usw. sein)
- Die Werte in der ersten Spalte werden nur als Referenz verwendet und nicht an die LLMs weitergegeben
- Nachfolgende Spalten sollten Markergene für jeden Cluster enthalten
- Eine Beispiel-CSV-Datei für Katzenherzgewebe ist im Paket unter `inst/extdata/Cat_Heart_markers.csv` enthalten

Beispiel-CSV-Struktur:
```
cluster,gene
0,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
1,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
2,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
3,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

Sie können auf die Beispieldaten in Ihrem R-Skript zugreifen mit:
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### Verwendung eines einzelnen LLM-Modells

Wenn Sie nur ein einzelnes LLM-Modell anstelle des Konsensusansatzes verwenden möchten, verwenden Sie die Funktion `annotate_cell_types()`. Dies ist nützlich, wenn Sie nur Zugriff auf einen API-Schlüssel haben oder ein bestimmtes Modell bevorzugen:

```r
# Erforderliche Pakete laden
library(mLLMCelltype)
library(Seurat)

# Laden Sie Ihr vorverarbeitetes Seurat-Objekt
pbmc <- readRDS("your_seurat_object.rds")

# Markergene für jeden Cluster finden
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Wählen Sie ein Modell von einem unterstützten Anbieter
# Unterstützte Modelle umfassen:
# - OpenAI: 'gpt-5.2', 'gpt-5', 'gpt-4.1', 'o3-pro', 'o3', 'o4-mini', 'o1', 'o1-pro'
# - Anthropic: 'claude-opus-4-6-20260205', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'
# - Qwen: 'qwen3-max', 'qwen-max-2025-01-25'
# - Stepfun: 'step-3', 'step-2-16k', 'step-2-mini'
# - Zhipu: 'glm-4.7', 'glm-4-plus'
# - MiniMax: 'minimax-m2.1', 'minimax-m2'
# - Grok: 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast', 'grok-3-mini'
# - OpenRouter: Zugriff auf Modelle von mehreren Anbietern über eine einzige API. Format: 'provider/model-name'
#   - OpenAI-Modelle: 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro'
#   - Anthropic-Modelle: 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'
#   - Meta-Modelle: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google-Modelle: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral-Modelle: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Andere Modelle: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Zelltyp-Annotation mit einem einzelnen LLM-Modell ausführen
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # Gewebekontext bereitstellen
  model = "claude-sonnet-4-5-20250929",  # Ein einzelnes Modell angeben
  api_key = "your-anthropic-key",  # API-Schlüssel direkt bereitstellen
  top_gene_count = 10
)

# Ein kostenloses OpenRouter-Modell verwenden
free_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "meta-llama/llama-4-maverick:free",  # Kostenloses Modell mit :free Suffix
  api_key = "your-openrouter-key",
  top_gene_count = 10
)

# Die Ergebnisse ausgeben
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
  ggtitle("Zelltypen annotiert durch einzelnes LLM-Modell")
```

#### Verschiedene Modelle vergleichen

Sie können auch Annotationen verschiedener Modelle vergleichen, indem Sie `annotate_cell_types()` mehrmals mit verschiedenen Modellen ausführen:

```r
# Zu testende Modelle definieren
models_to_test <- c(
  "claude-sonnet-4-5-20250929",  # Anthropic
  "gpt-5.2",                      # OpenAI
  "gemini-3-pro",              # Google
  "qwen3-max"          # Alibaba
)

# API-Schlüssel für verschiedene Anbieter
api_keys <- list(
  anthropic = "your-anthropic-key",
  openai = "your-openai-key",
  gemini = "your-gemini-key",
  qwen = "your-qwen-key"
)

# Jedes Modell testen und Ergebnisse speichern
results <- list()
for (model in models_to_test) {
  provider <- get_provider(model)
  api_key <- api_keys[[provider]]

  # Annotation ausführen
  results[[model]] <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = model,
    api_key = api_key,
    top_gene_count = 10
  )

  # Zum Seurat-Objekt hinzufügen
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", model))
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results[[model]])-1)),
    to = results[[model]]
  )
}
```

### Erweiterte Konsenskonfiguration: Angabe des Konsensprüfungsmodells

Der Parameter `consensus_check_model` (R) / `consensus_model` (Python) ermöglicht es Ihnen, anzugeben, welches LLM-Modell für die Konsensprüfung und Diskussionsmoderation verwendet werden soll. Dieser Parameter ist **kritisch** für die Genauigkeit der Konsensus-Annotation, da das Konsensprüfungsmodell:

1. Die semantische Ähnlichkeit zwischen verschiedenen Zelltyp-Annotationen bewertet
2. Konsensmetriken (Anteil und Entropie) berechnet
3. Diskussionen zwischen Modellen für kontroverse Cluster moderiert und synthetisiert
4. Endgültige Entscheidungen trifft, wenn Modelle nicht übereinstimmen

**⚠️ Wichtig: Wir empfehlen dringend, die leistungsfähigsten verfügbaren Modelle für die Konsensprüfung zu verwenden, da dies die Annotationsqualität direkt beeinflusst.**

#### Empfohlene Modelle für die Konsensprüfung (nach Leistung geordnet)

1. **Anthropic Claude-Modelle** (Empfohlen)
   - `claude-sonnet-4-5-20250929`
   - `claude-opus-4-1-20250805`

2. **OpenAI-Modelle**
   - `o1` / `o1-pro` - Erweiterte Argumentationsfähigkeiten
   - `gpt-5.2` / `gpt-5` - Starke Leistung bei verschiedenen Zelltypen
   - `gpt-4.1` - Neueste GPT-4-Variante

3. **Google Gemini-Modelle**
   - `gemini-3-pro` - Spitzenleistung mit verbesserter Argumentation
   - `gemini-3-flash` - Gute Leistung mit schnellerer Verarbeitung

4. **Andere leistungsstarke Modelle**
   - `deepseek-r1` / `deepseek-reasoner` - Starke Argumentationsfähigkeiten
   - `qwen3-max` - Hervorragend für wissenschaftliche Kontexte
   - `grok-4` - Fortgeschrittenes Sprachverständnis

#### R-Paket-Verwendung

```r
# Beispiel 1: Verwendung des besten verfügbaren Modells für die Konsensprüfung (Empfohlen)
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human brain",
  models = c("gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen3-max"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # Das leistungsfähigste Modell verwenden
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Beispiel 2: Verwendung eines leistungsstarken Modells, wenn Claude Opus nicht verfügbar ist
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "mouse liver",
  models = c("gpt-5.2", "gemini-3-flash", "qwen3-max"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # Alternatives leistungsstarkes Modell
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Beispiel 3: Verwendung von OpenAIs Argumentationsmodell für komplexe Fälle
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human immune cells",
  models = c("gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash"),
  api_keys = api_keys,
  consensus_check_model = "o1",  # OpenAIs fortgeschrittenes Argumentationsmodell
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# ⚠️ NICHT EMPFOHLEN: Vermeiden Sie die Verwendung weniger leistungsfähiger oder kostenloser Modelle für die Konsensprüfung
# da dies die Annotationsgenauigkeit erheblich reduzieren kann
```

#### Python-Paket-Verwendung

```python
# Beispiel 1: Verwendung des besten verfügbaren Modells für die Konsensprüfung (Empfohlen)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen3-max"],
    consensus_model="claude-sonnet-4-5-20250929",  # Das leistungsfähigste Modell verwenden
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Beispiel 2: Verwendung des Wörterbuchformats mit einem leistungsstarken Modell
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5.2", "gemini-3-flash", "qwen3-max"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Beispiel 3: Verwendung von Googles neuestem Modell für Konsens
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "qwen3-max"],
    consensus_model={"provider": "google", "model": "gemini-3-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Beispiel 4: Standardverhalten (verwendet Qwen mit leistungsstarkem Fallback)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash"],
    # Wenn nicht angegeben, standardmäßig qwen3-max (ein leistungsstarkes Modell)
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### Best Practices für die Auswahl des Konsensmodells

1. **Genauigkeit vor Kosten priorisieren**: Das Konsensprüfungsmodell spielt eine entscheidende Rolle bei der Bestimmung der endgültigen Annotationen. Die Verwendung eines weniger leistungsfähigen Modells hier kann den gesamten Annotationsprozess beeinträchtigen.

2. **Modellverfügbarkeit**: Stellen Sie sicher, dass Sie API-Zugriff auf Ihr gewähltes Konsensmodell haben. Das System wird Fallback-Modelle verwenden, wenn die primäre Wahl nicht verfügbar ist.

3. **Konsistenz**: Verwenden Sie dasselbe leistungsstarke Modell für alle Konsensprüfungen innerhalb eines Projekts, um konsistente Bewertungskriterien sicherzustellen.

4. **Komplexe Gewebe**: Für herausfordernde Gewebe (z.B. Gehirn, Immunsystem) sollten Sie die fortschrittlichsten Modelle wie Claude Opus, O1 oder Gemini 2.5 Pro in Betracht ziehen.

5. **Standardverhalten**: 
   - R: Verwendet das erste Modell in der `models`-Liste, wenn nicht angegeben
   - Python: Standardmäßig `qwen3-max` (ein leistungsstarkes Modell) mit `claude-sonnet-4-5-20250929` als Fallback

#### Warum die Modellqualität für die Konsensprüfung wichtig ist

Das Konsensprüfungsmodell muss:
- Die semantische Ähnlichkeit zwischen verschiedenen Zelltyp-Namen genau bewerten (z.B. erkennen, dass "T-Lymphozyt" und "T-Zelle" sich auf denselben Zelltyp beziehen)
- Biologischen Kontext und hierarchische Beziehungen verstehen
- Diskussionen von mehreren Modellen synthetisieren, um zu genauen Schlussfolgerungen zu gelangen
- Zuverlässige Vertrauensmetriken für nachgelagerte Analysen bereitstellen

Die Verwendung eines weniger leistungsfähigen Modells für diese kritischen Aufgaben kann zu Folgendem führen:
- Fehlidentifikation kontroverser Cluster
- Falsche Konsensberechnungen
- Schlechte Auflösung von Meinungsverschiedenheiten zwischen Modellen
- Letztendlich weniger genaue Zelltyp-Annotationen

## Visualisierungsbeispiel

### Zelltyp-Annotationsvisualisierung

Unten ist ein Beispiel für eine publikationsreife Visualisierung, die mit mLLMCelltype und SCpubr erstellt wurde und Zelltyp-Annotationen zusammen mit Unsicherheitsmetriken (Konsensusanteil und Shannon-Entropie) zeigt:

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualisierung" width="900"/>
</div>

*Abbildung: Linkes Panel zeigt Zelltyp-Annotationen auf UMAP-Projektion. Mittleres Panel zeigt den Konsensusanteil mit einem Gelb-Grün-Blau-Gradienten (tieferes Blau zeigt stärkere Übereinstimmung zwischen LLMs). Rechtes Panel zeigt Shannon-Entropie mit einem Orange-Rot-Gradienten (tieferes Rot zeigt geringere Unsicherheit, helleres Orange zeigt höhere Unsicherheit).*

### Markergen-Visualisierung

mLLMCelltype enthält jetzt erweiterte Markergen-Visualisierungsfunktionen, die sich nahtlos in den Konsensus-Annotations-Workflow integrieren:

```r
# Erforderliche Bibliotheken laden
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# Nach Durchführung der Konsensus-Annotation
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-sonnet-4.5", "openai/gpt-5.2"),
  api_keys = list(openrouter = "your_api_key")
)

# Markergen-Visualisierungen mit Seurat erstellen
# Konsensus-Annotationen zum Seurat-Objekt hinzufügen
cluster_ids <- as.character(Idents(pbmc_data))
cell_type_annotations <- consensus_results$final_annotations[cluster_ids]

# Fehlende Annotationen behandeln
if (any(is.na(cell_type_annotations))) {
  na_mask <- is.na(cell_type_annotations)
  cell_type_annotations[na_mask] <- paste("Cluster", cluster_ids[na_mask])
}

# Zum Seurat-Objekt hinzufügen
pbmc_data@meta.data$cell_type_consensus <- cell_type_annotations

# Dotplot der Markergene erstellen
DotPlot(pbmc_data, 
        features = top_markers,
        group.by = "cell_type_consensus") + 
  RotatedAxis()

# Heatmap der Markergene erstellen
DoHeatmap(pbmc_data, 
          features = top_markers,
          group.by = "cell_type_consensus")
```

**Hauptmerkmale der Markergen-Visualisierung:**

- **DotPlot**: Zeigt sowohl den Prozentsatz der Zellen, die jedes Gen exprimieren (Punktgröße) als auch das durchschnittliche Expressionsniveau (Farbintensität)
- **Heatmap**: Zeigt skalierte Expressionswerte mit Clustering von Genen und Zelltypen
- **Nahtlose Integration**: Arbeitet direkt mit Konsensus-Annotationsergebnissen, die zum Seurat-Objekt hinzugefügt wurden
- **Standard Seurat-Funktionen**: Verwendet vertraute Seurat-Visualisierungsfunktionen für Konsistenz

Für detaillierte Anweisungen und erweiterte Anpassungsoptionen siehe die [Visualisierungsanleitung](R/vignettes/06-visualization-guide.html).

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

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. [Lesen Sie unsere vollständige Forschungsarbeit auf bioRxiv](https://doi.org/10.1101/2025.04.10.647852)

## Mitwirken

Wir begrüßen und schätzen Beiträge aus der Community! Es gibt viele Möglichkeiten, wie Sie zu mLLMCelltype beitragen können:

### Probleme melden

Wenn Sie auf Fehler stoßen, Feature-Anfragen haben oder Fragen zur Verwendung von mLLMCelltype haben, [öffnen Sie bitte ein Issue](https://github.com/cafferychen777/mLLMCelltype/issues) in unserem GitHub-Repository. Beim Melden von Bugs bitte Folgendes angeben:

- Eine klare Beschreibung des Problems
- Schritte zur Reproduktion des Problems
- Erwartetes vs. tatsächliches Verhalten
- Ihre Betriebssystem- und Paketversion-Informationen
- Relevante Code-Snippets oder Fehlermeldungen

### Pull Requests

Wir ermutigen Sie, Code-Verbesserungen oder neue Funktionen durch Pull Requests beizutragen:

1. Forken Sie das Repository
2. Erstellen Sie einen neuen Branch für Ihre Funktion (`git checkout -b feature/amazing-feature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'Add some amazing feature'`)
4. Pushen Sie zum Branch (`git push origin feature/amazing-feature`)
5. Öffnen Sie einen Pull Request

### Bereiche für Beiträge

Hier sind einige Bereiche, in denen Beiträge besonders wertvoll wären:

- Hinzufügen von Unterstützung für neue LLM-Modelle
- Verbesserung von Dokumentation und Beispielen
- Optimierung der Leistung
- Hinzufügen neuer Visualisierungsoptionen
- Erweiterung der Funktionalität für spezialisierte Zelltypen oder Gewebe
- Übersetzungen der Dokumentation in verschiedene Sprachen

### Code-Stil

Bitte folgen Sie dem vorhandenen Code-Stil im Repository. Für R-Code folgen wir im Allgemeinen dem [tidyverse style guide](https://style.tidyverse.org/). Für Python-Code folgen wir [PEP 8](https://www.python.org/dev/peps/pep-0008/).

### Community

Treten Sie unserer [Discord-Community](https://discord.gg/pb2aZdG4) bei, um Echtzeit-Updates über mLLMCelltype zu erhalten, Fragen zu stellen, Ihre Erfahrungen zu teilen oder mit anderen Benutzern und Entwicklern zusammenzuarbeiten. Dies ist ein großartiger Ort, um sich mit dem Team und anderen Benutzern zu verbinden, die an Einzelzell-RNA-seq-Analysen arbeiten.

Vielen Dank für Ihre Hilfe bei der Verbesserung von mLLMCelltype!