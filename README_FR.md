<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="GitHub forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Rejoindre%20le%20chat-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://www.r-pkg.org/badges/version/mLLMCelltype" alt="CRAN version"></a>
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://cranlogs.r-pkg.org/badges/grand-total/mLLMCelltype" alt="CRAN downloads"></a>
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
  <a href="https://pypi.org/project/mllmcelltype/"><img src="https://img.shields.io/pypi/v/mllmcelltype" alt="PyPI version"></a>
</div>

# mLLMCelltype: Cadre de Consensus Multi-Modèles de Langage pour l'Annotation des Types Cellulaires

mLLMCelltype est un cadre de consensus multi-LLM itératif pour l'annotation des types cellulaires dans les données de séquençage d'ARN unicellulaire (scRNA-seq). En combinant les prédictions de plusieurs grands modèles de langage (OpenAI GPT-5.2/5, Anthropic Claude-4.6/4.5, Google Gemini-3, X.AI Grok-4, DeepSeek-V3, Alibaba Qwen3, Zhipu GLM-4, MiniMax, Stepfun, et OpenRouter), ce cadre vise à améliorer la précision des annotations tout en fournissant une quantification transparente de l'incertitude pour la recherche en bio-informatique et en biologie computationnelle.

## Résumé

mLLMCelltype est un outil open-source pour l'analyse transcriptomique unicellulaire qui utilise plusieurs grands modèles de langage pour identifier les types cellulaires à partir des données d'expression génique. Le logiciel implémente une approche de consensus où plusieurs modèles analysent les mêmes données et leurs prédictions sont combinées, ce qui aide à réduire les erreurs et fournit des métriques d'incertitude. mLLMCelltype s'intègre aux plateformes d'analyse unicellulaire populaires comme Scanpy et Seurat, permettant aux chercheurs de l'incorporer dans les flux de travail bio-informatiques existants. Contrairement à certaines méthodes traditionnelles, il ne nécessite pas de jeux de données de référence pour l'annotation.

## Table des matières
- [Actualités](#actualités)
- [Caractéristiques principales](#caractéristiques-principales)
- [Mises à jour récentes](#mises-à-jour-récentes)
- [Structure du répertoire](#structure-du-répertoire)
- [Installation](#installation)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Exemple de visualisation](#exemple-de-visualisation)
- [Citation](#citation)
- [Contributions](#contributions)

## Actualités

**Lancement de l'Application Web (18-06-2025)**

Nous sommes heureux d'annoncer le lancement de l'Application Web mLLMCelltype ! Vous pouvez maintenant accéder aux puissantes capacités d'annotation de types cellulaires de mLLMCelltype directement via votre navigateur web sans aucune installation requise.

**✨ Fonctionnalités Principales :**
- **Interface facile à utiliser** : Téléchargez vos données scRNA-seq et obtenez des annotations en quelques minutes
- **Consensus multi-LLM** : Choisissez parmi divers modèles IA incluant GPT-4, Claude, Gemini et plus
- **Traitement en temps réel** : Surveillez la progression de l'annotation avec des mises à jour en direct
- **Formats d'export multiples** : Téléchargez les résultats aux formats CSV, TSV, Excel ou JSON
- **Aucune configuration requise** : Commencez l'annotation immédiatement sans installer de packages

**🌐 Accéder à l'Application Web** : [https://mllmcelltype.com](https://mllmcelltype.com)

**⚠️ Phase de Test Beta** : L'application web est actuellement en phase de test beta. Nous accueillons vos commentaires et suggestions pour nous aider à améliorer la plateforme. Veuillez signaler tout problème ou partager votre expérience via nos [GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues) ou [communauté Discord](https://discord.gg/pb2aZdG4).

**📢 Important : Migration des Modèles Gemini (02-06-2025)**

Google a retiré plusieurs modèles Gemini 1.5 et en retirera davantage le 24 septembre 2025 :
- **Déjà retirés** : Gemini 1.5 Pro 001, Gemini 1.5 Flash 001
- **Seront retirés le 24 sept. 2025** : Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, Gemini 1.5 Flash-8B -001

**Migration recommandée** : Utilisez `gemini-3-pro` ou `gemini-3-flash` pour de meilleures performances et des capacités de raisonnement améliorées. Les alias `gemini-1.5-pro` et `gemini-1.5-flash` continueront à fonctionner jusqu'au 24 septembre 2025, car ils pointent vers les versions -002.

**📢 Important : Dépréciation des Modèles Claude (21-07-2025)**

Anthropic abandonnera plusieurs modèles Claude le 21 juillet 2025 :
- **Modèles à abandonner** : Claude 2, Claude 2.1, Claude 3 Sonnet (non versionné), Claude 3 Opus (non versionné)

**Migration recommandée** :
- Claude 2/2.1 → `claude-sonnet-4-5-20250929` ou `claude-3-5-sonnet-20241022`
- Claude 3 Sonnet → `claude-sonnet-4-5-20250929` ou `claude-3-7-sonnet-20250219`
- Claude 3 Opus → `claude-sonnet-4-5-20250929` ou `claude-3-opus-20240229`

Veuillez mettre à jour vos modèles avant le 21 juillet 2025 pour éviter toute interruption de service.

## Caractéristiques principales

- **Architecture de consensus multi-LLM** : Combine les prédictions de divers LLM pour réduire les limitations et les biais des modèles individuels
- **Processus de délibération structuré** : Permet aux LLM de partager leur raisonnement, d'évaluer les preuves et d'affiner les annotations à travers plusieurs cycles de discussion collaborative
- **Quantification transparente de l'incertitude** : Fournit des métriques quantitatives (Proportion de consensus et Entropie de Shannon) pour identifier les populations cellulaires ambiguës nécessitant une révision par des experts
- **Réduction des hallucinations** : La délibération entre modèles aide à identifier les prédictions inexactes ou non étayées grâce à une évaluation critique
- **Robustesse face au bruit d'entrée** : Maintient une haute précision même avec des listes de gènes marqueurs imparfaites grâce à la correction d'erreur collective
- **Support d'annotation hiérarchique** : Extension optionnelle pour l'analyse multi-résolution avec cohérence parent-enfant
- **Aucun ensemble de données de référence requis** : Effectue des annotations précises sans pré-entraînement ni données de référence
- **Chaînes de raisonnement complètes** : Documente l'ensemble du processus de délibération pour une prise de décision transparente
- **Intégration transparente** : Fonctionne directement avec les flux de travail standard Scanpy/Seurat et les sorties de gènes marqueurs
- **Conception modulaire** : Incorpore facilement de nouveaux LLM dès qu'ils deviennent disponibles

## Mises à jour récentes

### v1.2.3 (10-05-2025)

#### Corrections de bugs
- Correction de la gestion des erreurs lors de la vérification du consensus lorsque les réponses API sont NULL ou invalides
- Amélioration de la journalisation des erreurs pour les réponses d'erreur de l'API OpenRouter
- Ajout de vérifications robustes de NULL et de type dans la fonction check_consensus

#### Améliorations
- Diagnostic des erreurs amélioré pour les erreurs de l'API OpenRouter
- Ajout de la journalisation détaillée des messages d'erreur API et des structures de réponse
- Amélioration de la robustesse lors du traitement de formats de réponse API inattendus

### v1.2.2 (09-05-2025)

#### Corrections de bugs
- Correction de l'erreur 'argument non-caractère' qui se produisait lors du traitement des réponses API
- Ajout de vérifications de type robustes pour les réponses API sur tous les fournisseurs de modèles
- Amélioration de la gestion des erreurs pour les formats de réponse API inattendus

#### Améliorations
- Ajout de la journalisation détaillée des erreurs pour les problèmes de réponse API
- Implémentation de modèles de gestion des erreurs cohérents dans toutes les fonctions de traitement API
- Amélioration de la validation des réponses pour garantir une structure appropriée avant le traitement

### v1.2.1 (01-05-2025)

#### Améliorations
- Ajout du support pour l'API OpenRouter
- Ajout du support pour les modèles gratuits via OpenRouter
- Mise à jour de la documentation avec des exemples pour utiliser les modèles OpenRouter

### v1.2.0 (30-04-2025)

#### Fonctionnalités
- Ajout de fonctions de visualisation pour les résultats d'annotation de types cellulaires
- Ajout du support pour la visualisation des métriques d'incertitude
- Implémentation d'un algorithme amélioré de construction de consensus

### v1.1.5 (27-04-2025)

#### Corrections de bugs
- Correction d'un problème avec la validation des indices de cluster qui causait des erreurs lors du traitement de certains fichiers d'entrée CSV
- Amélioration de la gestion des erreurs pour les indices négatifs avec des messages d'erreur plus clairs

#### Améliorations
- Ajout d'un script d'exemple pour le flux de travail d'annotation basé sur CSV (cat_heart_annotation.R)
- Amélioration de la validation d'entrée avec des diagnostics plus détaillés
- Mise à jour de la documentation pour clarifier les exigences de format d'entrée CSV

Consultez [NEWS.md](R/NEWS.md) pour un changelog complet.

### Modèles pris en charge

- **OpenAI**: GPT-5.2/GPT-5/GPT-4.1 ([Clé API](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-4.6-Opus/Claude-4.5-Sonnet/Claude-4.5-Haiku ([Clé API](https://console.anthropic.com/))
- **Google**: Gemini-3-Pro/Gemini-3-Flash ([Clé API](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen3-Max ([Clé API](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([Clé API](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-M2.1 ([Clé API](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-3 ([Clé API](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4.7/GLM-4-Plus ([Clé API](https://bigmodel.cn/))
- **X.AI**: Grok-4/Grok-3 ([Clé API](https://accounts.x.ai/))
- **OpenRouter**: Accès à plusieurs modèles via une seule API ([Clé API](https://openrouter.ai/keys))
  - Prend en charge les modèles d'OpenAI, Anthropic, Meta, Google, Mistral et plus
  - Format: 'fournisseur/nom-du-modèle' (par exemple, 'openai/gpt-5.2', 'anthropic/claude-opus-4.5')
  - Modèles gratuits disponibles avec le suffixe `:free` (par exemple, 'deepseek/deepseek-r1:free', 'deepseek/deepseek-chat:free')

## Structure des répertoires

- `R/` : Interface et implémentation en langage R
- `python/` : Interface et implémentation Python

## Installation

### Version R

```r
# Installation depuis CRAN (recommandé)
install.packages("mLLMCelltype")

# Ou installer la version de développement depuis GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Version Python

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZgmtlaORogSy0-QsaF0CHwFWOyOD26d2?usp=sharing)

**Démarrage Rapide** : Essayez mLLMCelltype instantanément dans Google Colab sans aucune installation ! Cliquez sur le badge ci-dessus pour ouvrir notre notebook interactif avec des exemples et un guide étape par étape.

```bash
# Installation depuis PyPI
pip install mllmcelltype

# Ou installation depuis GitHub (notez le paramètre subdirectory)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### Note importante sur les dépendances

mLLMCelltype utilise une conception modulaire où différentes bibliothèques de fournisseurs LLM sont des dépendances optionnelles. Selon les modèles que vous prévoyez d'utiliser, vous devrez installer les packages correspondants :

```bash
# Pour utiliser les modèles OpenAI (GPT-5, etc.)
pip install "mllmcelltype[openai]"

# Pour utiliser les modèles Anthropic (Claude)
pip install "mllmcelltype[anthropic]"

# Pour utiliser les modèles Google (Gemini)
pip install "mllmcelltype[gemini]"

# Pour installer toutes les dépendances optionnelles en une fois
pip install "mllmcelltype[all]"
```

Si vous rencontrez des erreurs comme `ImportError: cannot import name 'genai' from 'google'`, cela signifie que vous devez installer le package du fournisseur correspondant. Par exemple :

```bash
# Pour les modèles Google Gemini
pip install google-genai
```

## Exemples d'utilisation

### Exemple d'utilisation en R

> **Note** : Pour des tutoriels et une documentation R plus détaillés, veuillez visiter le [site de documentation mLLMCelltype](https://cafferyang.com/mLLMCelltype/).

```r
# Charger les bibliothèques nécessaires pour l'analyse unicellulaire et l'annotation des types cellulaires
library(mLLMCelltype)  # Cadre de consensus multi-LLM pour l'annotation des types cellulaires
library(Seurat)  # Plateforme d'analyse unicellulaire largement utilisée

# Identifier les gènes marqueurs différentiellement exprimés pour chaque cluster de cellules
# only.pos = TRUE : sélectionner uniquement les gènes surexprimés dans chaque cluster
# min.pct = 0.25 : le gène doit être exprimé dans au moins 25% des cellules du cluster
# logfc.threshold = 0.25 : différence minimale d'expression entre les clusters (log-fold change)
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# Lancer le processus d'annotation des types cellulaires utilisant le cadre de consensus multi-LLM
# Cette fonction coordonne la délibération entre plusieurs LLM pour déterminer les types cellulaires
consensus_results <- interactive_consensus_annotation(
  input = markers,  # Tableau de gènes marqueurs identifiés par Seurat
  tissue_name = "human PBMC",  # Spécifier le type de tissu pour fournir un contexte biologique aux LLM
  models = c("gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-pro"),  # Utiliser les modèles LLM les plus récents
  api_keys = list(
    openai = "your_openai_api_key",
    anthropic = "your_anthropic_api_key",
    gemini = "your_gemini_api_key"
  ),
  top_gene_count = 10  # Nombre de gènes marqueurs à inclure pour chaque cluster
)

# Afficher les annotations finales de types cellulaires obtenues par consensus des LLM
print(consensus_results$final_annotations)

# Intégrer les annotations de types cellulaires consensus dans l'objet Seurat pour visualisation et analyse ultérieure
current_clusters <- as.character(Idents(seurat_obj))  # Obtenir les identifiants de cluster actuels
cell_types <- as.character(current_clusters)  # Créer une copie pour la modification
# Remplacer les identifiants de cluster par les annotations de types cellulaires correspondantes
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
# Ajouter les annotations comme nouvelle colonne dans les métadonnées de l'objet Seurat
seurat_obj$cell_type <- cell_types
```

#### Exemple d'entrée CSV

Vous pouvez également utiliser mLLMCelltype directement avec des fichiers CSV sans Seurat, ce qui est utile lorsque vous avez déjà des gènes marqueurs au format CSV :

```r
# Installer la version la plus récente de mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# Charger les packages nécessaires
library(mLLMCelltype)

# Créer des répertoires de cache et de journaux
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# Lire le contenu du fichier CSV
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# Ignorer la ligne d'en-tête
data_lines <- file_content[-1]

# Convertir les données au format liste, en utilisant des indices numériques comme clés
marker_genes_list <- list()
cluster_names <- c()

# D'abord collecter tous les noms de clusters
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# Ensuite créer marker_genes_list avec des indices numériques
for(i in seq_along(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # La première partie est le nom du cluster
  cluster_name <- parts[1]

  # Utiliser l'indice comme clé (indice à base 0, compatible avec Seurat)
  cluster_id <- as.character(i - 1)

  # Le reste sont des gènes
  genes <- parts[-1]

  # Filtrer les NA et les chaînes vides
  genes <- genes[!is.na(genes) & genes != ""]

  # Ajouter à marker_genes_list
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# Configurer les clés API
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Exécuter l'annotation par consensus
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # Exemple : "human heart"
    models = c("gemini-3-flash",
              "gemini-3-pro",
              "qwen3-max",
              "grok-4",
              "anthropic/claude-sonnet-4",
              "openai/gpt-5.2"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir,
    log_dir = log_dir
  )

# Enregistrer les résultats
saveRDS(consensus_results, "your_results.rds")

# Imprimer le résumé des résultats
cat("\nRésumé des résultats :\n")
cat("Champs disponibles :", paste(names(consensus_results), collapse=", "), "\n\n")

# Imprimer les annotations finales
cat("Annotations finales des types cellulaires :\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**Remarques sur le format CSV** :
- La première colonne du fichier CSV peut contenir n'importe quelle valeur (comme des noms de clusters, des séquences numériques comme 0,1,2,3 ou 1,2,3,4, etc.), qui seront utilisées comme indices
- Les valeurs de la première colonne sont uniquement utilisées comme référence et ne sont pas transmises aux modèles LLM
- Les colonnes suivantes doivent contenir des gènes marqueurs pour chaque cluster
- Un fichier CSV d'exemple pour le tissu cardiaque de chat est inclus dans le package : `inst/extdata/Cat_Heart_markers.csv`

Exemple de structure CSV :
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

Vous pouvez accéder aux données d'exemple dans votre script R en utilisant :
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### Python

```python
# Exemple d'utilisation de mLLMCelltype pour l'annotation de types cellulaires dans des données scRNA-seq avec Scanpy
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# Note : La journalisation est automatiquement configurée lors de l'importation de mllmcelltype
# Vous pouvez personnaliser la journalisation si nécessaire en utilisant le module logging

# Charger votre ensemble de données scRNA-seq au format AnnData
adata = sc.read_h5ad('your_data.h5ad')  # Remplacez par le chemin de votre ensemble de données scRNA-seq

# Effectuer le clustering Leiden pour l'identification des populations cellulaires si ce n'est pas déjà fait
if 'leiden' not in adata.obs.columns:
    print("Calcul du clustering leiden pour l'identification des populations cellulaires...")
    # Prétraiter les données unicellulaires : normaliser les comptages et log-transformer pour l'analyse d'expression génique
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # Normaliser à 10 000 comptages par cellule
        sc.pp.log1p(adata)  # Log-transformer les comptages normalisés

    # Réduction de dimensionnalité : calculer l'ACP pour les données scRNA-seq
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # Sélectionner les gènes informatifs
        sc.pp.pca(adata, use_highly_variable=True)  # Calculer les composantes principales

    # Clustering cellulaire : calculer le graphe de voisinage et effectuer la détection de communautés Leiden
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # Construire le graphe KNN pour le clustering
    sc.tl.leiden(adata, resolution=0.8)  # Identifier les populations cellulaires en utilisant l'algorithme Leiden
    print(f"Clustering Leiden terminé, {len(adata.obs['leiden'].cat.categories)} populations cellulaires distinctes identifiées")

# Identifier les gènes marqueurs pour chaque cluster cellulaire en utilisant l'analyse d'expression différentielle
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # Test de somme de rangs de Wilcoxon pour la détection de marqueurs

# Extraire les principaux gènes marqueurs pour chaque cluster cellulaire à utiliser dans l'annotation de types cellulaires
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # Sélectionner les 10 principaux gènes différentiellement exprimés comme marqueurs pour chaque cluster
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# IMPORTANT : mLLMCelltype nécessite des symboles de gènes (par ex. KCNJ8, PDGFRA) et non des ID Ensembl (par ex. ENSG00000176771)
# Si votre objet AnnData utilise des ID Ensembl, convertissez-les en symboles de gènes pour une annotation précise :
# Code de conversion exemple :
# if 'Gene' in adata.var.columns:  # Vérifier si les symboles de gènes sont disponibles dans les métadonnées
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# IMPORTANT : mLLMCelltype nécessite des ID de cluster numériques
# La colonne 'cluster' doit contenir des valeurs numériques ou des valeurs qui peuvent être converties en numérique.
# Les ID de cluster non numériques (par ex. "cluster_1", "T_cells", "7_0") peuvent causer des erreurs ou un comportement inattendu.
# Si vos données contiennent des ID de cluster non numériques, créez un mapping entre les ID originaux et les ID numériques :
# Code de standardisation exemple :
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# Configurer les clés API pour les grands modèles de langage utilisés dans l'annotation de consensus
# Au moins une clé API est requise pour l'annotation de consensus multi-LLM
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # Pour les modèles GPT-5.2/5 (recommandé)
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Pour les modèles Claude-4.6/4.5
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # Pour les modèles Google Gemini-3
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # Pour les modèles Alibaba Qwen3
# Fournisseurs LLM optionnels supplémentaires pour améliorer la diversité du consensus :
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # Pour les modèles DeepSeek-V3
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # Pour les modèles Zhipu GLM-4
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # Pour les modèles Stepfun
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # Pour les modèles MiniMax
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # Pour accéder à plusieurs modèles via OpenRouter

# Exécuter l'annotation de consensus de types cellulaires multi-LLM avec délibération itérative
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # Dictionnaire de gènes marqueurs pour chaque cluster
    species="human",            # Spécifier l'organisme pour une annotation appropriée des types cellulaires
    tissue="blood",            # Spécifier le contexte tissulaire pour une annotation plus précise
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen3-max"],  # Plusieurs LLM pour le consensus
    consensus_threshold=1,     # Proportion minimale requise pour l'accord de consensus
    max_discussion_rounds=3    # Nombre de rondes de délibération entre modèles pour le raffinement
)

# Alternativement, utiliser OpenRouter pour accéder à plusieurs modèles via une seule API
# Ceci est particulièrement utile pour accéder aux modèles gratuits avec le suffixe :free
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Exemple utilisant des modèles OpenRouter gratuits (aucun crédit requis)
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Meta Llama 4 Maverick (gratuit)
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"},  # NVIDIA Nemotron Ultra 253B (gratuit)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},   # DeepSeek Chat v3 (gratuit)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"}               # Microsoft MAI-DS-R1 (gratuit)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# Récupérer les annotations finales de consensus de types cellulaires de la délibération multi-LLM
final_annotations = consensus_results["consensus"]

# Intégrer les annotations de consensus de types cellulaires dans l'objet AnnData original
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# Ajouter des métriques de quantification d'incertitude pour évaluer la confiance de l'annotation
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])  # Niveau d'accord
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])  # Incertitude d'annotation

# Préparer pour la visualisation : calculer les embeddings UMAP s'ils ne sont pas déjà disponibles
# UMAP fournit une représentation 2D des populations cellulaires pour la visualisation
if 'X_umap' not in adata.obsm:
    print("Calcul des coordonnées UMAP...")
    # S'assurer que les voisins sont calculés en premier
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("Coordonnées UMAP calculées")

# Visualiser les résultats avec une esthétique améliorée
# Visualisation de base
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='right', frameon=True, title='Annotations de Consensus mLLMCelltype')

# Visualisation plus personnalisée
import matplotlib.pyplot as plt

# Définir la taille et le style de la figure
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Créer un UMAP plus prêt pour la publication
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data',
         frameon=True, title='Annotations de Consensus mLLMCelltype',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# Visualiser les métriques d'incertitude
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sc.pl.umap(adata, color='consensus_proportion', ax=ax1, title='Proportion de Consensus',
         cmap='viridis', vmin=0, vmax=1, size=30)
sc.pl.umap(adata, color='entropy', ax=ax2, title='Incertitude d\'Annotation (Entropie de Shannon)',
         cmap='magma', vmin=0, size=30)
plt.tight_layout()
```

### Utilisation d'un Seul Modèle OpenRouter Gratuit

Pour les utilisateurs qui préfèrent une approche plus simple avec un seul modèle, le modèle gratuit Microsoft MAI-DS-R1 via OpenRouter peut être utilisé :

```python
import os
from mllmcelltype import annotate_clusters

# Note : La journalisation est automatiquement configurée

# Définir votre clé API OpenRouter
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Définir les gènes marqueurs pour chaque cluster
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # Cellules T
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # Cellules B
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monocytes
}

# Annoter en utilisant le modèle gratuit Microsoft MAI-DS-R1
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Modèle gratuit
)

# Afficher les annotations
for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

Cette approche est rapide, précise et ne nécessite aucun crédit API, ce qui la rend idéale pour des analyses rapides ou lorsque vous avez un accès limité aux API.

#### Extraction des Gènes Marqueurs des Objets AnnData

Si vous utilisez Scanpy avec des objets AnnData, vous pouvez facilement extraire les gènes marqueurs directement des résultats de `rank_genes_groups` :

```python
import os
import scanpy as sc
from mllmcelltype import annotate_clusters

# Note : La journalisation est automatiquement configurée

# Définir votre clé API OpenRouter
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Charger et prétraiter vos données
adata = sc.read_h5ad('your_data.h5ad')

# Effectuer le prétraitement et le clustering s'ils n'ont pas déjà été effectués
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata)

# Trouver les gènes marqueurs pour chaque cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Extraire les principaux gènes marqueurs pour chaque cluster
marker_genes = {
    cluster: adata.uns['rank_genes_groups']['names'][cluster][:10].tolist()
    for cluster in adata.obs['leiden'].cat.categories
}

# Annoter en utilisant le modèle gratuit Microsoft MAI-DS-R1
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',  # ajuster selon votre type de tissu
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Modèle gratuit
)

# Ajouter les annotations à l'objet AnnData
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Visualiser les résultats
sc.pl.umap(adata, color='cell_type', legend_loc='on data',
           frameon=True, title='Types Cellulaires Annotés par MAI-DS-R1')
```

Cette méthode extrait automatiquement les principaux gènes différentiellement exprimés pour chaque cluster à partir des résultats de `rank_genes_groups`, facilitant l'intégration de mLLMCelltype dans votre flux de travail Scanpy.

### R

> **Note** : Pour des tutoriels et une documentation R plus détaillés, veuillez visiter le [site de documentation mLLMCelltype](https://cafferyang.com/mLLMCelltype/).

## Visualisation de l'incertitude

mLLMCelltype fournit deux métriques pour quantifier l'incertitude des annotations :

1. **Proportion de consensus** : La proportion de modèles qui s'accordent sur une prédiction particulière
2. **Entropie de Shannon** : Mesure l'incertitude dans la distribution des prédictions

Pour visualiser ces métriques :

```r
library(Seurat)
library(ggplot2)
library(cowplot)
library(SCpubr)

# Calculer les métriques d'incertitude
uncertainty_metrics <- calculate_uncertainty_metrics(consensus_results)

# Ajouter les métriques d'incertitude à l'objet Seurat
current_clusters <- as.character(Idents(pbmc))
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# Visualiser les annotations de types cellulaires
p1 <- SCpubr::do_DimPlot(sample = pbmc,
                       group.by = "cell_type",
                       label = TRUE,
                       repel = TRUE,
                       pt.size = 0.1) +
      ggtitle("Cell Type Annotations") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Visualiser la proportion de consensus
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "consensus_proportion",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("yellow", "green", "blue"),
                         limits = c(min(pbmc$consensus_proportion),  # Définir la valeur minimale
                                   max(pbmc$consensus_proportion)),  # Définir la valeur maximale
                         na.value = "lightgrey") +  # Couleur pour les valeurs manquantes
      ggtitle("Consensus Proportion") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Visualiser l'entropie
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "entropy",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("darkred", "red", "orange"),
                         limits = c(min(pbmc$entropy),  # Définir la valeur minimale
                                   max(pbmc$entropy)),  # Définir la valeur maximale
                         na.value = "lightgrey") +  # Couleur pour les valeurs manquantes
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Combiner les graphiques avec des largeurs égales
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

### Utilisation d'un seul modèle LLM

Si vous ne disposez que d'une seule clé API ou si vous préférez utiliser un modèle LLM spécifique, vous pouvez utiliser la fonction `annotate_cell_types()` :

```r
# Charger l'objet Seurat prétraité
pbmc <- readRDS("your_seurat_object.rds")

# Trouver les gènes marqueurs pour chaque cluster
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Choisir un modèle parmi les fournisseurs pris en charge
# Modèles pris en charge incluent :
# - OpenAI : 'gpt-5.2', 'gpt-5', 'gpt-4.1', 'o3-pro', 'o3', 'o4-mini', 'o1', 'o1-pro'
# - Anthropic : 'claude-opus-4-6-20260205', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'
# - DeepSeek : 'deepseek-chat', 'deepseek-reasoner'
# - Google : 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'
# - Qwen : 'qwen3-max', 'qwen-max-2025-01-25'
# - Stepfun : 'step-3', 'step-2-16k', 'step-2-mini'
# - Zhipu : 'glm-4.7', 'glm-4-plus'
# - MiniMax : 'minimax-m2.1', 'minimax-m2'
# - Grok : 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast', 'grok-3-mini'
# - OpenRouter : Accès à plusieurs modèles via une seule API. Format : 'fournisseur/nom-du-modèle'
#   - Modèles OpenAI : 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro'
#   - Modèles Anthropic : 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'
#   - Modèles Meta : 'meta-llama/llama-4-maverick', 'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct'
#   - Modèles Google : 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-2.5-pro'
#   - Modèles Mistral : 'mistralai/mistral-large', 'mistralai/magistral-medium-2506'
#   - Autres modèles : 'deepseek/deepseek-r1', 'deepseek/deepseek-chat-v3.1', 'microsoft/mai-ds-r1'

# Exécuter l'annotation des types cellulaires avec un seul modèle LLM
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # fournir le contexte tissulaire
  model = "claude-sonnet-4-5-20250929",  # spécifier un seul modèle
  api_key = "your-anthropic-key",  # fournir directement la clé API
  top_gene_count = 10
)

# Afficher les résultats
print(single_model_results)

# Ajouter les annotations à l'objet Seurat
# single_model_results est un vecteur de caractères avec une annotation par cluster
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# Visualiser les résultats
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("Types cellulaires annotés par un seul modèle LLM")
```

#### Comparaison de différents modèles

Vous pouvez également comparer les annotations de différents modèles en exécutant `annotate_cell_types()` plusieurs fois avec différents modèles :

```r
# Utiliser différents modèles pour l'annotation
models <- c("claude-sonnet-4-5-20250929", "gpt-5.2", "gemini-3-pro", "qwen3-max", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# Créer une colonne pour chaque modèle
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )

  # Créer un nom de colonne basé sur le modèle
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))

  # Ajouter les annotations à l'objet Seurat
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# Visualiser les résultats des différents modèles
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 4")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-5")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_5_pro", label = TRUE) + ggtitle("Gemini 2.5 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# Combiner les graphiques
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

### Configuration Avancée du Consensus : Spécification du Modèle de Vérification du Consensus

Le paramètre `consensus_check_model` (R) / `consensus_model` (Python) vous permet de spécifier quel modèle LLM utiliser pour la vérification du consensus et la modération des discussions. Ce paramètre est **critique** pour la précision de l'annotation de consensus car le modèle de vérification du consensus :

1. Évalue la similarité sémantique entre différentes annotations de types cellulaires
2. Calcule les métriques de consensus (proportion et entropie)
3. Modère et synthétise les discussions entre modèles pour les clusters controversés
4. Prend les décisions finales lorsque les modèles sont en désaccord

**⚠️ Important : Nous recommandons fortement d'utiliser les modèles les plus performants disponibles pour la vérification du consensus, car cela impacte directement la qualité de l'annotation.**

#### Modèles Recommandés pour la Vérification du Consensus (Classés par Performance)

1. **Modèles Anthropic Claude** (Recommandé)
   - `claude-sonnet-4-5-20250929`
   - `claude-opus-4-1-20250805`

2. **Modèles OpenAI**
   - `o1` / `o1-pro` - Capacités de raisonnement avancées
   - `gpt-5.2` / `gpt-5` - Performance solide sur divers types cellulaires
   - `gpt-4.1` - Dernière variante de GPT-4

3. **Modèles Google Gemini**
   - `gemini-3-pro` - Modèle Gemini avec raisonnement
   - `gemini-3-flash` - Bonne performance avec traitement plus rapide

4. **Autres Modèles Haute Performance**
   - `deepseek-r1` / `deepseek-reasoner` - Fortes capacités de raisonnement
   - `qwen3-max` - Modèle Qwen
   - `grok-4` - Compréhension avancée du langage

#### Utilisation du Package R

```r
# Exemple 1 : Utiliser le meilleur modèle disponible pour la vérification du consensus (Recommandé)
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human brain",
  models = c("gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen3-max"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # Utiliser le modèle le plus performant
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Exemple 2 : Utiliser un modèle haute performance quand Claude Opus n'est pas disponible
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "mouse liver",
  models = c("gpt-5.2", "gemini-3-flash", "qwen3-max"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",  # Modèle alternatif haute performance
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Exemple 3 : Utiliser le modèle de raisonnement d'OpenAI pour les cas complexes
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human immune cells",
  models = c("gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash"),
  api_keys = api_keys,
  consensus_check_model = "o1",  # Modèle de raisonnement avancé d'OpenAI
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# ⚠️ NON RECOMMANDÉ : Évitez d'utiliser des modèles moins performants ou gratuits pour la vérification du consensus
# car cela peut réduire significativement la précision de l'annotation
```

#### Utilisation du Package Python

```python
# Exemple 1 : Utiliser le meilleur modèle disponible pour la vérification du consensus (Recommandé)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash", "qwen3-max"],
    consensus_model="claude-sonnet-4-5-20250929",  # Utiliser le modèle le plus performant
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Exemple 2 : Utiliser le format dictionnaire avec un modèle haute performance
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5.2", "gemini-3-flash", "qwen3-max"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Exemple 3 : Utiliser le dernier modèle de Google pour le consensus
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "qwen3-max"],
    consensus_model={"provider": "google", "model": "gemini-3-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Exemple 4 : Comportement par défaut (utilise Qwen avec sauvegarde haute performance)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash"],
    # Si non spécifié, utilise qwen3-max par défaut (un modèle haute performance)
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### Meilleures Pratiques pour la Sélection du Modèle de Consensus

1. **Prioriser la Précision sur le Coût** : Le modèle de vérification du consensus joue un rôle crucial dans la détermination des annotations finales. Utiliser un modèle moins performant ici peut compromettre l'ensemble du processus d'annotation.

2. **Disponibilité du Modèle** : Assurez-vous d'avoir accès API à votre modèle de consensus choisi. Le système utilisera des modèles de secours si le choix principal n'est pas disponible.

3. **Cohérence** : Utilisez le même modèle haute performance pour toutes les vérifications de consensus au sein d'un projet pour garantir des critères d'évaluation cohérents.

4. **Tissus Complexes** : Pour les tissus difficiles (par ex. cerveau, système immunitaire), envisagez d'utiliser les modèles les plus avancés comme Claude Opus, O1 ou Gemini 2.5 Pro.

5. **Comportement Par Défaut** : 
   - R : Utilise le premier modèle dans la liste `models` si non spécifié
   - Python : Par défaut utilise `qwen3-max` (un modèle haute performance) avec `claude-sonnet-4-5-20250929` comme sauvegarde

#### Pourquoi la Qualité du Modèle est Importante pour la Vérification du Consensus

Le modèle de vérification du consensus doit :
- Évaluer avec précision la similarité sémantique entre différents noms de types cellulaires (par ex. reconnaître que "lymphocyte T" et "cellule T" font référence au même type cellulaire)
- Comprendre le contexte biologique et les relations hiérarchiques
- Synthétiser les discussions de plusieurs modèles pour atteindre des conclusions précises
- Fournir des métriques de confiance fiables pour l'analyse en aval

Utiliser un modèle moins performant pour ces tâches critiques peut conduire à :
- Mauvaise identification des clusters controversés
- Calculs de consensus incorrects
- Mauvaise résolution des désaccords entre modèles
- En fin de compte, des annotations de types cellulaires moins précises

## Exemples de Visualisation

### Visualisation de l'Annotation de Types Cellulaires

Voici un exemple de visualisation prête à la publication créée avec mLLMCelltype et SCpubr, montrant les annotations de types cellulaires ainsi que les métriques d'incertitude (Proportion de consensus et Entropie de Shannon) :

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Figure : Le panneau de gauche montre les annotations de types cellulaires sur la projection UMAP. Le panneau du milieu affiche la proportion de consensus en utilisant un dégradé jaune-vert-bleu (un bleu plus profond indique un accord plus fort entre les LLM). Le panneau de droite montre l'entropie de Shannon en utilisant un dégradé orange-rouge (un rouge plus profond indique une incertitude plus faible, un orange plus clair indique une incertitude plus élevée).*

### Visualisation des Gènes Marqueurs

mLLMCelltype inclut maintenant des fonctions améliorées de visualisation des gènes marqueurs qui s'intègrent parfaitement avec le flux de travail d'annotation de consensus :

```r
# Charger les bibliothèques requises
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# Après avoir exécuté l'annotation de consensus
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-sonnet-4.5", "openai/gpt-5.2"),
  api_keys = list(openrouter = "your_api_key")
)

# Créer des visualisations de gènes marqueurs en utilisant Seurat
# Ajouter les annotations de consensus à l'objet Seurat
cluster_ids <- as.character(Idents(pbmc))
cell_type_annotations <- consensus_results$final_annotations[cluster_ids]

# Gérer les annotations manquantes
if (any(is.na(cell_type_annotations))) {
  na_mask <- is.na(cell_type_annotations)
  cell_type_annotations[na_mask] <- paste("Cluster", cluster_ids[na_mask])
}

# Ajouter à l'objet Seurat
pbmc@meta.data$cell_type_consensus <- cell_type_annotations

# Créer un dotplot de gènes marqueurs
DotPlot(pbmc, 
        features = top_markers,
        group.by = "cell_type_consensus") + 
  RotatedAxis()

# Créer une carte de chaleur de gènes marqueurs
DoHeatmap(pbmc, 
          features = top_markers,
          group.by = "cell_type_consensus")
```

Les fonctions de visualisation fournissent :
- **DotPlot** : Montre le pourcentage de cellules exprimant chaque gène (taille du point) et le niveau d'expression moyen (intensité de la couleur)
- **Heatmap** : Visualise les valeurs d'expression mises à l'échelle avec regroupement des gènes et des types cellulaires
- **Intégration transparente** : Fonctionne directement avec les résultats d'annotation de consensus ajoutés à l'objet Seurat
- **Fonctions Seurat standard** : Utilise les fonctions de visualisation familières de Seurat pour la cohérence

## Citation

Si vous utilisez mLLMCelltype dans votre recherche, veuillez citer :

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

Vous pouvez également citer au format texte simple :

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852

## Contributions

Nous accueillons et apprécions les contributions de la communauté ! Il existe de nombreuses façons de contribuer à mLLMCelltype :

### Signaler des problèmes

Si vous rencontrez des bugs, avez des demandes de fonctionnalités ou des questions sur l'utilisation de mLLMCelltype, veuillez [ouvrir une issue](https://github.com/cafferychen777/mLLMCelltype/issues) sur notre dépôt GitHub. Lors du signalement de bugs, veuillez inclure :

- Une description claire du problème
- Les étapes pour reproduire le problème
- Comportement attendu vs comportement réel
- Informations sur votre système d'exploitation et les versions des packages
- Tout extrait de code pertinent ou messages d'erreur

### Pull Requests

Nous vous encourageons à contribuer des améliorations de code ou de nouvelles fonctionnalités via des pull requests :

1. Forkez le dépôt
2. Créez une nouvelle branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committez vos modifications (`git commit -m 'Ajouter une fonctionnalité incroyable'`)
4. Poussez vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

### Domaines de contribution

Voici quelques domaines où les contributions seraient particulièrement précieuses :

- Ajout de support pour de nouveaux modèles LLM
- Amélioration de la documentation et des exemples
- Optimisation des performances
- Ajout de nouvelles options de visualisation
- Extension des fonctionnalités pour des types cellulaires ou tissus spécialisés
- Traductions de la documentation dans différentes langues

### Style de code

Veuillez suivre le style de code existant dans le dépôt. Pour le code R, nous suivons généralement le [guide de style tidyverse](https://style.tidyverse.org/). Pour le code Python, nous suivons [PEP 8](https://www.python.org/dev/peps/pep-0008/).

Merci de nous aider à améliorer mLLMCelltype !