<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltype est un cadre de consensus multi-LLM itératif pour l'annotation des types cellulaires dans les données de séquençage d'ARN unicellulaire. En exploitant les forces complémentaires de plusieurs grands modèles de langage (OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, et OpenRouter), ce cadre améliore considérablement la précision des annotations tout en fournissant une quantification transparente de l'incertitude.

## Caractéristiques principales

- **Architecture de consensus multi-LLM** : Exploite l'intelligence collective de divers LLM pour surmonter les limitations et les biais des modèles individuels
- **Processus de délibération structuré** : Permet aux LLM de partager leur raisonnement, d'évaluer les preuves et d'affiner les annotations à travers plusieurs cycles de discussion collaborative
- **Quantification transparente de l'incertitude** : Fournit des métriques quantitatives (Proportion de consensus et Entropie de Shannon) pour identifier les populations cellulaires ambiguës nécessitant une révision par des experts
- **Réduction des hallucinations** : La délibération entre modèles supprime activement les prédictions inexactes ou non étayées grâce à une évaluation critique
- **Robustesse face au bruit d'entrée** : Maintient une haute précision même avec des listes de gènes marqueurs imparfaites grâce à la correction d'erreur collective
- **Support d'annotation hiérarchique** : Extension optionnelle pour l'analyse multi-résolution avec cohérence parent-enfant
- **Aucun ensemble de données de référence requis** : Effectue des annotations précises sans pré-entraînement ni données de référence
- **Chaînes de raisonnement complètes** : Documente l'ensemble du processus de délibération pour une prise de décision transparente
- **Intégration transparente** : Fonctionne directement avec les flux de travail standard Scanpy/Seurat et les sorties de gènes marqueurs
- **Conception modulaire** : Incorpore facilement de nouveaux LLM dès qu'ils deviennent disponibles

### Modèles pris en charge

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([Clé API](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([Clé API](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([Clé API](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([Clé API](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([Clé API](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([Clé API](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([Clé API](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([Clé API](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([Clé API](https://accounts.x.ai/))
- **OpenRouter**: Accès à plusieurs modèles via une seule API ([Clé API](https://openrouter.ai/keys))
  - Prend en charge les modèles d'OpenAI, Anthropic, Meta, Google, Mistral et plus
  - Format: 'fournisseur/nom-du-modèle' (par exemple, 'openai/gpt-4o', 'anthropic/claude-3-opus')

## Structure des répertoires

- `R/` : Interface et implémentation en langage R
- `python/` : Interface et implémentation Python

## Installation

### Version R

```r
# Installation depuis GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Version Python

```bash
pip install mllmcelltype
```

## Démarrage rapide

### Exemple d'utilisation en R

```r
library(mLLMCelltype)
library(Seurat)

# Préparer la liste des gènes marqueurs
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# Effectuer l'annotation des types cellulaires
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

# Vérifier les résultats
print(consensus_results$final_annotations)

# Ajouter les annotations à l'objet Seurat
current_clusters <- as.character(Idents(seurat_obj))
cell_types <- as.character(current_clusters)
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
seurat_obj$cell_type <- cell_types
```

### Exemple d'utilisation en Python

```python
import scanpy as sc
import mllmcelltype as mct

# Charger l'objet AnnData
adata = sc.read_h5ad("your_data.h5ad")

# Effectuer le clustering (ignorer si déjà fait)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Identifier les gènes marqueurs
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# Convertir les gènes marqueurs au format d'entrée de mLLMCelltype
markers_dict = mct.utils.convert_scanpy_markers(adata)

# Effectuer l'annotation des types cellulaires
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

# Ajouter les annotations à l'objet AnnData
adata.obs["cell_type"] = adata.obs["leiden"].map(
    lambda x: consensus_results["final_annotations"].get(x, "Unknown")
)
```

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
# - OpenAI : 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic : 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek : 'deepseek-chat', 'deepseek-reasoner'
# - Google : 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen : 'qwen-max-2025-01-25'
# - Stepfun : 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu : 'glm-4-plus', 'glm-3-turbo'
# - MiniMax : 'minimax-text-01'
# - Grok : 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter : Accès à plusieurs modèles via une seule API. Format : 'fournisseur/nom-du-modèle'
#   - Modèles OpenAI : 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Modèles Anthropic : 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Modèles Meta : 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Modèles Google : 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Modèles Mistral : 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Autres modèles : 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Exécuter l'annotation des types cellulaires avec un seul modèle LLM
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # fournir le contexte tissulaire
  model = "claude-3-7-sonnet-20250219",  # spécifier un seul modèle
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
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
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
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# Combiner les graphiques
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## Exemple de visualisation

Voici un exemple de visualisation prête à la publication créée avec mLLMCelltype et SCpubr, montrant les annotations de types cellulaires ainsi que les métriques d'incertitude (Proportion de consensus et Entropie de Shannon) :

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Figure : Le panneau de gauche montre les annotations de types cellulaires sur la projection UMAP. Le panneau du milieu affiche la proportion de consensus en utilisant un dégradé jaune-vert-bleu (un bleu plus profond indique un accord plus fort entre les LLM). Le panneau de droite montre l'entropie de Shannon en utilisant un dégradé orange-rouge (un rouge plus profond indique une incertitude plus faible, un orange plus clair indique une incertitude plus élevée).*

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