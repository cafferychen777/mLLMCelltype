<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltype est un cadre de consensus multi-LLM itératif pour l'annotation des types cellulaires dans les données de séquençage d'ARN unicellulaire. En exploitant les forces complémentaires de plusieurs grands modèles de langage (GPT, Claude, Gemini, Qwen, etc.), ce cadre améliore considérablement la précision des annotations tout en fournissant une quantification transparente de l'incertitude.

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
  models = c("gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro"),
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

## Exemple de visualisation

Voici un exemple de visualisation prête à la publication créée avec mLLMCelltype et SCpubr, montrant les annotations de types cellulaires ainsi que les métriques d'incertitude (Proportion de consensus et Entropie de Shannon) :

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Figure : Le panneau de gauche montre les annotations de types cellulaires sur la projection UMAP. Le panneau du milieu affiche la proportion de consensus en utilisant un dégradé jaune-vert-bleu (un bleu plus profond indique un accord plus fort entre les LLM). Le panneau de droite montre l'entropie de Shannon en utilisant un dégradé orange-rouge (un rouge plus profond indique une incertitude plus faible, un orange plus clair indique une incertitude plus élevée).*

## Licence

MIT

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