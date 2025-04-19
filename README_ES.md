<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltype es un marco de consenso iterativo multi-LLM para la anotación de tipos celulares en datos de secuenciación de ARN unicelular. Al aprovechar las fortalezas complementarias de múltiples modelos de lenguaje grande (GPT, Claude, Gemini, Grok, DeepSeek, Qwen, etc.), este marco mejora significativamente la precisión de anotación mientras proporciona una cuantificación transparente de la incertidumbre.

## Características Principales

- **Arquitectura de Consenso Multi-LLM**: Aprovecha la inteligencia colectiva de diversos LLMs para superar las limitaciones y sesgos de modelos individuales
- **Proceso de Deliberación Estructurado**: Permite a los LLMs compartir razonamientos, evaluar evidencias y refinar anotaciones a través de múltiples rondas de discusión colaborativa
- **Cuantificación Transparente de Incertidumbre**: Proporciona métricas cuantitativas (Proporción de Consenso y Entropía de Shannon) para identificar poblaciones celulares ambiguas que requieren revisión por expertos
- **Reducción de Alucinaciones**: La deliberación entre modelos suprime activamente predicciones inexactas o sin respaldo mediante evaluación crítica
- **Robustez ante Ruido de Entrada**: Mantiene alta precisión incluso con listas de genes marcadores imperfectas mediante corrección colectiva de errores
- **Soporte para Anotación Jerárquica**: Extensión opcional para análisis multi-resolución con consistencia padre-hijo
- **No Requiere Conjunto de Datos de Referencia**: Realiza anotaciones precisas sin entrenamiento previo o datos de referencia
- **Cadenas de Razonamiento Completas**: Documenta el proceso completo de deliberación para una toma de decisiones transparente
- **Integración Perfecta**: Funciona directamente con flujos de trabajo estándar de Scanpy/Seurat y salidas de genes marcadores
- **Diseño Modular**: Incorpora fácilmente nuevos LLMs a medida que estén disponibles

## Estructura de Directorios

- `R/`: Interfaz e implementación en lenguaje R
- `python/`: Interfaz e implementación en Python

## Instalación

### Versión R

```r
# Instalar desde GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Versión Python

```bash
# Instalar desde PyPI
pip install mllmcelltype

# O instalar desde GitHub
pip install git+https://github.com/cafferychen777/mLLMCelltype.git
```

### Modelos Soportados

- **OpenAI**: GPT-4.5/GPT-4o ([Clave API](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([Clave API](https://console.anthropic.com/settings/keys))
- **Google**: Gemini-1.5-Pro/Gemini-1.5-Flash ([Clave API](https://aistudio.google.com/app/apikey))
- **Alibaba**: Qwen2-72B-Instruct ([Clave API](https://dashscope.console.aliyun.com/apiKey))
- **Meta**: Llama-3.1-405B-Instruct ([Clave API](https://replicate.com/account/api-tokens))
- **Mistral**: Mistral-Large-2 ([Clave API](https://console.mistral.ai/api-keys/))
- **Cohere**: Command-R+ ([Clave API](https://dashboard.cohere.com/api-keys))

## Uso Rápido

### Python

```python
from mllmcelltype import annotate_cell_types

# Ejemplo básico con un diccionario de genes marcadores
marker_dict = {
    "cluster_0": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B"],
    "cluster_1": ["CD3D", "CD3E", "CD3G", "CD4", "IL7R"],
    "cluster_2": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "cluster_3": ["CD14", "LYZ", "CST3", "FCGR3A", "MS4A7"]
}

# Configurar claves API (solo se necesita una, pero se pueden proporcionar múltiples)
api_keys = {
    "openai": "sk-...",  # Opcional: Clave API de OpenAI
    "anthropic": "sk-ant-...",  # Opcional: Clave API de Anthropic
    "google": "...",  # Opcional: Clave API de Google
    "qwen": "..."  # Opcional: Clave API de Qwen
}

# Obtener anotaciones
results = annotate_cell_types(
    marker_dict=marker_dict,
    api_keys=api_keys,
    num_llms=3,  # Número de LLMs a utilizar (seleccionados automáticamente según las claves disponibles)
    consensus_method="discussion",  # Método de consenso: "discussion" o "voting"
    rounds=2,  # Número de rondas para el método de discusión
    return_discussion=True  # Devolver el historial completo de discusión
)

# Imprimir resultados
print("\nAnotaciones finales:")
for cluster, annotation in results["annotations"].items():
    print(f"{cluster}: {annotation}")

print("\nMétricas de incertidumbre:")
for cluster, metrics in results["uncertainty_metrics"].items():
    print(f"{cluster}: Proporción de consenso = {metrics['consensus_proportion']:.2f}, Entropía = {metrics['entropy']:.2f}")
```

### R

```r
library(mLLMCelltype)

# Ejemplo básico con una lista de genes marcadores
marker_list <- list(
  cluster_0 = c("CD3D", "CD3E", "CD3G", "CD8A", "CD8B"),
  cluster_1 = c("CD3D", "CD3E", "CD3G", "CD4", "IL7R"),
  cluster_2 = c("CD19", "MS4A1", "CD79A", "CD79B"),
  cluster_3 = c("CD14", "LYZ", "CST3", "FCGR3A", "MS4A7")
)

# Configurar claves API (solo se necesita una, pero se pueden proporcionar múltiples)
set_api_keys(
  openai = "sk-...",  # Opcional: Clave API de OpenAI
  anthropic = "sk-ant-...",  # Opcional: Clave API de Anthropic
  google = "...",  # Opcional: Clave API de Google
  qwen = "..."  # Opcional: Clave API de Qwen
)

# Obtener anotaciones
results <- annotate_cell_types(
  marker_list = marker_list,
  num_llms = 3,  # Número de LLMs a utilizar (seleccionados automáticamente según las claves disponibles)
  consensus_method = "discussion",  # Método de consenso: "discussion" o "voting"
  rounds = 2,  # Número de rondas para el método de discusión
  return_discussion = TRUE  # Devolver el historial completo de discusión
)

# Ver resultados
print("Anotaciones finales:")
print(results$annotations)

print("Métricas de incertidumbre:")
print(results$uncertainty_metrics)
```

## Integración con Scanpy

```python
import scanpy as sc
from mllmcelltype import annotate_from_anndata

# Cargar datos
adata = sc.datasets.pbmc3k()

# Preprocesamiento estándar
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.8)
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')

# Configurar claves API
api_keys = {
    "openai": "sk-...",  # Clave API de OpenAI
    "anthropic": "sk-ant-..."  # Clave API de Anthropic
}

# Anotar tipos celulares directamente desde AnnData
adata = annotate_from_anndata(
    adata=adata,
    cluster_key='leiden',  # Columna que contiene las etiquetas de cluster
    api_keys=api_keys,
    num_llms=2,  # Usar 2 LLMs
    top_n_genes=20,  # Número de genes marcadores principales a considerar
    consensus_method="discussion",
    rounds=2,
    add_to_obs=True,  # Agregar anotaciones a adata.obs
    add_uncertainty_metrics=True  # Agregar métricas de incertidumbre a adata.obs
)

# Visualizar resultados
sc.pl.umap(adata, color='cell_type', legend_loc='on data')
sc.pl.umap(adata, color='consensus_proportion', cmap='viridis')
sc.pl.umap(adata, color='entropy', cmap='inferno_r')
```

## Integración con Seurat

```r
library(Seurat)
library(mLLMCelltype)

# Cargar datos
pbmc <- readRDS("pbmc3k.rds")

# Configurar claves API
set_api_keys(
  openai = "sk-...",  # Clave API de OpenAI
  anthropic = "sk-ant-..."  # Clave API de Anthropic
)

# Extraer genes marcadores de Seurat
markers <- get_seurat_markers(pbmc, group.by = "seurat_clusters")

# Anotar tipos celulares
results <- annotate_cell_types(
  marker_list = markers,
  num_llms = 2,
  consensus_method = "discussion",
  rounds = 2
)

# Agregar anotaciones y métricas al objeto Seurat
pbmc$cell_type <- results$annotations[as.character(pbmc$seurat_clusters)]
pbmc$consensus_proportion <- results$uncertainty_metrics$consensus_proportion[as.character(pbmc$seurat_clusters)]
pbmc$entropy <- results$uncertainty_metrics$entropy[as.character(pbmc$seurat_clusters)]

# Visualizar resultados
DimPlot(pbmc, group.by = "cell_type", label = TRUE) + NoLegend()

# Visualizar métricas de incertidumbre
library(ggplot2)
library(cowplot)

# Crear gráficos para cada métrica
p1 <- DimPlot(pbmc, group.by = "cell_type", label = TRUE) + 
      NoLegend() + 
      ggtitle("Anotaciones de Tipos Celulares") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

p2 <- FeaturePlot(pbmc, features = "consensus_proportion", 
                  pt.size = 1.5,
                  cols = c("yellow", "green", "blue"),  # Gradiente amarillo-verde-azul
                  order = TRUE) +  # Ordenar células por valor
      ggtitle("Proporción de Consenso") +
      scale_color_gradientn(colors = c("yellow", "green", "blue"),
                       limits = c(min(pbmc$consensus_proportion),  # Establecer valor mínimo
                                  max(pbmc$consensus_proportion)),  # Establecer valor máximo
                       na.value = "lightgrey") +  # Color para valores faltantes
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

p3 <- FeaturePlot(pbmc, features = "entropy", 
                  pt.size = 1.5,
                  cols = c("darkred", "red", "orange"),  # Gradiente rojo oscuro-rojo-naranja
                  order = TRUE) +  # Ordenar células por valor
      ggtitle("Entropía de Shannon") +
      scale_color_gradientn(colors = c("darkred", "red", "orange"),
                       direction = -1,  # Dirección de oscuro a claro (invertido)
                       limits = c(min(pbmc$entropy),  # Establecer valor mínimo
                                  max(pbmc$entropy)),  # Establecer valor máximo
                       na.value = "lightgrey") +  # Color para valores faltantes
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Combinar gráficos con anchos iguales
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

## Ejemplo de Visualización

A continuación se muestra un ejemplo de visualización lista para publicación creada con mLLMCelltype y SCpubr, que muestra anotaciones de tipos celulares junto con métricas de incertidumbre (Proporción de Consenso y Entropía de Shannon):

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="Visualización de mLLMCelltype" width="900"/>
</div>

*Figura: El panel izquierdo muestra anotaciones de tipos celulares en proyección UMAP. El panel central muestra la proporción de consenso usando un gradiente amarillo-verde-azul (azul más profundo indica mayor acuerdo entre LLMs). El panel derecho muestra la entropía de Shannon usando un gradiente naranja-rojo (rojo más profundo indica menor incertidumbre, naranja más claro indica mayor incertidumbre).*

## Cita

Si utiliza mLLMCelltype en su investigación, por favor cite:

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

También puede citar esto en formato de texto plano:

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852