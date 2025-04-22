<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype Logo" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

mLLMCelltype es un marco de consenso iterativo multi-LLM para la anotación de tipos celulares en datos de secuenciación de ARN unicelular. Al aprovechar las fortalezas complementarias de múltiples modelos de lenguaje grande (OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, y OpenRouter), este marco mejora significativamente la precisión de anotación mientras proporciona una cuantificación transparente de la incertidumbre.

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

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([Clave API](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([Clave API](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([Clave API](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([Clave API](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([Clave API](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([Clave API](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([Clave API](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([Clave API](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([Clave API](https://accounts.x.ai/))
- **OpenRouter**: Acceso a múltiples modelos a través de una sola API ([Clave API](https://openrouter.ai/keys))
  - Compatible con modelos de OpenAI, Anthropic, Meta, Google, Mistral y más
  - Formato: 'proveedor/nombre-modelo' (por ejemplo, 'openai/gpt-4o', 'anthropic/claude-3-opus')

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

### Uso de un Solo Modelo LLM

Si solo tiene una clave API o prefiere usar un modelo LLM específico, puede utilizar la función `annotate_cell_types()`:

```r
# Cargar objeto Seurat preprocesado
pbmc <- readRDS("your_seurat_object.rds")

# Encontrar genes marcadores para cada cluster
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Elegir un modelo de cualquier proveedor compatible
# Modelos compatibles incluyen:
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Acceso a múltiples modelos a través de una sola API. Formato: 'proveedor/nombre-modelo'
#   - Modelos OpenAI: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Modelos Anthropic: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Modelos Meta: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Modelos Google: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Modelos Mistral: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Otros modelos: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Ejecutar anotación de tipos celulares con un solo modelo LLM
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # proporcionar contexto del tejido
  model = "claude-3-7-sonnet-20250219",  # especificar un solo modelo
  api_key = "your-anthropic-key",  # proporcionar la clave API directamente
  top_gene_count = 10
)

# Imprimir resultados
print(single_model_results)

# Añadir anotaciones al objeto Seurat
# single_model_results es un vector de caracteres con una anotación por cluster
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# Visualizar resultados
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("Tipos Celulares Anotados por un Solo Modelo LLM")
```

#### Comparación de Diferentes Modelos

También puede comparar anotaciones de diferentes modelos ejecutando `annotate_cell_types()` múltiples veces con diferentes modelos:

```r
# Usar diferentes modelos para anotación
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# Crear una columna para cada modelo
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )
  
  # Crear nombre de columna basado en el modelo
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))
  
  # Añadir anotaciones al objeto Seurat
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# Visualizar resultados de diferentes modelos
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# Combinar gráficos
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## Visualización de Ejemplo

A continuación se muestra un ejemplo de visualización de calidad de publicación creada con mLLMCelltype y SCpubr, que muestra anotaciones de tipos celulares junto con métricas de incertidumbre (proporción de consenso y entropía de Shannon):

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="Visualización de mLLMCelltype" width="900"/>
</div>

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