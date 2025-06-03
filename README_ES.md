<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - Marco de consenso multi-modelos de lenguaje para la anotación de tipos celulares en datos de secuenciación de ARN unicelular" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=Descubre%20mLLMCelltype%3A%20Un%20marco%20de%20consenso%20multi-LLM%20para%20la%20anotaci%C3%B3n%20de%20tipos%20celulares%20en%20datos%20scRNA-seq%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Unirse%20al%20chat-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
</div>

# mLLMCelltype: Marco de Consenso Multi-Modelos de Lenguaje para la Anotación de Tipos Celulares

mLLMCelltype es un marco avanzado de consenso iterativo multi-LLM para la anotación precisa y confiable de tipos celulares en datos de secuenciación de ARN unicelular (scRNA-seq). Al aprovechar la inteligencia colectiva de múltiples modelos de lenguaje grande (OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, y OpenRouter), este marco mejora significativamente la precisión de anotación mientras proporciona una cuantificación transparente de la incertidumbre para la investigación en bioinformática y biología computacional.

## Resumen

mLLMCelltype es una herramienta de código abierto para el análisis transcriptómico unicelular que utiliza múltiples modelos de lenguaje grande para identificar tipos celulares a partir de datos de expresión génica. El software implementa un enfoque de consenso donde varios modelos analizan los mismos datos y sus predicciones se combinan, lo que ayuda a reducir errores y proporciona métricas de incertidumbre. mLLMCelltype se integra con plataformas populares de análisis unicelular como Scanpy y Seurat, permitiendo a los investigadores incorporarlo a flujos de trabajo bioinformáticos existentes. A diferencia de algunos métodos tradicionales, no requiere conjuntos de datos de referencia para la anotación.

## Tabla de Contenidos
- [Noticias](#noticias)
- [Características Principales](#características-principales)
- [Actualizaciones Recientes](#actualizaciones-recientes)
- [Estructura del Directorio](#estructura-del-directorio)
- [Instalación](#instalación)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Ejemplo de Visualización](#ejemplo-de-visualización)
- [Citación](#citación)
- [Contribuciones](#contribuciones)

## Noticias

🎉 **Abril 2025**: ¡Estamos encantados de anunciar que, en solo dos semanas desde la publicación de nuestro preprint, mLLMCelltype ha superado las 200 estrellas en GitHub! También hemos visto una gran cobertura por parte de varios medios de comunicación y creadores de contenido. Extendemos nuestro más sincero agradecimiento a todos los que han apoyado este proyecto a través de estrellas, compartiendo y contribuciones. Su entusiasmo impulsa nuestro continuo desarrollo y mejora de mLLMCelltype.

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

# O instalar desde GitHub (note el parámetro subdirectory)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

### Modelos Soportados

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([Clave API](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([Clave API](https://console.anthropic.com/))
- **Google**: Gemini-2.5-Pro/Gemini-2.0-Flash/Gemini-2.0-Flash-Lite ([Clave API](https://ai.google.dev/?authuser=2))
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

# Definir un diccionario de genes marcadores para cada cluster celular
# Cada cluster contiene genes específicos que caracterizan diferentes tipos de células inmunes en PBMC
marker_dict = {
    "cluster_0": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B"],  # Marcadores de linfocitos T CD8+
    "cluster_1": ["CD3D", "CD3E", "CD3G", "CD4", "IL7R"],  # Marcadores de linfocitos T CD4+
    "cluster_2": ["CD19", "MS4A1", "CD79A", "CD79B"],      # Marcadores de linfocitos B
    "cluster_3": ["CD14", "LYZ", "CST3", "FCGR3A", "MS4A7"]  # Marcadores de monocitos/macrófagos
}

# Configurar las claves API necesarias para acceder a los diferentes servicios de LLM
# Se pueden proporcionar múltiples claves para utilizar diversos modelos en el proceso de consenso
api_keys = {
    "openai": "sk-...",        # Clave API de OpenAI para acceder a modelos como GPT-4o y GPT-4.1
    "anthropic": "sk-ant-...", # Clave API de Anthropic para acceder a modelos como Claude-3.7 y Claude-3.5
    "google": "...",          # Clave API de Google para acceder a modelos como Gemini-2.5-pro
    "qwen": "..."             # Clave API de Qwen para acceder a modelos como Qwen2.5
}

# Iniciar el proceso de anotación de tipos celulares utilizando el marco de consenso multi-LLM
# Esta función coordina la deliberación entre múltiples LLMs para determinar los tipos celulares
results = annotate_cell_types(
    marker_dict=marker_dict,  # Diccionario con genes marcadores específicos para cada cluster
    api_keys=api_keys,        # Claves API para los diferentes servicios de LLM
    num_llms=3,               # Utilizar 3 modelos LLM diferentes para generar anotaciones independientes
    consensus_method="discussion",  # Utilizar el método de discusión para resolver discrepancias entre LLMs
    rounds=2,                 # Realizar 2 rondas de discusión para refinar las anotaciones y resolver conflictos
    return_discussion=True    # Incluir el historial completo de discusión en los resultados para transparencia
)

# Mostrar las anotaciones finales de tipos celulares obtenidas por consenso de los LLMs
print("\nAnotaciones finales:")
for cluster, annotation in results["annotations"].items():
    print(f"{cluster}: {annotation}")

print("\nMétricas de incertidumbre:")
for cluster, metrics in results["uncertainty_metrics"].items():
    print(f"{cluster}: Proporción de consenso = {metrics['consensus_proportion']:.2f}, Entropía = {metrics['entropy']:.2f}")
```

### R

> **Nota**: Para tutoriales y documentación más detallados en R, visite el [sitio web de documentación de mLLMCelltype](https://cafferyang.com/mLLMCelltype/).

```r
# Cargar la biblioteca mLLMCelltype para la anotación de tipos celulares basada en consenso multi-LLM
library(mLLMCelltype)

# Definir una lista de genes marcadores para cada cluster celular en formato R
# Cada cluster contiene genes específicos que caracterizan diferentes tipos de células inmunes en PBMC
marker_list <- list(
  cluster_0 = c("CD3D", "CD3E", "CD3G", "CD8A", "CD8B"),  # Marcadores de linfocitos T CD8+
  cluster_1 = c("CD3D", "CD3E", "CD3G", "CD4", "IL7R"),  # Marcadores de linfocitos T CD4+
  cluster_2 = c("CD19", "MS4A1", "CD79A", "CD79B"),      # Marcadores de linfocitos B
  cluster_3 = c("CD14", "LYZ", "CST3", "FCGR3A", "MS4A7")  # Marcadores de monocitos/macrófagos
)

# Configurar las claves API necesarias para acceder a los diferentes servicios de LLM
# La función set_api_keys almacena las claves de forma segura para su uso en la sesión actual
set_api_keys(
  openai = "sk-...",        # Clave API de OpenAI para acceder a modelos como GPT-4o y GPT-4.1
  anthropic = "sk-ant-...", # Clave API de Anthropic para acceder a modelos como Claude-3.7 y Claude-3.5
  google = "...",          # Clave API de Google para acceder a modelos como Gemini-2.5-pro
  qwen = "..."             # Clave API de Qwen para acceder a modelos como Qwen2.5
)

# Iniciar el proceso de anotación de tipos celulares utilizando el marco de consenso multi-LLM
# Esta función coordina la deliberación entre múltiples LLMs para determinar los tipos celulares
results <- annotate_cell_types(
  marker_list = marker_list,  # Lista con genes marcadores específicos para cada cluster
  num_llms = 3,               # Utilizar 3 modelos LLM diferentes para generar anotaciones independientes
  consensus_method = "discussion",  # Utilizar el método de discusión para resolver discrepancias entre LLMs
  rounds = 2,                 # Realizar 2 rondas de discusión para refinar las anotaciones y resolver conflictos
  return_discussion = TRUE    # Incluir el historial completo de discusión en los resultados para transparencia
)

# Mostrar las anotaciones finales de tipos celulares obtenidas por consenso de los LLMs
print("Anotaciones finales:")
print(results$annotations)  # Imprimir el diccionario de anotaciones finales para cada cluster

print("Métricas de incertidumbre:")
print(results$uncertainty_metrics)
```

#### Ejemplo de entrada CSV

También puede usar mLLMCelltype directamente con archivos CSV sin Seurat, lo que es útil cuando ya tiene genes marcadores en formato CSV:

```r
# Instalar la versión más reciente de mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# Cargar paquetes necesarios
library(mLLMCelltype)

# Crear directorios de caché y registros
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# Leer el contenido del archivo CSV
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# Omitir la línea de encabezado
data_lines <- file_content[-1]

# Convertir datos a formato de lista, usando índices numéricos como claves
marker_genes_list <- list()
cluster_names <- c()

# Primero recopilar todos los nombres de clústeres
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# Luego crear marker_genes_list con índices numéricos
for(i in seq_along(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # La primera parte es el nombre del clúster
  cluster_name <- parts[1]

  # Usar índice como clave (base 0, compatible con Seurat)
  cluster_id <- as.character(i - 1)

  # El resto son genes
  genes <- parts[-1]

  # Filtrar NA y cadenas vacías
  genes <- genes[!is.na(genes) & genes != ""]

  # Agregar a marker_genes_list
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# Configurar claves API
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Ejecutar anotación de consenso
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # Ejemplo: "human heart"
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

# Guardar resultados
saveRDS(consensus_results, "your_results.rds")

# Imprimir resumen de resultados
cat("\nResumen de resultados:\n")
cat("Campos disponibles:", paste(names(consensus_results), collapse=", "), "\n\n")

# Imprimir anotaciones finales
cat("Anotaciones finales de tipos celulares:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**Notas sobre el formato CSV**:
- La primera columna del archivo CSV puede contener cualquier valor (como nombres de clústeres, secuencias numéricas como 0,1,2,3 o 1,2,3,4, etc.), que se usarán como índices
- Los valores en la primera columna solo se usan como referencia y no se pasan a los modelos LLM
- Las columnas siguientes deben contener genes marcadores para cada clúster
- Se incluye un archivo CSV de ejemplo para tejido cardíaco de gato en el paquete: `inst/extdata/Cat_Heart_markers.csv`

Ejemplo de estructura CSV:
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

Puede acceder a los datos de ejemplo en su script R usando:
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

## Integración con Scanpy

```python
import scanpy as sc
from mllmcelltype import annotate_from_anndata

# Cargar el conjunto de datos PBMC3K de ejemplo que contiene perfiles de expresión génica de células mononucleares de sangre periférica
adata = sc.datasets.pbmc3k()

# Realizar el preprocesamiento estándar del conjunto de datos de scRNA-seq
# Normalizar los recuentos por célula para corregir las diferencias en la profundidad de secuenciación
sc.pp.normalize_per_cell(adata)
# Transformación logarítmica para estabilizar la varianza y hacer que los datos sean más adecuados para análisis posteriores
sc.pp.log1p(adata)
# Seleccionar los 2000 genes más variables para reducir dimensionalidad y enfocarse en genes informativos
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
# Análisis de componentes principales para reducir dimensionalidad mientras se conserva la variación biológica significativa
sc.pp.pca(adata)
# Construir un grafo de vecindad para identificar células similares basado en perfiles de expresión
sc.pp.neighbors(adata)
# Aplicar el algoritmo de clustering Leiden para identificar grupos de células con perfiles de expresión similares
sc.tl.leiden(adata, resolution=0.8)
# Identificar genes marcadores diferencialmente expresados para cada cluster usando el test de Wilcoxon
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')

# Configurar las claves API necesarias para acceder a los servicios de LLM
api_keys = {
    "openai": "sk-...",  # Clave API de OpenAI para acceder a modelos como GPT-4o
    "anthropic": "sk-ant-..."  # Clave API de Anthropic para acceder a modelos como Claude-3.7
}

# Ejecutar el proceso de anotación de tipos celulares utilizando el marco de consenso multi-LLM
# Esta función integra directamente con objetos AnnData de Scanpy para una experiencia fluida
adata = annotate_from_anndata(
    adata=adata,  # Objeto AnnData que contiene datos de expresión génica y resultados de clustering
    cluster_key='leiden',  # Especificar la columna en adata.obs que contiene las etiquetas de cluster
    api_keys=api_keys,  # Proporcionar las claves API para los servicios LLM
    num_llms=2,  # Utilizar 2 modelos LLM diferentes para generar anotaciones independientes
    top_n_genes=20,  # Considerar los 20 genes marcadores más significativos para cada cluster
    consensus_method="discussion",  # Utilizar el método de discusión para resolver discrepancias entre LLMs
    rounds=2,  # Número de rondas de discusión para refinar las anotaciones
    add_to_obs=True,  # Guardar las anotaciones finales en adata.obs para fácil acceso y visualización
    add_uncertainty_metrics=True  # Incluir métricas de incertidumbre para evaluar la confianza de las anotaciones
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

## Contribuciones

¡Damos la bienvenida y agradecemos las contribuciones de la comunidad! Hay muchas formas en las que puede contribuir a mLLMCelltype:

### Reportar Problemas

Si encuentra errores, tiene solicitudes de funciones o preguntas sobre el uso de mLLMCelltype, por favor [abra un issue](https://github.com/cafferychen777/mLLMCelltype/issues) en nuestro repositorio de GitHub. Al reportar errores, incluya:

- Una descripción clara del problema
- Pasos para reproducir el problema
- Comportamiento esperado vs. comportamiento real
- Información sobre su sistema operativo y versiones de paquetes
- Fragmentos de código relevantes o mensajes de error

### Pull Requests

Le animamos a contribuir con mejoras de código o nuevas funciones a través de pull requests:

1. Haga un fork del repositorio
2. Cree una nueva rama para su función (`git checkout -b feature/amazing-feature`)
3. Confirme sus cambios (`git commit -m 'Añadir una función increíble'`)
4. Envíe a la rama (`git push origin feature/amazing-feature`)
5. Abra un Pull Request

### Áreas para Contribución

Aquí hay algunas áreas donde las contribuciones serían particularmente valiosas:

- Añadir soporte para nuevos modelos LLM
- Mejorar la documentación y ejemplos
- Optimizar el rendimiento
- Añadir nuevas opciones de visualización
- Extender la funcionalidad para tipos de células o tejidos especializados
- Traducciones de la documentación a diferentes idiomas

### Estilo de Código

Por favor, siga el estilo de código existente en el repositorio. Para código R, generalmente seguimos la [guía de estilo tidyverse](https://style.tidyverse.org/). Para código Python, seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/).

¡Gracias por ayudar a mejorar mLLMCelltype!