# mLLMCelltype Reproducibility

This directory preserves the public method entry points from the original
reproducibility repository for the published mLLMCelltype article in
*Communications Biology* (2026), DOI:
[10.1038/s42003-026-10420-8](https://doi.org/10.1038/s42003-026-10420-8).

The original repository history remains available through the subtree merge.
To avoid two drifting copies of the same utilities, shared preparation,
marker, ontology, label-transfer, and visualization code now lives in the
parent [`scripts/`](../scripts/) directory. Runnable examples live in
[`examples/`](../examples/). This directory contains only the compact method
wrappers under [`methods/`](methods/).

Run commands from the parent `paper/` directory. Downloaded datasets belong in
`paper/data/`, and generated outputs belong in `paper/results/`; both are
excluded from Git.

## Data availability

All analyses in this study were performed using publicly accessible datasets.

### Comprehensive Cellular Atlas Datasets

| Dataset | Description | Source | Link |
|---------|-------------|--------|------|
| **Tabula Sapiens (TS)** | Multi-organ human cell atlas | UCSC Cell Browser | https://cells.ucsc.edu/?ds=tabula-sapiens |
| **Human Cell Landscape (HCL)** | Comprehensive human cell atlas | figshare | https://figshare.com/articles/dataset/HCL_DGE_Data/7235471 |
| **Mouse Cell Atlas (MCA)** | Comprehensive mouse cell atlas | figshare | https://figshare.com/s/865e694ad06d5857db4b |

### Tissue-Specific Atlas Datasets

| Dataset | Description | Source | Link |
|---------|-------------|--------|------|
| **Human Lung Cell Atlas (HLCA)** | Integrated human lung atlas (2.4M cells) | CELLxGENE | https://cellxgene.cziscience.com/e/9f222629-9e39-47d0-b83f-e08d610c7479.cxg/ |
| **Lung Cell Atlas (LCA)** | Lung cell atlas collection | CELLxGENE | https://cellxgene.cziscience.com/collections/5d445965-6f1a-4b68-ba3a-b8f765155d3a |
| **Developmental Human Thymus** | Thymus atlas (250K+ cells, childhood to adulthood) | CELLxGENE | https://cellxgene.cziscience.com/collections/de13e3e2-23b6-40ed-a413-e9e12d7d3910 |
| **Human Neural Organoid Cell Atlas (HNOCA)** | Neural organoid cell atlas | CELLxGENE | https://cellxgene.cziscience.com/collections/de379e5f-52d0-498c-9801-0f850823c847 |
| **Lifespan Immune Atlas** | Human peripheral immune cells across lifespan | Synapse | https://www.synapse.org/Synapse:syn61609846 |

### Cross-Tissue and Reference Datasets

| Dataset | Description | Source | Link |
|---------|-------------|--------|------|
| **HuBMAP** | Human BioMolecular Atlas Project | Azimuth Portal | https://azimuth.hubmapconsortium.org/ |
| **GTEx** | Genotype-Tissue Expression project | GTEx Portal | https://gtexportal.org/home/datasets |

### Technology-Specific and Disease-Related Datasets

| Dataset | Description | Source | Accession/Link |
|---------|-------------|--------|----------------|
| **Drop-seq data** | Alternative sequencing platform | HLCA subset | Subset from HLCA by study field |
| **snRNA-seq data** | Single-nucleus RNA-seq | HLCA subset | Subset from HLCA by study field |
| **BCL (B-cell lymphoma)** | Disease dataset | Zenodo | https://zenodo.org/record/7813151 |
| **Colon Cancer** | Cancer dataset | GEO | GSE132465 |
| **Lung Cancer** | Cancer dataset | GEO | GSE131907 |

### Reference Databases for Baseline Methods

| Resource | Description | Link |
|----------|-------------|------|
| **Cell Ontology (CL)** | Standardized cell type nomenclature | https://obofoundry.org/ontology/cl.html |
| **Monaco Immune Data** | Reference for SingleR (immune tissues) | Via `celldex` R package |
| **Blueprint/ENCODE** | Reference for SingleR (broad tissues) | Via `celldex` R package |

### popV pretrained models (Hugging Face)

The model catalog changes independently of this frozen analysis. Use the canonical
[popV organization on Hugging Face](https://huggingface.co/popV) to select and
download the tissue-specific reference named by the relevant method script.
