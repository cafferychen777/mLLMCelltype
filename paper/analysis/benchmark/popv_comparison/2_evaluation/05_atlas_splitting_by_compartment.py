# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Splitting of HLCA by compartment(s)

# Here we'll split the HLCA into three subsets for figure generation, i.e. into epithelial, immune, and endothelial-stromal sub-HLCAs, and pre-calculate the umaps, so that this does not have to be done in the figure generating scripts.

# ### Load modules and set paths:

import scanpy as sc
import os
import matplotlib.pyplot as plt

# for pretty code formatting:

# %load_ext lab_black

path_HLCA = "../../data/HLCA_core_h5ads/HLCA_v2.h5ad"
dir_HLCA_subsets = "../../data/HLCA_core_h5ads/HLCA_subsets/"

# ### Split atlas, re-calculate umaps, and store:

adata = sc.read(path_HLCA)

# set mapping of clusters to compartments:

cl2comp = {"0": "epithelial", "1": "immune", "2": "endothelial_and_stromal"}

# initiate dictionary to store the atlas subsets in:

subadatas = dict()

# Now subset to each of the specified groups using clusters, and re-calcualte neighbor graph (based on scANVI embeddign) and umaps, then store:

for cl_number, comp_name in cl2comp.items():
    subadata = adata[adata.obs.leiden_1 == cl_number, :].copy()
    sc.pp.neighbors(subadata, n_neighbors=15, use_rep="X_scanvi_emb")
    sc.tl.umap(subadata)
    subadata.obsm["X_umap_scanvi"] = subadata.obsm["X_umap"]
    subadatas[comp_name] = subadata
    subadata.write(os.path.join(dir_HLCA_subsets, f"HLCA_{comp_name}.h5ad"))
    del subadata
