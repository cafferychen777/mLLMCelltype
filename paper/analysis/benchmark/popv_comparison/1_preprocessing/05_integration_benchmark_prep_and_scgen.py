# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Integration benchmarking prep and integration with scGen

# In this notebook, we prepare the benchmarking atlas (subset of total atlas) for integration. As 2 methods in the benchmarking require cell labels, we will create common (coarse) cell labels across all cells in the benchmarking atlas.<br>
# We will also run scGen, as this method was not included in the automated integration of scIB.

import scanpy as sc
import numpy as np
import pandas as pd
import scgen # only needed for final part, running scgen
import sys

# ### create integration object from normally annotated object:

adata = sc.read(
    "../../data/HLCA_core_h5ads/benchmarking_atlas/Barb_Kras_Krop_Lafy_Meye_Mish_MishNew_Nawi_Seib_Teic_log1p.h5ad"
)

adata.obs['last_author/PI'].unique()

# ### make label-vector for scGen: level 3 where possible, otherwise level 2

lev2_adata = adata[adata.obs.ann_highest_res == 2, :].copy()
lev2_adata.shape

lev1_adata = adata[adata.obs.ann_highest_res == 1, :].copy()
lev1_adata.shape

set(lev2_adata.obs.ann_level_2)

lev1_df = lev1_adata.obs.groupby("ann_level_1").agg(
    {"ann_level_1": "count", "last_author/PI": "nunique"}
)  # , 'dataset':'first'})
lev2_df = lev2_adata.obs.groupby("ann_level_2").agg(
    {"ann_level_2": "count", "last_author/PI": "nunique"}
)

lev1_df

df_all_lev2 = adata.obs.groupby("ann_level_2").agg(
    {"ann_level_2": "count", "ann_level_3": "nunique", "last_author/PI": "nunique"}
)
df_all_lev1 = adata.obs.groupby("ann_level_1").agg(
    {"ann_level_1": "count", "ann_level_3": "nunique", "last_author/PI": "nunique"}
)

# per celltype, show proportion of cells that has max-level 2 annotations:

lev2_df["proportion under-annotated"] = np.round(
    lev2_df.ann_level_2 / df_all_lev2.loc[lev2_df.index, "ann_level_2"], 2
)
lev2_df["ncells with higher ann"] = (
    df_all_lev2.loc[lev2_df.index, "ann_level_2"] - lev2_df.ann_level_2
)
lev2_df.rename(
    columns={"ann_level_2": "n_cells", "last_author/PI": "n_datasets"}, inplace=True
)

lev2_df

# Plan for level 2:
# 1) set all blood vessels to level 2 annotation
# 2) set all fibroblasts to level 2 annotation
# 3) remove level 2 lymphoid annotations, mixed bags
# 4) set all mesothelium to level 2 annotation
# 5) set all smooth muscle to level 2 annotation
#

lev1_df["proportion under-annotated"] = np.round(
    lev1_df.ann_level_1 / df_all_lev1.loc[lev1_df.index, "ann_level_1"], 2
)
lev1_df["ncells with finer ann"] = (
    df_all_lev1.loc[lev1_df.index, "ann_level_1"] - lev1_df.ann_level_1
)
lev1_df.rename(
    columns={"ann_level_1": "n_cells", "last_author/PI": "n_datasets"}, inplace=True
)

lev1_df

# Plan for level 1:
# 1) take out Cycling cells (since probably very mixed) and Unicorns and Artifacts
# 2) take out epithelial level 1 (from dataset Sasha, "Alveolar epithelial type 2 cells + Secretory cells", he also has clusters of these separated.
# 3) take out level1 stroma (from Sasha's and from Lafyatis data, small and very mixed clusters)

# ## plan execution

n_cells_removed = dict()

# ### level 1:

obs = adata.obs.copy()

# convert ann_level_... columns to list instead of categories, so that it's easier to add new categories:
for l in range(1, 6):
    obs["ann_level_" + str(l)] = obs["ann_level_" + str(l)].tolist()

obs.shape

# 1.1: take out cycling and unicorns and artefacts
n_cells_before = obs.shape[0]
obs = obs.loc[
    [ann not in ["Cycling cells", "Unicorns and artifacts"] for ann in obs.ann_level_1],
    :,
]
print(
    "Number of cycling + unicorn and artifact cells removed:",
    n_cells_before - obs.shape[0],
)

n_cells_removed["Cycling cells"] = sum(adata.obs.ann_level_1 == "Cycling cells")
print("number of cycling cells:", n_cells_removed["Cycling cells"])

# 1.2 and 1.3: take out level 1 epithelial and level 1 stroma
n_cells_before = obs.shape[0]
not_lev_1_epi_or_stroma = obs.index[
    [ann not in ["1_Epithelial", "1_Stroma"] for ann in obs.ann_level_2.values]
].tolist()
obs = obs.loc[not_lev_1_epi_or_stroma, :]
n_cells_removed["1_Epithelial_and_1_Stroma"] = n_cells_before - obs.shape[0]
print(
    "Number of lev1 epithelial and stroma cells removed: ",
    n_cells_removed["1_Epithelial_and_1_Stroma"]
)


# ### level 2:

# 2.1: set all blood vessels to level 2 annotation:
lev2_bld_vs_cells = obs.index[obs.ann_level_2.values == "Blood vessels"].tolist()
obs.loc[lev2_bld_vs_cells, ["ann_level_3", "ann_level_4", "ann_level_5"]] = [
    "2_Blood vessels",
    "2_Blood vessels",
    "2_Blood vessels",
]
print("Number of cells set to level 2 Blood vessels:", len(lev2_bld_vs_cells))

# 2.2: set all fibroblasts to level 2 annotation:
lev2_fib_cells = obs.index[obs.ann_level_2.values == "Fibroblast lineage"].tolist()
obs.loc[lev2_fib_cells, ["ann_level_3", "ann_level_4", "ann_level_5"]] = [
    "2_Fibroblast lineage",
    "2_Fibroblast lineage",
    "2_Fibroblast lineage",
]
print("Number of cells set to level 2 Fibroblast lineage:", len(lev2_fib_cells))

# 2.3: ...remove level 2 lymphoid annotations, mixed bags... (might want to ask Sasha and Martijn to annotate better: we lose 1300 cells)
n_cells_before = obs.shape[0]
not_lev2_lymph_cells = obs.index[obs.ann_level_3.values != "2_Lymphoid"].tolist()
obs = obs.loc[not_lev2_lymph_cells, :]
n_cells_removed["2_Lymphoid"] = n_cells_before - obs.shape[0]
print("Number of level-2 lymphoid cells removed:", n_cells_removed["2_Lymphoid"])

# 2.4: set all mesothelium cells to level 2 annotation:
lev2_mes_cells = obs.index[obs.ann_level_2.values == "Mesothelium"].tolist()
obs.loc[lev2_mes_cells, ["ann_level_3", "ann_level_4", "ann_level_5"]] = [
    "2_Mesothelium",
    "2_Mesothelium",
    "2_Mesothelium",
]
print("Number of cells set to level 2 Mesothelium:", len(lev2_mes_cells))

# 2.5: set all smooth muscle to level 2 annotation
lev2_sm_cells = obs.index[obs.ann_level_2.values == "Smooth Muscle"].tolist()
obs.loc[lev2_sm_cells, ["ann_level_3", "ann_level_4", "ann_level_5"]] = [
    "2_Smooth Muscle",
    "2_Smooth Muscle",
    "2_Smooth Muscle",
]
print("Number of cells set to level 2 Smooth Muscle:", len(lev2_sm_cells))

n_cells_removed

np.sum([x for x in n_cells_removed.values()])

adata_scgen = adata[obs.index, :].copy()

adata_scgen.shape

adata_scgen.obs["scgen_labels"] = obs.loc[adata_scgen.obs.index, "ann_level_3"]

set(adata_scgen.obs.scgen_labels)

sc.pl.umap(
    adata_scgen,
    color=["last_author/PI", "ann_level_2", "ann_level_3", "scgen_labels"],
    ncols=1,
)

adata_scgen.write(
    "../../data/HLCA_core_h5ads/benchmarking_atlas/Barb_Kras_Krop_Lafy_Meye_Mish_MishNew_Nawi_Seib_Teic_log1p_scGEN_INPUT.h5ad"
)

adata.obs.ann_highest_res = pd.Categorical(adata.obs.ann_highest_res)

sc.pl.umap(adata, color=['last_author/PI','ann_level_2','ann_level_3', 'ann_highest_res'], ncols=1)

# take out cells that don't have level 3 annotations, and keep only highly variable genes:
adata = adata[[highestres in [3,4,5] for highestres in adata.obs.ann_highest_res],:].copy()
adata = adata[:,adata.var.highly_variable].copy()

adata.write(
    "../../data/HLCA_core_h5ads/benchmarking_atlas/Barb_Kras_Krop_Lafy_Meye_Mish_MishNew_Nawi_Seib_Teic_log1p_scGEN_INPUT_gene_filtered.h5ad"
)

sc.pl.umap(adata, color=['last_author/PI','ann_level_2','ann_level_3', 'ann_highest_res'], ncols=1)

# ### Run scgen:

# make sure this is 1.1.5!

scgen.__version__

# specify data to load:

scaled_or_unscaled = "unscaled" # choose "scaled" or "unscaled"
hvg_or_full = "hvg" # choose "hvg" or "full_feature"

# load data, this is data pre-formatted etc. during integration benchmark, but derived from data as prepared above:

adata = sc.read("../../results/integration_benchmarking/benchmarking_results/prepare/{}/{}/adata_pre.h5ad".format(scaled_or_unscaled, hvg_or_full))

adata.shape

# run scgen:

network = scgen.VAEArithKeras(x_dimension= adata.shape[1], model_path="./models/batch" )

network.train(train_data=adata, n_epochs=50,batch_size=128,verbose=True)

corrected_adata =  scgen.batch_removal(network, adata, batch_key="dataset", cell_label_key="scgen_labels")

corrected_adata.obs.tail(5)

# Note that scgen creates corrected counts, and that a PCA of those corrected counts will create a corrected embedding:

corrected_adata = sc.pp.pca(corrected_adata, n_comps=50, copy=True)

sc.pp.neighbors(corrected_adata)
sc.tl.umap(corrected_adata)

sc.pl.umap(corrected_adata, color=["dataset", "scgen_labels"], wspace=.5, frameon=False)

# store result

corrected_adata.obsm['X_emb'] = corrected_adata.obsm['X_pca']

corrected_adata.write("../../results/integration_benchmarking/benchmarking_results/scgen_benchmarking_results/integration/{}/{}/scgen.h5ad".format(scaled_or_unscaled, hvg_or_full))

# store embedding

emb = pd.DataFrame(index=corrected_adata.obs.index)
for col in ['scgen_labels','dataset']:
    emb[col] = corrected_adata.obs[col]
emb.index.set_names("CellID", inplace=True)
emb['UMAP1'] = corrected_adata.obsm["X_umap"][:,0]
emb['UMAP2'] = corrected_adata.obsm["X_umap"][:,1]

emb.to_csv("../../results/integration_benchmarking/benchmarking_results/scgen_benchmarking_results/embeddings/{}/{}/scgen_full.csv".format(scaled_or_unscaled, hvg_or_full))
