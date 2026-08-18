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

# ## Second round of cell filtering after SCRAN normalization

# As SCRAN, our normalization method, creates extreme outputs for some cells (cells with a relatively low number of genes detected, maybe empty droplets?), we filter those out after SCRAN normalization. SCRAN normalization itself was performed without notebook, using a python script you can find in the scripts folder of the repository.

# #### load modules

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# for pretty code formatting, but not needed to run the notebook:

# %load_ext lab_black

# #### set paths:

# input data
path_scran_normalized_data = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_SCRAN_normalized.h5ad"
# filtered out cells output
path_cells_removed_data = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_SCRAN_normalized_cells_removed.h5ad"
# cells that passed filtering output
path_scran_filtered_data = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_SCRAN_normalized_filt.h5ad"

# #### perform filtering:

# import scran-normalized adata

adata = sc.read(path_scran_normalized_data)

# check SCRAN results:

# size factor should not be 0 anywhere:

sum(adata.obs.size_factors == 0)

# check some normalized counts:

adata.X[0, :7].toarray(0)

# their raw counterparts:

adata.layers["counts"][0, :7].toarray()

# and the normalized counts multiplied by size factors, to check if these recreate the raw counts:

adata.X[0, :7].toarray() * adata.obs.size_factors[0]

# sanity check: total counts should correlate well with size factor:

plt.scatter(adata.obs.total_counts, adata.obs.size_factors, s=1)
plt.xlabel("total counts")
plt.ylabel("size factor")
plt.show()

# calculate post-normalization total counts:

new_totals = np.array(np.sum(adata.X, axis=1))

# check distribution (there are few cells with very high total counts, we need to filter those):

plt.hist(np.log10(new_totals), bins=50)
plt.show()

np.max(new_totals)

# exploratory plots, red lines represent filtering settings (max_log10(total counts post normalization)=10**5, min_size_factor=0.01)

plt.scatter(
    adata.obs.n_genes_detected.values,
    np.log10(new_totals),
    s=1,
)
plt.hlines(y=5, xmin=0, xmax=10000, color="red")
plt.xlabel("n genes detected")
plt.ylabel("log10(total counts after normalization)")
plt.title("norm total counts vs n genes detected")
plt.show()

plt.hist(np.log10(new_totals), bins=50)
plt.xlabel("log10(total counts after normalization)")
plt.ylabel("n cells")
plt.vlines(x=5, ymin=0, ymax=130000, color="red")
plt.title("post-normalization total counts distribution")
plt.show()

plt.scatter(np.log10(adata.obs.size_factors), np.log10(adata.obs.n_genes_detected), s=1)
plt.vlines(x=np.log10(0.01), ymin=2.25, ymax=4, color="red")
plt.xlabel("SCRAN size factor")
plt.ylabel("n genes detected")
plt.title("SCRAN size factor vs n_genes_detected")
plt.show()

plt.hist(np.log10(adata.obs.size_factors), bins=50)
plt.xlabel("log10(1/size_factor)")
plt.ylabel("ncells")
plt.vlines(x=np.log10(0.01), ymin=0, ymax=60000, color="red")
plt.title("SCRAN size factor distribution")
plt.show()

plt.scatter(np.log10(adata.obs.size_factors.values), np.log10(new_totals), s=1)
plt.vlines(x=np.log10(0.01), ymin=3.5, ymax=5, color="red")
plt.hlines(y=5, xmin=-2, xmax=1, color="red")
plt.xlabel("log10(size factor)")
plt.ylabel("log10(total count after SCRAN norm)")
plt.show()

# filter:

cells_to_filter_out = adata[
    [
        norm_total_count_filter or sf_filter
        for norm_total_count_filter, sf_filter in zip(
            (new_totals > 10 ** 5).flatten().tolist(), adata.obs.size_factors < 0.01
        )
    ],
    :,
].copy()

# check number of cells filtered out, dataset source and cell type:

cells_to_filter_out

cells_to_filter_out.obs.study.value_counts()

cells_to_filter_out.obs.ann_level_3.value_counts()

# #### Store removed cells and remaining cells:

# store cells that are filtered out:
cells_to_filter_out.write(path_cells_removed_data)

# remove filtered cells from data:

filter_boolean = ~adata.obs.index.isin(cells_to_filter_out.obs.index)

adata.n_obs

adata = adata[filter_boolean, :].copy()

adata.n_obs

# store resulting anndata object:

adata.write(path_scran_filtered_data)
