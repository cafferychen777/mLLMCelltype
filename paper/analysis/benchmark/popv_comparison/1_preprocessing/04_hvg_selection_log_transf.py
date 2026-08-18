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

# ## Highly variable gene selection and log-transformation

# In this notebook we select highly variable genes and perform log-transformation of normalized counts for downstream analysis

# #### Import modules:

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# #### Set paths:

path_input_data = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_SCRAN_normalized_filt.h5ad"
path_output_data = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_log1p.h5ad"

# #### Perform hvg selection:

# import data:

adata = sc.read(path_input_data)


# select highly variable genes...

# function to calculate variances on *sparse* matrix
def vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


# calculate mean, variance, dispersion per gene:

means = np.mean(adata.X, axis=0)

variances = vars(adata.X, axis=0)

dispersions = variances / means

# set min_mean cutoff (base this on the plot). We do not want to include the leftmost noisy genes that have high dispersions due to low means.

min_mean = 0.06

# plot mean versus dispersion plot:
# now plot
plt.scatter(
    np.log1p(means).tolist()[0], np.log(dispersions).tolist()[0], s=2
)
plt.vlines(x=np.log1p(min_mean),ymin=-2,ymax=8,color='red')
plt.xlabel("log1p(mean)")
plt.ylabel("log(dispersion)")
plt.title("DISPERSION VERSUS MEAN")
plt.show()

# log-transform data:

sc.pp.log1p(adata)

# now calculate highly variable genes:

sc.pp.highly_variable_genes(adata, batch_key="dataset",min_mean=min_mean, flavor="cell_ranger",n_top_genes=2000)

# check selection of genes:

boolean_to_color = {
    True: "crimson",
    False: "steelblue",
}  # make a dictionary that translates the boolean to colors
hvg_colors = adata.var.highly_variable.map(boolean_to_color)  # 'convert' the boolean
# now plot
plt.scatter(
    np.log1p(means).tolist()[0], np.log(dispersions).tolist()[0], s=1, c=hvg_colors
)
plt.xlabel("log1p(mean)")
plt.ylabel("log(dispersion)")
plt.title("DISPERSION VERSUS MEAN")
plt.show()

# store

adata.write(path_output_data)
