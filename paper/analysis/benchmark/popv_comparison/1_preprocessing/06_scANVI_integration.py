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

# # Integration of HLCA datasets using scANVI

# In this notebook we integrate the HLCA datasets using scANVI. Note that scANVI should be run on a GPU.

# #### import modules and set paths:

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# We use the scArches package here, as it includes scANVI as a dependency, and we will need scArches later anyway when mapping new data to the atlas.

import scanpy as sc
import anndata
import scarches as sca

# set parameters for printing and figures:

torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

# paths:

path_HLCA_unintegrated = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_log1p.h5ad"
path_HLCA_unintegrated_prepped = "../../data/HLCA_core_h5ads/HLCA_v1_integration/HLCA_v1_scANVI_input.h5ad"
dir_out = "../../results/scANVI_integration"

# ## Prepare HLCA for integration:

# ### Prepare cell-type labeling:

# Load unintegrated HLCA:

adata = sc.read(path_HLCA_unintegrated)

adata

# scANVI uses cell type labels for the integration process. We will here select specific labels to use for the integration. Where possible, we will use level 3 annotations. For cell types that are rarely/never annotated at this level, we will fall back to level 2 annotations. Cells that only have level 1 annotations will be set as "unlabeled", except for "proliferating cells", which will keep their labels.

# check which cells have only level 1 annotations:

lev1_obs = adata.obs.loc[adata.obs.ann_highest_res == 1, :].copy()
print("Number of cells with only level 1 annotation:", lev1_obs.shape[0])

# count number of cells per ann_level_1 group that have no further annotations
lev1_annotations = lev1_obs.groupby("ann_level_1").agg({"ann_level_1":"count"}).rename(columns={"ann_level_1":"n_cells_w_only_l1_ann"})
# count total cells in each ann_level_1 group
lev1_annotations['total'] = adata.obs.groupby("ann_level_1").agg({"ann_level_1":"count"}).loc[lev1_annotations.index,"ann_level_1"]

lev1_annotations

# We will let all "Proliferating cells" keep their labels, and will set all other cells as "unlabeled" (since they only make up a small part of their group)

lev1_labeled = adata.obs.index[adata.obs.ann_level_1 == "Proliferating cells"]

lev1_unlabeled = adata.obs.index[[max_res == 1 and lev1label != "Proliferating cells" for max_res, lev1label in zip(adata.obs.ann_highest_res, adata.obs.ann_level_1)]]

# sanity check (should be True):

lev1_annotations.n_cells_w_only_l1_ann.sum() == len(lev1_labeled) + len(lev1_unlabeled)

# Now check which cells have only up to level 2 annotations:

lev2_obs = adata.obs.loc[adata.obs.ann_highest_res == 2, :].copy()
print("Number of cells with only level 1/2 annotation:", lev2_obs.shape[0])

# count number of cells per ann_level_2 group that have no further annotations
lev2_annotations = lev2_obs.groupby("ann_level_2").agg({"ann_level_2":"count"}).rename(columns={"ann_level_2":"n_cells_w_only_l2_ann"})
# count total cells in each ann_level_2 group
lev2_annotations['total'] = adata.obs.groupby("ann_level_2").agg({"ann_level_2":"count"}).loc[lev2_annotations.index,"ann_level_2"]

lev2_annotations

# We will set all cells with level 2 annotations "Fibroblast lineage", "Mesothelium", "Lymphatic EC", and "Smooth Muscle" to their level 2 annotations. All other cels with only level 2 annotations will be set to unlabeled.

lev2_labels_to_keep = ["Fibroblast lineage", "Mesothelium", "Lymphatic EC", "Smooth Muscle"]
lev2_labeled = adata.obs.index[[lev2_lab in lev2_labels_to_keep for lev2_lab in adata.obs.ann_level_2]]

# sanity check (should be True):

len(lev2_labeled) == lev2_annotations.loc[lev2_labels_to_keep,"total"].sum()

lev2_unlabeled = adata.obs.index[[max_res == 2 and lev2_lab not in lev2_labels_to_keep for max_res, lev2_lab in zip(adata.obs.ann_highest_res, adata.obs.ann_level_2)]]

# sanity check (should be True):

len(lev2_unlabeled) == lev2_annotations.loc[~lev2_annotations.index.isin(lev2_labels_to_keep),"n_cells_w_only_l2_ann"].sum()

# now generate scANVI labels by pooling all the information obtained above into the adata column "scanvi_label":

adata.obs['scanvi_label'] = adata.obs.ann_level_3.tolist()
adata.obs.loc[lev1_labeled,'scanvi_label'] = adata.obs.loc[lev1_labeled, "ann_level_1"]
adata.obs.loc[lev1_unlabeled,'scanvi_label'] = "unlabeled"
adata.obs.loc[lev2_labeled,'scanvi_label'] = adata.obs.loc[lev2_labeled, "ann_level_2"]
adata.obs.loc[lev2_unlabeled,'scanvi_label'] = "unlabeled"

# plot to check:

sc.set_figure_params(figsize=(8,8))
sc.pl.umap(adata,color='scanvi_label',size=1)

# show where unlabeled cells are located in umap:

sc.set_figure_params(figsize=(5,5))
sc.pl.umap(adata,color='scanvi_label',size=1,groups=['unlabeled'])

# ### Subset to hvgs and create raw layers

# subset data to highly variable genes, and make sure we have raw counts as data.

adata = adata[:,adata.var.highly_variable].copy()

adata.shape

# adata currently has normalized counts, so we need to use the .layers['counts'] layer to set raw.X. Also set adata.X to the counts, since it is not fully clear to me which layer cANVI uses for analysis.

adata.X = adata.layers['counts']
adata.raw = adata
raw = adata.raw.to_adata()
raw.X = adata.layers['counts']
adata.raw = raw

# sanity check:

adata.raw.X[:5,:8].toarray()

adata.X[:5,:8].toarray()

adata.layers['counts'][:5,:8].toarray()

# write and load here, if switching from CPU to GPU node now:

# +
# adata.write(path_HLCA_unintegrated_prepped)

# +
# adata = sc.read(path_HLCA_unintegrated_prepped)
# -

# ### Set relevant anndata.obs labels and training parameters
#
# We use parameters as provided in scANVI tutorial.

# print datasets, these will be the "conditions" used for batch correction:

adata.obs.dataset.unique().tolist()

# +
condition_key = 'dataset'
cell_type_key = 'scanvi_label'
unlabeled_category = "unlabeled"

vae_epochs = 500
scanvi_epochs = 200

early_stopping_kwargs = {
    "early_stopping_metric": "elbo",
    "save_best_state_metric": "elbo",
    "patience": 10,
    "threshold": 0,
    "reduce_lr_on_plateau": True,
    "lr_patience": 8,
    "lr_factor": 0.1,
}
early_stopping_kwargs_scanvi = {
    "early_stopping_metric": "accuracy",
    "save_best_state_metric": "accuracy",
    "on": "full_dataset",
    "patience": 10,
    "threshold": 0.001,
    "reduce_lr_on_plateau": True,
    "lr_patience": 8,
    "lr_factor": 0.1,
}
# -

# ## Run scANVI:

sca.dataset.setup_anndata(adata, batch_key=condition_key, labels_key=cell_type_key)

# Default parameters, except n_latent=30 (higher than default) since we're processing large data with diverse cell types (was also used in benchmarking), and gene_likelihood='nb' instead of zinb since all data are UMI based (hence negative binomially distributed).

vae = sca.models.SCANVI(
    adata,
    unlabeled_category,
    n_layers=2,
    n_latent = 30, # to allow for capturing more heterogeneity
    encode_covariates=True,
    deeply_inject_covariates=False,
    use_layer_norm="both",
    use_batch_norm="none",
    gene_likelihood="nb", # because we have UMI data
    use_cuda=True #to use GPU
)

print("Labelled Indices: ", len(vae._labeled_indices))
print("Unlabelled Indices: ", len(vae._unlabeled_indices))

vae.train(
    n_epochs_unsupervised=vae_epochs,
    n_epochs_semisupervised=scanvi_epochs,
    unsupervised_trainer_kwargs=dict(early_stopping_kwargs=early_stopping_kwargs),
    semisupervised_trainer_kwargs=dict(metrics_to_monitor=["elbo", "accuracy"],
                                       early_stopping_kwargs=early_stopping_kwargs_scanvi),
    frequency=1
)

# store model:

model_dir = os.path.join(dir_out, "scanvi_model") # this is the directory name/path of the directory *to be created*

vae.save(model_dir, overwrite=False)

# generate anndata of integrated latent embedding:

reference_latent = sc.AnnData(vae.get_latent_representation())
reference_latent.obs.index = adata.obs.index

# and store embedding:

reference_latent.write(os.path.join(dir_out, "scANVI_embedding.h5ad"))

# Add embedding to full HLCA and store:

# +
#...
