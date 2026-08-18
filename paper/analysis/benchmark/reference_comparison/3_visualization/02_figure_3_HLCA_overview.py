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

# # HLCA Figure 3, Extended Data Figure 2-6

# In this notebook we'll generate the figures from figure 3 of the HLCA preprint, plus corresponding extended figures (2-8, except 7b).

# ### Import modules, set paths, load files:

# +
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys
import os
from scipy.stats import entropy
from collections import Counter
from collections import OrderedDict

sys.path.append("../../scripts/")
import reference_based_harmonizing
import sankey
# -

# for pretty code formatting (not needed to run code):

# %load_ext lab_black

# set figure parameters:

sc.set_figure_params(dpi=140, figsize=(8, 8))
sns.set_style("ticks")

# set paths:

path_HLCA = "../../data/HLCA_core_h5ads/HLCA_v2.h5ad"
path_celltype_anns_ordered = "../../supporting_files/celltype_structure_and_colors/manual_anns_and_leveled_anns_ordered.csv"  # contains (ordered) final annotations and matching colors
path_marker_genes = "../../results/markers/markergenes.csv"
dir_HLCA_subsets = "../../data/HLCA_core_h5ads/HLCA_subsets/"
dir_figures = "../../results/figures/"

# Load files:

adata = sc.read_h5ad(path_HLCA)  # HLCA
marker_genes = pd.read_csv(path_marker_genes, index_col=0)
ct_df = pd.read_csv(path_celltype_anns_ordered, index_col=0)  # cell type info
cts = ct_df.index.tolist()  # get biological order of cts

# Import atlas split into 3 subsets for figures (epithelial, immune, and endothelial + stromal). These objects also contain pre-computed umaps etc., which would take quite some time to re-compute.
subadatas = dict()
subadatas["epi"] = sc.read_h5ad(os.path.join(dir_HLCA_subsets, "HLCA_epithelial.h5ad"))
subadatas["imm"] = sc.read_h5ad(os.path.join(dir_HLCA_subsets, "HLCA_immune.h5ad"))
subadatas["str_end"] = sc.read_h5ad(
    os.path.join(dir_HLCA_subsets, "HLCA_endothelial_and_stromal.h5ad")
)
# clean up categoricals:
for subcomp in subadatas.keys():
    subadatas[subcomp].obs.manual_ann = subadatas[
        subcomp
    ].obs.manual_ann.cat.remove_unused_categories()
    # order biologically
    subadatas[subcomp].obs.manual_ann = subadatas[
        subcomp
    ].obs.manual_ann.cat.reorder_categories(
        [ct for ct in ct_df.index if ct in subadatas[subcomp].obs.manual_ann.values]
    )

# initiate empty dictionary for figures:

FIGURES = dict()

# generate color mapping for manual annotations:

ct_to_col = {ct: col for ct, col in zip(ct_df.index, ct_df.colors)}

# ## Generate figures:

# ### 3a HLCA umap, level 1 annotations:

FIGURES["3a_atlas_umap_ann_level_1"] = sc.pl.umap(
    adata,
    color="original_ann_level_1",
    groups=["Epithelial", "Endothelial", "Immune", "Stroma"],
    frameon=False,
    #     title=,
    return_fig=True,
    na_in_legend=False,
    size=3,
    #     legend_loc="on data",
    legend_fontsize=22,
    sort_order=False,
)

# ### 3b + 3d, extended data fig. 2, label + subject entropy and manual re-annotations:

# First generate umaps that specify which part of the HLCA we're subsetting to (they're shown in the top left corners of the large umaps in figure 3b):

FIGURES["3b_epithelial_subsetting"] = sc.pl.umap(
    adata,
    color="original_ann_level_1",
    groups=["Epithelial"],
    frameon=False,
    return_fig=True,
    na_in_legend=False,
    legend_fontsize=20,
    size=3,
    sort_order=False,
)

FIGURES["3b_immune_subsetting"] = sc.pl.umap(
    adata,
    color="original_ann_level_1",
    groups=["Immune"],
    frameon=False,
    return_fig=True,
    na_in_legend=False,
    legend_fontsize=20,
    size=3,
    sort_order=False,
)

FIGURES["3b_endothelial_stromal_subsetting"] = sc.pl.umap(
    adata,
    color="original_ann_level_1",
    groups=["Endothelial", "Stroma"],
    frameon=False,
    return_fig=True,
    na_in_legend=False,
    legend_fontsize=20,
    size=1.5,
    sort_order=False,
)

# For each subset, calculate and plot label entropy per cluster, subject entropy per cluster, and final annotations (different font sizes etc. for generation of paper figure):

for subcomp, subadata in subadatas.items():
    if subcomp == "epi":
        dotsize = 5
    elif subcomp == "imm":
        dotsize = 6.5
    elif subcomp == "str_end":
        dotsize = 8
    FIGURES[f"3b_{subcomp}_umap_label_entropy"] = sc.pl.umap(
        subadata,
        color="entropy_original_ann_level_3_clean_leiden_3",
        frameon=False,
        cmap="Reds",
        vmin=0,
        vmax=1.4,
        title=subcomp,
        return_fig=True,
        size=dotsize,
    )
    FIGURES[f"ED2_{subcomp}_umap_subj_entropy"] = sc.pl.umap(
        subadata,
        color="entropy_subject_ID_leiden_3",
        frameon=False,
        cmap="Reds",
        vmin=0,
        vmax=4,
        title=f"{subcomp} subject entropy",
        return_fig=True,
        size=dotsize,
    )
    FIGURES[f"3d_{subcomp}_umap_annotations_labels_on_top"] = sc.pl.umap(
        subadata,
        color="manual_ann",
        frameon=False,
        return_fig=True,
        legend_loc="on data",
        legend_fontsize=16,
        size=dotsize,
        palette=ct_to_col,
    )
    FIGURES[f"3d_{subcomp}_umap_annotations_labels_on_side"] = sc.pl.umap(
        subadata, color="manual_ann", frameon=False, return_fig=True, size=dotsize
    )
    FIGURES[f"3d_{subcomp}_umap_annotations_labels_only"] = sc.pl.umap(
        subadata,
        color="manual_ann",
        frameon=False,
        legend_loc="on data",
        legend_fontsize=9,
        size=0,
        return_fig=True,
        palette=ct_to_col,
    )

# In the paper, we distinguish between high and low label and subject entropy. We chose those cutoffs as sensibly as possible using the code below:

# Calculate cluster statistics (e.g. number of subjects per cluster, subject and label entropy per cluster, the final annotation of the cluster, number of cells per cluster):

cl_subj_df = adata.obs.groupby("leiden_3").agg(
    {
        "subject_ID": "nunique",
        "entropy_subject_ID_leiden_3": "first",
        "entropy_original_ann_level_3_clean_leiden_3": "first",
        "ann_level_1": "first",
        "ann_level_3": "first",
        "leiden_3": "count",
    }
)
cl_subj_df.rename(columns={"leiden_3": "ncells"}, inplace=True)

# Check total number of subjects, level 3 original annotations, and leiden_3 clusters:

nsubjects = adata.obs.subject_ID.nunique()
nlev3anns = adata.obs.original_ann_level_3_clean.nunique()
nclusters = adata.obs.leiden_3.nunique()

print(
    f"Number of subjects: {nsubjects}\nNumber of unique original level 3 annotations (after harmonization): {nlev3anns}\nNumber of clusters: {nclusters}"
)

# We'll set the threshold for low entropy using the following example case: 95% of cells coming from one subject, while the rest is equally divided over the other subjects. Other distributions resulting in the same or a lower entropy value will be considered to have low subject entropy.

p_subj = [0.95] + [0.05 / (nsubjects - 1)] * (nsubjects - 1)
subj_entr_cutoff = entropy(p_subj)
print(subj_entr_cutoff)

# Based on this threshold, we can calculate the number of clutsers with high subject entropy:

print(
    "Number of clusters with high subject entropy:",
    (cl_subj_df.entropy_subject_ID_leiden_3 > subj_entr_cutoff).sum(),
)

# Check the final level 1 and level 3 annotations of these clusters:

cl_subj_df.loc[
    cl_subj_df.entropy_subject_ID_leiden_3 < subj_entr_cutoff,
    ["ann_level_1", "ann_level_3"],
]

# Now we will determine the same type of threshold for cell-type label entropy. We'll consider label/annotation entropy low when 3/4 of the cluster has one label, while the rest has one other label (or other distributions resulting in the same or a lower entropy value)

p_ann = [0.75] + [0.25]  # / (nlev3anns - 1)] * (nlev3anns - 1)
ann_entr_cutoff = entropy(p_ann)
print(ann_entr_cutoff)

# Based on this threshold, we can check the number of clusters that have low label entropy:

print(
    "Number of clusters with low label entropy:",
    (cl_subj_df.entropy_original_ann_level_3_clean_leiden_3 < ann_entr_cutoff).sum(),
)

# Some more summary stats:

print("Median number of subjects per cluster:", cl_subj_df.subject_ID.median())
print("Min:", cl_subj_df.subject_ID.min(), "Max:", cl_subj_df.subject_ID.max())

# ## 3c: Re-annotation of cluster with high cell-type label entropy:

# Here we will focus on cluster 1.2.1, the immune cluster with highest cell-type label entropy, and show how it was originally labeled versus how it was finally re-annotated.

# Subset to cluster, set all cell type that are not DCs, monocytes or macrophages at level 3 to "other" to prevent over-crowding of the plot.

adata_121 = adata[adata.obs.leiden_3.values == "1.2.1", :].copy()
al3_to_filt = dict(
    zip(
        adata_121.obs.original_ann_level_3_clean.unique(),
        len(adata_121.obs.original_ann_level_3_clean.unique()) * ["Other"],
    )
)
for ct_to_keep in ["Dendritic cells", "Monocytes", "Macrophages"]:
    al3_to_filt[ct_to_keep] = ct_to_keep
adata_121.obs[
    "original_ann_level_3_clean_filt"
] = adata_121.obs.original_ann_level_3_clean.map(al3_to_filt)

# Plot sankey plot showing original labels versus final annotations for this cluster:

original_anns = adata_121.obs.loc[:, "original_ann_level_3_clean_filt"]
final_anns = adata_121.obs.loc[:, "manual_ann"]
fig, ax = plt.subplots(figsize=(8, 3))
sankey.sankey(
    x=original_anns,
    y=final_anns,
    title="",
    title_left="Original label (level 3)",
    title_right="Final annotation",
    ax=ax,
    fontsize="5",
    colors=ct_to_col,
    colorside="right",
    alpha=1,
)
FIGURES["3c_high_label_entropy_cluster_reannotation_cl121_sankey"] = fig
plt.show()
plt.close()

# ### Extended data figure 3: marker expression of DC2s from figure 3c (above)

# We will show marker expression of all cells with final annotation DC2. As many of these cells were originally labeled as macrophages and monocytes, we will also add cells finally annotated as monocytes and macrophages (from any cluster) for comparison.

# We include DC2, monoyte and macrophage markers:

markers = [
    "CD1C",
    "CLEC10A",
    "FCER1A",
    "FCN1",
    "S100A12",
    "CD300E",
    "MARCO",
    "APOE",
    "GPNMB",
]

# Store barcodes of monocytes, macrophages, and DC2s:

monocytes = adata.obs.index[adata.obs.ann_level_3 == "Monocytes"].tolist()
macrophages = adata.obs.index[adata.obs.ann_level_3 == "Macrophages"].tolist()
DC2s = adata.obs.index[adata.obs.ann_level_4 == "DC2"].tolist()

# split DC2s based on original label:

DC2s_l_DC = adata.obs.index[
    [
        x and y
        for x, y in zip(
            adata.obs.ann_level_4 == "DC2",
            adata.obs.original_ann_level_3 == "Dendritic cells",
        )
    ]
].tolist()
DC2s_l_mono = adata.obs.index[
    [
        x and y
        for x, y in zip(
            adata.obs.ann_level_4 == "DC2",
            adata.obs.original_ann_level_3 == "Monocytes",
        )
    ]
].tolist()
DC2s_l_macro = adata.obs.index[
    [
        x and y
        for x, y in zip(
            adata.obs.ann_level_4 == "DC2",
            adata.obs.original_ann_level_3 == "Macrophages",
        )
    ]
].tolist()
DC2s_l_other = adata.obs.index[
    [
        x and y
        for x, y in zip(
            adata.obs.ann_level_4 == "DC2",
            ~adata.obs.original_ann_level_3.isin(
                ["Monocytes", "Dendritic cells", "Macrophages"]
            ),
        )
    ]
].tolist()

# Check that we have all DC2s now:

len(DC2s_l_DC) + len(DC2s_l_mono) + len(DC2s_l_macro) + len(DC2s_l_other) == len(DC2s)

# Create anndata object with our 3 cell types of interest:

adata_myeloid = adata[monocytes + macrophages + DC2s, markers].copy()

# Create custom labels, combining level 3 and 4 annotations, and specifying original level 3 labels for DC2s:

# custom label
adata_myeloid.obs["Myeloid_custom_label"] = None
adata_myeloid.obs.loc[monocytes, "Myeloid_custom_label"] = "Monocytes"
adata_myeloid.obs.loc[macrophages, "Myeloid_custom_label"] = "Macrophages"
adata_myeloid.obs.loc[DC2s_l_DC, "Myeloid_custom_label"] = "DC2 (original label: DC)"
adata_myeloid.obs.loc[
    DC2s_l_mono, "Myeloid_custom_label"
] = "DC2 (original label: Monocytes)"
adata_myeloid.obs.loc[
    DC2s_l_macro, "Myeloid_custom_label"
] = "DC2 (original label: Macrophages)"
adata_myeloid.obs.loc[
    DC2s_l_other, "Myeloid_custom_label"
] = "DC2 (original label other)"

# Normalize myeloid data so that all genes are on the same scale, with the 99th expression percentile among all cells from the 3 cell types is set to 1:

perc_99 = np.percentile(adata_myeloid.X.toarray(), 99, axis=0)
adata_myeloid.X = adata_myeloid.X / perc_99

# Set fontsize for figure:

sc.set_figure_params(fontsize=12)

# Convert adata to pandas dataframe for heatmap plotting. Calculate one mean value per subject-celtype pair:

myeloid_df = pd.DataFrame(
    adata_myeloid.X,
    index=adata_myeloid.obs.Myeloid_custom_label,
    columns=adata_myeloid.var.index,
)
myeloid_df["subject_ID"] = adata_myeloid.obs.subject_ID.values
myeloid_df_per_ct_per_subject = myeloid_df.groupby(
    ["Myeloid_custom_label", "subject_ID"]
).agg({gene: "mean" for gene in markers})
# remove rows with nans
myeloid_df_per_ct_per_subject.dropna(axis=0, inplace=True)

yticks = []
yticks_unfiltered = myeloid_df_per_ct_per_subject.index.get_level_values(
    level=0
).tolist()
for t in yticks_unfiltered:
    if t in yticks:
        yticks.append("")
    else:
        yticks.append(t)
fig_heatmap = sns.clustermap(
    figsize=(5, 5),
    data=myeloid_df_per_ct_per_subject.values,
    row_cluster=False,
    col_cluster=False,
    vmin=0,
    vmax=1,
    cmap="Reds",
    yticklabels=yticks,
    xticklabels=myeloid_df_per_ct_per_subject.columns,
)
FIGURES[f"ED3_DC2s_mono_macro_per_subject_heatmap"] = fig_heatmap
plt.show()
plt.close()

# ## 3e: Comparison of original labels to final annotations:

# In this figure we indicate, for each final cell type annotation, to what extend it was originally correctly labeled, "under-labeled" (correct but at lower level of detail), or mis-labeled. The analysis for this was done in notebook 7: "7_manual_ann_ingestion_and_removal_of_doublets_etc".

# WE'll see mislabeled here instead of misannotated to distinguish original albels from final annotations.

adata.obs["reannotation_type_summ"] = adata.obs.reannotation_type.map(
    {
        "Misannotated": "Mislabeled",
        "Underannotated": "Underlabeled",
        "Correctly annotated": "Correctly labeled",
    }
)

# Calculate the number of cells for each annotation type, for each final annotation ("manual_ann"), and convert to percentages:

annotation_types_per_manann = pd.crosstab(
    adata.obs.manual_ann, adata.obs.reannotation_type_summ
)
annotation_types_per_manann = (
    annotation_types_per_manann.div(annotation_types_per_manann.sum(axis=1), axis=0)
    * 100
)

# Add row with overall numbers (i.e. numbers for all cells, not a single cell type):

overall_annotation_types = (
    adata.obs.reannotation_type_summ.value_counts() / adata.n_obs * 100
)

annotation_types_per_manann.index = annotation_types_per_manann.index.tolist()

annotation_types_per_manann.loc[
    "Overall", overall_annotation_types.index
] = overall_annotation_types.values

# Plot barplot:

sns.set_style("ticks")

fz = 8
with plt.rc_context(
    {
        "figure.figsize": (10, 2),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    ax = annotation_types_per_manann.loc[
        cts + ["Overall"], ["Correctly labeled", "Underlabeled", "Mislabeled"]
    ].plot(
        kind="bar",
        stacked=True,
        color=["limegreen", "steelblue", "red"],
        edgecolor="none",
    )
    ax.tick_params(which="minor")
    plt.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=(1.02, 0.57), fontsize=fz, frameon=False)
    plt.ylabel("% of cells", fontsize=fz)
    plt.xlabel("Annotation", fontsize=fz)
    plt.yticks(np.arange(0, 101, 20), fontsize=fz)
    FIGURES["3e_annotation_accuracy"] = ax.get_figure()
#     plt.show()

# Some stats mentioned in the paper:

print(
    "Number of annotations with majority correctly labeled:",
    (annotation_types_per_manann["Correctly labeled"] > 50).sum(),
)

print(
    "Number of annotations with majority mislabeled:",
    (annotation_types_per_manann["Mislabeled"] > 50).sum(),
)

print("Total number of cell types:", len(adata.obs.manual_ann.unique()))

# Percentages of cells with each label:

adata.obs.reannotation_type_summ.value_counts() / adata.n_obs * 100

# ## Extended Data Figure 5. Final annotations and original annotations of mislabeled cells

# We will now zoom in on the "mislabeled" cells from the figure above, and check how they were originally labeled versus what their final annoations were, to get an impression of how/why/where cells were mislabeled.

# Set figure params:

sc.set_figure_params(dpi=140, figsize=(8, 8))

# Extract final annotations of the mislabeled cells.

final_anns = adata.obs.loc[
    adata.obs.reannotation_type_summ == "Mislabeled", "manual_ann"
]

# Extract matching original labels at the finest level available (we'll use level 5 and remove any prefixes for forward-propagated, i.e. originally coarser labels, e.g. 3_DCs will be set to DCs).

original_anns_unf_uncl = adata.obs.original_ann_level_5
# remove prefixes:
original_anns_unf = [
    lab[2:] if lab[1] == "_" else lab for lab in original_anns_unf_uncl
]
# keep only misannotated cells:
original_anns = [
    lab
    for lab, anntype in zip(original_anns_unf, adata.obs.reannotation_type_summ)
    if anntype == "Mislabeled"
]

# Split by compartment, to make figures more legible. Extract compartment for all misannotated cells:

compartments = adata.obs.loc[
    adata.obs.reannotation_type_summ == "Mislabeled", "ann_level_1"
]

# Set a minimum percentage for the label to be included in the sankey. We cannot include all labels as this will over-crowd the plot. Hence we will set labels with an abundance lower than this percentage to "Other".

min_perc_cells_sankey_right = 1

# Extract final and original anns per compartment, and rename original labels according to limit above to "Other".

final_anns_per_comp = dict()
original_anns_per_comp = dict()
for comp in set(compartments):
    comp_final = [
        lab for lab, labcomp in zip(final_anns, compartments) if labcomp == comp
    ]
    comp_original = [
        lab for lab, labcomp in zip(original_anns, compartments) if labcomp == comp
    ]
    comp_original_freqs = Counter(comp_original)
    min_n_cells = min_perc_cells_sankey_right * len(comp_final) / 100
    groupcleaning = {
        group: (group if comp_original_freqs[group] >= min_n_cells else "Other")
        for group in set(comp_original)
    }
    comp_original = [groupcleaning[lab] for lab in comp_original]
    final_anns_per_comp[comp] = comp_final
    original_anns_per_comp[comp] = comp_original

# Plot Sankeys. Here we will put the final annotations left, and the original labels right.

for comp in set(compartments):
    if comp == "Stroma":
        title = f"Misannotations\n({comp.lower()}l cells)"
    else:
        title = f"Misannotations\n({comp.lower()} cells)"
    fig, ax = plt.subplots(figsize=(5, 6))
    sankey.sankey(
        x=final_anns_per_comp[comp],
        y=original_anns_per_comp[comp],
        title=title,
        title_left="Final annotation",
        title_right="Original label",
        ax=ax,
        fontsize="5",
        colors={ct: col for ct, col in zip(ct_df.index, ct_df.colors)},
        colorside="left",
        alpha=0.8,
    )
    plt.show()
    plt.close()
    FIGURES[f"ED_5_final_vs_original_ann_mislabled_cells_{comp}"] = fig

# ### Marker gene dotplot:

matplotlib.rcParams["patch.edgecolor"] = "black"

for comp in adata.obs.ann_level_1.unique():
    print(comp)
    # get markers in the right order, including compartmental/subcomparmental markers:
    cts_in_comp = [ct for ct in ct_df.index if ct_df.loc[ct, "Level_1"] == comp]
    # this will be the marker dictionary used for the figure
    marker_dict_comp = {}
    # loop through cts
    for ct in cts_in_comp:
        ct_markers = list(marker_genes.loc[:, f"{ct}_marker"].dropna())
        ct_markers_for = list(marker_genes.loc[:, f"{ct}_marker_for"].dropna())
        n_markers = len(ct_markers)
        # if fewer than 3 markers, all markers are specfific for the cell type
        if n_markers <= 3:
            marker_dict_comp[ct] = ct_markers
        else:
            # if more than 3 markers, the first three are for the compartment
            lev1_marker_for = list(set(ct_markers_for[:3]))
            if len(lev1_marker_for) > 1:
                raise ValueError(
                    "Something is wrong with the marker indexing. Exiting."
                )
            marker_dict_comp[lev1_marker_for[0]] = ct_markers[:3]
            # now isolate the remaining markers and what they are a marker for
            n_levels_remaining = len(set(ct_markers_for[3:]))
            # count number of markers for lev2
            # if only 1 level remaining, the markers after the first three are specific for the cell type
            if n_levels_remaining == 1:
                marker_dict_comp[ct] = ct_markers[3:]
            # if 2 levels remaining, the first ones (check how many) are for the
            # second-level compartment, and the remainder for the cell type
            else:
                lev2_marker_for = ct_markers_for[3]
                n_lev2_markers = sum(
                    [mf == lev2_marker_for for mf in ct_markers_for[3:]]
                )
                marker_dict_comp[lev2_marker_for] = ct_markers[3 : 3 + n_lev2_markers]
                marker_dict_comp[ct] = ct_markers[3 + n_lev2_markers :]
    # now order markers. We first want the markers for the compartment, then
    # for the subcompartments and their members (actual cell types)
    marker_order = [comp]
    lev2_cts = ct_df.loc[ct_df.Level_1 == comp, "Level_2"].values
    for lev2_ct in lev2_cts:
        if lev2_ct not in marker_order:
            if lev2_ct in marker_dict_comp.keys():
                marker_order.append(lev2_ct)
            lev2_sub_cts = [
                ct for ct, ct2 in zip(ct_df.index, ct_df.Level_2) if ct2 == lev2_ct
            ]
            marker_order += lev2_sub_cts
    marker_dict_comp_ordered = OrderedDict(
        {ct: marker_dict_comp[ct] for ct in marker_order}
    )
    # now plot
    fig = sc.pl.dotplot(
        adata[adata.obs.ann_level_1 == comp, :],
        var_names=marker_dict_comp_ordered,
        groupby="manual_ann",
        standard_scale="var",
        return_fig=True,
    )
    FIGURES[f"ED_6_ct_markers_{comp}"] = fig
    fig.show()

# ## 3f, Extended Data Figure 7a: Rare cell analysis

# First some basic stats on which datasets annotated their rare cells (i.e. neuro-endocrine, tuft cells, ionocytes). These stats are  mentioned in the paper.

rare_cells_per_ds = (
    adata.obs.groupby(["original_ann_level_4_clean", "dataset"])
    .agg({"original_ann_level_4_clean": "count"})
    .loc[["Neuroendocrine", "Ionocyte", "Tuft"], :]
).rename(
    columns={
        "original_ann_level_4_clean": "ncells_original",
    }
)
rare_cells_per_ds_final_ann = (
    adata.obs.groupby(["ann_level_4", "dataset"])
    .agg({"ann_level_4": "count"})
    .loc[["Neuroendocrine", "Ionocyte", "Tuft"], :]
).rename(
    columns={
        "ann_level_4": "ncells_final",
    }
)

rare_cells_per_ds["ncells_final"] = rare_cells_per_ds_final_ann.loc[
    rare_cells_per_ds.index, "ncells_final"
]

# Number of datasets that originally labeled specific rare cells:

(rare_cells_per_ds.loc[:, ["ncells_original"]] > 0).groupby(
    "original_ann_level_4_clean"
).agg({"ncells_original": "sum"})

# Number of datasets with finally annotated rare cells:

(rare_cells_per_ds.loc[:, ["ncells_final"]] > 0).groupby(
    "original_ann_level_4_clean"
).agg({"ncells_final": "sum"})

# Now calculate precision (faction of cells in cluster that are labeled as cell type under consideration) and recall (percentage of cells labeled as cell type under consideration that are captured in the cluster) of "rare cell clusters". Also discussed in the text.

rare_cell_clusters = ["0.7.0", "0.7.1", "0.7.2"]

cluster_sizes = (
    adata.obs.groupby("leiden_3")
    .agg({"leiden_3": "count"})
    .rename(columns={"leiden_3": "n_cells"})
)
cluster_perc = cluster_sizes / cluster_sizes.sum() * 100
cluster_perc.rename(columns={"n_cells": "perc"}, inplace=True)

rare_cells_cluster_ass = (
    adata.obs.groupby(["original_ann_level_4_clean", "leiden_3"])
    .agg({"leiden_3": "count"})
    .rename(columns={"leiden_3": "n_cells"})
)

# neuroendocrine
rare_cell_recall = (
    rare_cells_cluster_ass.loc["Neuroendocrine"]
    .rename(columns={"n_cells": "n_ne"})
    .sort_values(by="n_ne", ascending=False)
)
rare_cell_recall["recall_ne"] = round(
    rare_cell_recall.n_ne / rare_cell_recall.n_ne.sum(), 3
)
rare_cell_recall["prec_ne"] = round(
    rare_cell_recall.n_ne / cluster_sizes.loc[rare_cell_recall.index, "n_cells"], 3
)
# ionoctyes
rare_cell_recall["n_io"] = rare_cells_cluster_ass.loc["Ionocyte"].loc[
    rare_cell_recall.index
]
rare_cell_recall["recall_io"] = round(
    rare_cell_recall.n_io / rare_cell_recall.n_io.sum(), 3
)
rare_cell_recall["prec_io"] = round(
    rare_cell_recall.n_io / cluster_sizes.loc[rare_cell_recall.index, "n_cells"], 3
)
# brush/tuft
rare_cell_recall["n_tuft"] = rare_cells_cluster_ass.loc["Tuft"].loc[
    rare_cell_recall.index
]
rare_cell_recall["recall_tuft"] = round(
    rare_cell_recall.n_tuft / rare_cell_recall.n_tuft.sum(), 3
)
rare_cell_recall["prec_tuft"] = round(
    rare_cell_recall.n_tuft / cluster_sizes.loc[rare_cell_recall.index, "n_cells"], 3
)

# print percentage of total cells for rare cell clusters:

cluster_perc.loc[
    rare_cell_clusters,
]

# print rare cell recall for same clusters:

rare_cell_recall.loc[rare_cell_clusters]

# Now split the rare cell clusters into "false positives" (cells annotated as rare or a subtype thereof but not in our rare cell clusters), "false negatives" (cells not annotated as rare or a subtype thereof, but in our rare cell clusters anyway) and "true positives" (cells annotated as rare or a subtype thereof, and also present in one of our rare cell clusters).

rare_adata = adata[adata.obs.leiden_3.isin(rare_cell_clusters), :].copy()
rare_false_pos_adata = adata[
    (
        ~adata.obs.leiden_3.isin(rare_cell_clusters)
        & adata.obs.original_ann_level_3.isin(["Rare"])
    ),
    :,
].copy()
rare_true_pos_adata = adata[
    (
        adata.obs.leiden_3.isin(rare_cell_clusters)
        & adata.obs.original_ann_level_3.isin(["Rare"])
    ),
    :,
].copy()
rare_false_neg_adata = adata[
    (
        adata.obs.leiden_3.isin(rare_cell_clusters)
        & ~adata.obs.original_ann_level_3.isin(["Rare"])
    ),
    :,
].copy()
# rename 3_Rare to Rare for figures
for ad in [rare_false_pos_adata, rare_true_pos_adata, rare_false_neg_adata]:
    ann_lev_4_mapper = {ct: ct for ct in ad.obs.original_ann_level_4}
    ann_lev_4_mapper["3_Rare"] = "Rare"
    ad.obs.original_ann_level_4 = ad.obs.original_ann_level_4.map(ann_lev_4_mapper)

# sanity check (should be true)

rare_adata.n_obs == rare_false_neg_adata.n_obs + rare_true_pos_adata.n_obs

# Set figure params:

sc.set_figure_params(dpi=140, figsize=(8, 8), fontsize=12)
sns.set_style("ticks")
matplotlib.rcParams["patch.edgecolor"] = "black"

# Show overall marker expression (not shown in paper):

sc.pl.dotplot(
    rare_adata,
    var_names=[
        "FOXI1",
        "CFTR",
        "LRMP",
        "ASCL2",
        "CALCA",
        "CHGA",
    ],
    groupby="leiden_3",
    var_group_labels=[
        "Ionocyte\nmarkers",
        "Tuft\nmarkers",
        "Neuroendocrine\nmarkers",
        "Immune",
        "Stroma",
        "Endothelial",
    ],
    var_group_positions=[(0, 1), (2, 3), (4, 5)],
    show=True,
    size_title="Positive cells\nin group (%)",
)

# Now plot dotplots per group:

FIGURES["ED7a_rare_cell_dotplot_false_positives"] = sc.pl.dotplot(
    rare_false_pos_adata,
    var_names=[
        "FOXI1",
        "CFTR",
        "LRMP",
        "ASCL2",
        "CALCA",
        "CHGA",
        "CD44",
        "LYZ",
        "SCGB1A1",
        "C20orf85",
        "KRT17",
    ],
    groupby="original_ann_level_4",
    categories_order=["Rare", "Ionocyte", "Tuft", "Neuroendocrine"],
    var_group_labels=[
        "Ionocyte",
        "Tuft",
        "Neuroendocrine",
        "Stromal",
        "SMG",
        "Secretory",
        "Cilliated",
        "Basal",
    ],
    var_group_positions=[
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
    ],
    return_fig=True,
    show=True,
    size_title="Positive cells\nin group (%)",
    vmax=2.2,
)

FIGURES["ED7a_rare_cell_dotplot_false_positives"].show()

FIGURES["ED7a_rare_cell_dotplot_false_negatives"] = sc.pl.dotplot(
    rare_false_neg_adata,
    var_names=[
        "FOXI1",
        "CFTR",
        "LRMP",
        "ASCL2",
        "CALCA",
        "CHGA",
        "CD44",
        "LYZ",
        "SCGB1A1",
        "C20orf85",
        "KRT17",
    ],
    groupby="leiden_3",
    var_group_labels=[
        "Ionocyte",
        "Tuft",
        "Neuroendocrine",
        "Stromal",
        "SMG",
        "Secretory",
        "Cilliated",
        "Basal",
    ],
    var_group_positions=[
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
    ],
    return_fig=True,
    show=True,
    size_title="Positive cells\nin group (%)",
    vmax=2.2,
)

FIGURES["ED7a_rare_cell_dotplot_false_negatives"].show()

FIGURES["ED7a_rare_cell_dotplot_true_positives"] = sc.pl.dotplot(
    rare_true_pos_adata,
    var_names=[
        "FOXI1",
        "CFTR",
        "LRMP",
        "ASCL2",
        "CALCA",
        "CHGA",
        "CD44",
        "LYZ",
        "SCGB1A1",
        "C20orf85",
        "KRT17",
    ],
    groupby="leiden_3",
    var_group_labels=[
        "Ionocyte",
        "Tuft",
        "Neuroendocrine",
        "Stromal",
        "SMG",
        "Secretory",
        "Cilliated",
        "Basal",
    ],
    var_group_positions=[
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
    ],
    return_fig=True,
    show=True,
    size_title="Positive cells\nin group (%)",
    vmax=2.2,
)

FIGURES["ED7a_rare_cell_dotplot_true_positives"].show()

# Now prepare the umaps from figure 3f:

# We'll clean up labels, setting all cell types with fewer than min_perc% of cells among the rare cell clusters to "Other", for legibility of figure.

min_perc = 2  # minimum percentage of cells with label for label to be kept

original_annotation_counts = rare_adata.obs.original_ann_level_4.value_counts()

original_annotations_to_keep = [
    ct
    for ct, count in zip(
        original_annotation_counts.index, original_annotation_counts.values
    )
    if count / rare_adata.n_obs * 100 > min_perc
]

rare_ann_mapping = {ct: "Other" for ct in original_annotation_counts.index}
for ct in original_annotations_to_keep:
    if ct[1] == "_":
        ct_new = ct[2:]
    else:
        ct_new = ct
    rare_ann_mapping[ct] = ct_new
rare_adata.obs[
    "original_ann_level_4_clean_fig"
] = rare_adata.obs.original_ann_level_4.map(rare_ann_mapping)

# Check which labels are not set to "Other":

original_annotations_to_keep

# Re-calculate neighbor graph and umap (make sure to use integrated embedding as basis for graph):

sc.pp.neighbors(rare_adata, n_neighbors=15, use_rep="X_scanvi_emb")

sc.tl.umap(rare_adata)

markers = [
    "FOXI1",
    "LRMP",
    "CALCA",
]

# set figure parameters:

sc.set_figure_params(dpi=140, figsize=(3, 3.5))
sns.set_style("ticks")

FIGURES["3f_rare_cell_umaps"] = sc.pl.umap(
    rare_adata,
    color=["ann_level_4", "original_ann_level_4_clean_fig", "study"] + markers,
    ncols=3,
    sort_order=False,
    wspace=0.8,
    vmax="p99",
    frameon=False,
    size=50,
    title=["Final annotation", "Original label", "Study"] + markers,
    return_fig=True,
)

# ## 3g, Extended Data figure 8a,b, Migratory DC analysis:

# Here we'll take a look at migratory DCs, a relatively rare cell type that was not annotated in most of the datasets, but which we detected using the integration in many datasets. Here we'll show marker expression to confirm that they are actually migratory DCs.

# subset adata to DCs (cluster 1.2.1) and re-embed:

dc_adata = adata[adata.obs.leiden_3.isin(["1.2.1"]), :].copy()

# Now extract mDC clusters from larger leiden_3 cluster, and re-embed:

mdc_adata = adata[adata.obs.leiden_5.isin(["1.2.1.2.1", "1.2.1.2.2"]), :].copy()

# check how cells from mdc clusters were annotated, and how specific those annotations were:

mdc_per_study_ann = pd.crosstab(mdc_adata.obs.original_ann_level_4, mdc_adata.obs.study)
dc_per_study_ann = pd.crosstab(dc_adata.obs.original_ann_level_4, dc_adata.obs.study)

mdc_per_study_ann

# perc. of mdcs in atlas:

mdc_adata.n_obs / adata.n_obs * 100

# number of  mdcs per study:

mdc_n_by_study = mdc_adata.obs.study.value_counts()

mdc_n_by_study

# show marker expression in mdcs compared to other dcs. Label by cell type + study, and number of migratory dcs for mdc labels.

dc_adata.obs["mdc"] = "Other DC"
dc_adata.obs.loc[dc_adata.obs.index.isin(mdc_adata.obs.index), "mdc"] = "Migratory DC"
dc_adata.obs["mdc_by_study"] = [
    f"{mdc} ({study}, n={mdc_n_by_study[study]})"
    if mdc == "Migratory DC"
    else f"{mdc} ({study}"
    for mdc, study in zip(dc_adata.obs.mdc, dc_adata.obs.study)
]
dc_adata.obs["study_n_mdcs"] = [
    f"{study} ({mdc_n_by_study[study]})" for study in dc_adata.obs.study
]

# Set colors and figure parameters:

custom_palette_1 = {"Migratory DC": "maroon", "Other DC": "grey"}

sc.set_figure_params(figsize=(4, 3), frameon=False, transparent=True, fontsize=20)
sns.set_style("ticks")

# Now plot ccr7 expression. Note that this figure is automatically saved in a "figures" folder in your current directory (couldn't figure out how to change that...)

fig, ax = plt.subplots()
sc.pl.violin(
    dc_adata, keys=["CCR7"], groupby="mdc", vmax=2.2, ax=ax, palette=custom_palette_1
)
FIGURES["3g_migratory_DCs_CCR7_violin"] = fig
fig.show()

# Clean up mdc_by_study labels to look pretty (starting Migratory DC labels with a space, so that they're plotted next to one another):

mdc_by_study_relabeling = {
    old_name: old_name.replace("Migratory DC", "")
    .replace("Other DC ", "")
    .replace("(", "")
    .replace(")", "")
    for old_name in dc_adata.obs.mdc_by_study.unique()
}

dc_adata.obs["mdc_by_study_pretty_labels"] = dc_adata.obs.mdc_by_study.map(
    mdc_by_study_relabeling
)

# set colors to red for all mdcs, and to grey for all other dcs. Set figure parameters.

custom_palette_2 = {
    (study): ("maroon" if study.startswith(" ") else "grey")
    for study in dc_adata.obs.mdc_by_study_pretty_labels.unique()
}

sc.set_figure_params(figsize=(8, 3), frameon=False, transparent=True, fontsize=14)
sns.set_style("ticks")

# Now generate violin plot showing marker expression split by study. Note that this figure is automatically saved in a "figures" folder in your current directory (couldn't figure out how to change that...)

fig, ax = plt.subplots()
sc.pl.stacked_violin(
    dc_adata,
    var_names=[
        "CCR7",
        "LAD1",
        "CCL19",
    ],  # "CCL22", "BIRC3"], # other markers work but couldn't find references
    groupby="mdc_by_study_pretty_labels",
    ax=ax,
    swap_axes=True,
    # palette=custom_palette_2,
    rotation=90,
    standard_scale="var",
    colorbar_title="Median norm.\nexpr. in group",
)
fig.show()

markers_to_plot = [
    "CCR7",
    "LAD1",
    "CCL19",
]
mig_dc_plot_df = pd.DataFrame(dc_adata.obs.loc[:, ["mdc", "study_n_mdcs"]])
for gene in markers_to_plot:
    mig_dc_plot_df[gene] = dc_adata[:, gene].X.toarray()
fz = 12
fig = plt.figure()
figcount = 1
axes = dict()
with plt.rc_context(
    {
        "figure.figsize": (12, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    for gene in markers_to_plot:
        if figcount == 1:
            axes[1] = fig.add_subplot(3, 1, 1)
        else:
            axes[figcount] = fig.add_subplot(3, 1, figcount, sharex=axes[1])
        sns.violinplot(
            x="study_n_mdcs",
            y=gene,
            hue="mdc",
            data=mig_dc_plot_df,
            split=True,
            ax=axes[figcount],
            scale="width",
            palette=custom_palette_1,
            hue_order=["Other DC", "Migratory DC"],
        )
        if figcount == 3:
            plt.tick_params(axis="x", rotation=90)
            plt.xlabel("Study (n migratory DCs)")
            leg = plt.legend(loc=(1.01, 0.4), frameon=False, title="")
            leg._legend_box.align = "left"
        else:
            plt.xlabel("")
            plt.tick_params(axis="x", labelbottom=False)
            axes[figcount].get_legend().remove()
        figcount += 1
FIGURES["ED8b_migratory_DCs_by_study_marker_violin"] = fig

# Now show umap of all DCs:

sc.pp.neighbors(dc_adata, use_rep="X_scanvi_emb", n_neighbors=30)
sc.tl.umap(dc_adata)

sc.set_figure_params(figsize=(8, 8), frameon=False, transparent=True, fontsize=30)
sns.set_style("ticks")

FIGURES["ED_8a_migratory_DC_umaps"] = sc.pl.umap(
    dc_adata,
    color=["manual_ann", "CCR7", "LAD1", "CCL19"],
    vmax="p99",
    size=60,
    frameon=False,
    sort_order=False,
    wspace=0.8,
    return_fig=True,
)

# ## Novel cell types, marker expression:

sc.set_figure_params(dpi=140, figsize=(8, 8))
sns.set_style("ticks")
matplotlib.rcParams["patch.edgecolor"] = "black"

novel_cts_epithelial = {
    "Hillock-like": ["KRT13", "KRT14", "LY6D", "KRT6A"],
    "AT0 + pre-TB secretory": ["SFTPB", "SCGB3A2"],
    "AT0": ["SFTPC"],
    "pre-TB secretory": ["SCGB3A1"],
}

fig = sc.pl.dotplot(
    subadatas["epi"],
    groupby="manual_ann",
    var_names=novel_cts_epithelial,
    standard_scale="var",
    return_fig=True,
)
FIGURES["ED_8e_novel_ct_epithelial"] = fig
fig.show()

# show consistency across studies (figures not included in paper but mentioned in text):

min_n_cells_per_study = 5

for ct, ct_markers in novel_cts_epithelial.items():
    if ct == "AT0 + pre-TB secretory":
        cts_to_include = ["AT0", "pre-TB secretory"]
    else:
        cts_to_include = [ct]
    ct_adata = adata[adata.obs.manual_ann.isin(cts_to_include), ct_markers].copy()
    cells_per_study = ct_adata.obs.study.value_counts()
    studies_to_keep = cells_per_study.loc[
        cells_per_study > min_n_cells_per_study
    ].index.tolist()
    ct_adata_filt = ct_adata[ct_adata.obs.study.isin(studies_to_keep), :].copy()
    ct_adata_filt.obs.study.cat.remove_unused_categories(inplace=True)
    study_renamer = {st: f"{st} (n={cells_per_study[st]})" for st in studies_to_keep}
    ct_adata_filt.obs["study (n cells)"] = ct_adata_filt.obs.study.map(study_renamer)
    #     ct_marker_df = pd.DataFrame(ct_adata.X.toarray(), index=ct_adata.obs, columns=ct_markers)
    #     ct_marker_df['study'] = ct_adata.obs.study
    sc.pl.stacked_violin(
        ct_adata_filt, var_names=ct_markers, groupby="study (n cells)", title=ct
    )

novel_cts_immune = {
    "Hematopoietic stem cells": ["STMN1", "PRSS57", "CD34"],
    "Migratory DCs": ["CCR7", "LAD1", "CCL19"],
}

fig = sc.pl.dotplot(
    subadatas["imm"],
    groupby="manual_ann",
    var_names=novel_cts_immune,
    standard_scale="var",
    return_fig=True,
)
FIGURES["ED_8c_novel_ct_immune"] = fig
fig.show()

for ct, ct_markers in novel_cts_immune.items():
    cts_to_include = [ct]
    ct_adata = adata[adata.obs.manual_ann.isin(cts_to_include), ct_markers].copy()
    cells_per_study = ct_adata.obs.study.value_counts()
    studies_to_keep = cells_per_study.loc[
        cells_per_study > min_n_cells_per_study
    ].index.tolist()
    ct_adata_filt = ct_adata[ct_adata.obs.study.isin(studies_to_keep), :].copy()
    ct_adata_filt.obs.study.cat.remove_unused_categories(inplace=True)
    study_renamer = {st: f"{st} (n={cells_per_study[st]})" for st in studies_to_keep}
    ct_adata_filt.obs["study (n cells)"] = ct_adata_filt.obs.study.map(study_renamer)
    #     ct_marker_df = pd.DataFrame(ct_adata.X.toarray(), index=ct_adata.obs, columns=ct_markers)
    #     ct_marker_df['study'] = ct_adata.obs.study
    sc.pl.stacked_violin(
        ct_adata_filt, var_names=ct_markers, groupby="study (n cells)", title=ct
    )

novel_cts_stroma = {
    "Smooth muscle FAM83D+": [
        "MYH11",  # smooth muscle
        "CNN1",  # smooth muscle
        "PLN",  # smooth muscle
        "FAM83D",  # also includes a newly found marker now: FAM83D
    ]
}

subadata_stroma = adata[adata.obs.ann_level_1 == "Stroma", :].copy()

fig = sc.pl.dotplot(
    subadata_stroma,
    groupby="manual_ann",
    var_names=novel_cts_stroma,
    standard_scale="var",
    return_fig=True,
)
FIGURES["ED_8d_novel_ct_stroma"] = fig
fig.show()

# create obs variable with labels as in figure (SM FAM83D+ split by study)

stroma_study_ct_counts = pd.crosstab(
    subadata_stroma.obs.study, subadata_stroma.obs.manual_ann
)
subadata_stroma.obs["manual_ann_split_fam83d"] = [
    f"({study}, n={stroma_study_ct_counts.loc[study,ct]}) SM FAM83D+"
    if ct == "Smooth muscle FAM83D+"
    else ct
    for ct, study in zip(subadata_stroma.obs.manual_ann, subadata_stroma.obs.study)
]

# color sm fam83+ red, others grey:

colors_manual_ann_split_fam83d = {
    (ct): ("maroon" if "FAM83D" in ct else "lightgrey")
    for ct in sorted(subadata_stroma.obs.manual_ann_split_fam83d.unique())
}

# exclude FAM83+ SM cells from studies with fewer than min_n_smfam83d_cells smooth muscle FAM83D+ cells from figure:

min_n_smfam83d_cells = 3
studies_with_fewer_than_n_fam83_cells = [
    st
    for st in stroma_study_ct_counts.index
    if stroma_study_ct_counts.loc[st, "Smooth muscle FAM83D+"] < min_n_smfam83d_cells
]
cells_to_include = [
    idx
    for idx, ct, stu in zip(
        subadata_stroma.obs.index,
        subadata_stroma.obs.manual_ann,
        subadata_stroma.obs.study,
    )
    if (
        ct != "Smooth muscle FAM83D+"
        or stu not in studies_with_fewer_than_n_fam83_cells
    )
]

# plot:

FIGURES["ED_8f_novel_ct_fam83_per_study"], ax = plt.subplots(figsize=(8, 3))
sc.pl.violin(
    subadata_stroma[cells_to_include, :],
    keys="FAM83D",
    groupby="manual_ann_split_fam83d",
    rotation=90,
    palette=colors_manual_ann_split_fam83d,
    ax=ax,
    xlabel="(For SM FAM83+ cells: study, n cells) Annotation",
)

for ct, ct_markers in novel_cts_stroma.items():
    cts_to_include = [ct]
    ct_adata = adata[adata.obs.manual_ann.isin(cts_to_include), ct_markers].copy()
    cells_per_study = ct_adata.obs.study.value_counts()
    studies_to_keep = cells_per_study.loc[
        cells_per_study > min_n_cells_per_study
    ].index.tolist()
    ct_adata_filt = ct_adata[ct_adata.obs.study.isin(studies_to_keep), :].copy()
    ct_adata_filt.obs.study.cat.remove_unused_categories(inplace=True)
    study_renamer = {st: f"{st} (n={cells_per_study[st]})" for st in studies_to_keep}
    ct_adata_filt.obs["study (n cells)"] = ct_adata_filt.obs.study.map(study_renamer)
    #     ct_marker_df = pd.DataFrame(ct_adata.X.toarray(), index=ct_adata.obs, columns=ct_markers)
    #     ct_marker_df['study'] = ct_adata.obs.study
    sc.pl.stacked_violin(
        ct_adata_filt, var_names=ct_markers, groupby="study (n cells)", title=ct
    )

# ## Fraction of proliferating cells per cell type:

# Merge seperate annotations for proliferating cells into other cell types for this figure:

ct_prolif_mapping = {ct: ct for ct in adata.obs.manual_ann.unique()}
ct_prolif_mapping["T cells proliferating"] = "T cell lineage"
ct_prolif_mapping[
    "CD8 T cells"
] = "T cell lineage"  # as we don't know to which of the two the prolif. T cells belong
ct_prolif_mapping["CD4 T cells"] = "T cell lineage"
ct_prolif_mapping["Alveolar Mph proliferating"] = "Alveolar macrophages"
ct_prolif_mapping["AT2 proliferating"] = "AT2"
ct_prolif_mapping["Lymphatic EC mature"] = "Lymphatic EC"
ct_prolif_mapping["Lymphatic EC differentiating"] = "Lymphatic EC"
ct_prolif_mapping["Lymphatic EC proliferating"] = "Lymphatic EC"

adata.obs["manual_ann_prolif_merged"] = adata.obs.manual_ann.map(ct_prolif_mapping)

# order categories in biological order:

ct_prolif_order_df = pd.Series(index=adata.obs.manual_ann_prolif_merged.unique())
for ct in ct_prolif_order_df.index:
    for row in range(0, ct_df.shape[0]):
        if ct in ct_df.iloc[row, :].values:
            ct_prolif_order_df.loc[ct] = row
            continue
ct_prolif_order = ct_prolif_order_df.sort_values().index.tolist()

adata.obs.manual_ann_prolif_merged = pd.Categorical(
    adata.obs.manual_ann_prolif_merged, categories=ct_prolif_order, ordered=False
)

# Now calculate percentage of positive cells per ct and mean exp for MKI67:

adata.obs["MKI67_positive"] = adata[:, "MKI67"].X.toarray() > 0
adata.obs["MKI67"] = adata[:, "MKI67"].X.toarray()

mki67_ct_percentage = (
    adata.obs.groupby("manual_ann_prolif_merged").agg({"MKI67_positive": "mean"}) * 100
)

mki67_ct_mean_exp = (
    adata.obs.groupby("manual_ann_prolif_merged").agg({"MKI67": "mean"}) * 100
)

# and plot:

sns.set_style("ticks")

fz = 10
with plt.rc_context(
    {
        "figure.figsize": (12, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    fig, ax = plt.subplots()
    ax.bar(x=mki67_ct_percentage.index, height=mki67_ct_percentage.MKI67_positive)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("% of cells MKI67+")
    ax.set_xlabel("Annotation")
    FIGURES["ED_4c_proliferating_cells"] = fig

# Plot mean MKI67 count (we did not include this in the paper)

fz = 10
with plt.rc_context(
    {
        "figure.figsize": (12, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    fig, ax = plt.subplots()
    ax.bar(x=mki67_ct_mean_exp.index, height=mki67_ct_mean_exp.MKI67)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("MKI67 mean count")
    ax.set_xlabel("Annotation")

# ## Extra (not included in paper but nice anyway):Study and subject composition per cell type:

# Calculate percentage of cells from each study, for each cell type:

study_count_per_ct = pd.crosstab(adata.obs.manual_ann, adata.obs.study)
study_perc_per_ct = (
    study_count_per_ct.divide(study_count_per_ct.sum(axis=1), axis=0) * 100
)
anatomical_loc_count_per_ct = pd.crosstab(
    adata.obs.manual_ann, adata.obs.anatomical_region_level_1
)
anatomical_loc_perc_per_ct = (
    anatomical_loc_count_per_ct.divide(anatomical_loc_count_per_ct.sum(axis=1), axis=0)
    * 100
)

fz = 12
with plt.rc_context(
    {
        "figure.figsize": (14, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    fig, ax = plt.subplots()
    study_perc_per_ct.plot(
        kind="bar", stacked=True, ax=ax, cmap="tab20", edgecolor="none"
    )
    leg = plt.legend(
        loc=(1.01, -0.1),
        frameon=False,
        title="Study:",
        fontsize=fz,
    )
    plt.ylabel("Percentage of cells")
    plt.xlabel("Annotation")
    leg._legend_box.align = "left"
    FIGURES["ED_4a_study_composition_per_ct"] = fig

fz = 12
with plt.rc_context(
    {
        "figure.figsize": (14, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    fig, ax = plt.subplots()
    anatomical_loc_perc_per_ct.plot(kind="bar", stacked=True, ax=ax, edgecolor="none")
    #     leg = plt.legend(
    #         loc=(1.01, 0.1), frameon=False, title=, fontsize=fz
    #     )
    handles, labels = ax.get_legend_handles_labels()
    labels = [lab.capitalize() for lab in labels]
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc=(1.01, 0.3),
        fontsize=fz,
        frameon=False,
        title="Sample anatomical location:",
    )
    plt.ylabel("Percentage of cells")
    plt.xlabel("Annotation")
    leg._legend_box.align = "left"
    FIGURES["ED_4b_anatomical_location_composition_per_ct"] = fig

# For subjects, do the same. But as we cannot color each subject with a visibly distinct color (>100 subjects), we will rank them per cell type and color by rank. The most common subject in a single cell type will therefore always be named "1", etc.. Subject 1 for cell type A will therefore be different from subject 1 from cell type B. This plot will give you an impression of subject "diversity" per cell type.

# calculate percentages contributed per subject for every cell type
subjects_per_ct = pd.crosstab(adata.obs.manual_ann, adata.obs.subject_ID)
subjects_per_ct_norm = subjects_per_ct.divide(subjects_per_ct.sum(axis=1), axis=0) * 100

# create dataframe in which we will store the top 10 subject percentages
subjects_per_ct_top10subjects = pd.DataFrame(
    index=subjects_per_ct_norm.index, columns=range(1, 11)
)
# fill in dataframe with top 10 percentages
for ct in subjects_per_ct_top10subjects.index:
    subjects_per_ct_top10subjects.loc[ct,] = (
        subjects_per_ct_norm.loc[
            ct,
        ]
        .sort_values(ascending=False)[:10]
        .values
    )
# add a final column with "other subjects"
subjects_per_ct_top10subjects["other"] = 100 - subjects_per_ct_top10subjects.sum(axis=1)

fz = 8
with plt.rc_context(
    {
        "figure.figsize": (12, 3),
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "axes.labelsize": fz,
        "font.size": fz,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
):
    fig, ax = plt.subplots()
    subjects_per_ct_top10subjects.plot(
        kind="bar", stacked=True, ax=ax, edgecolor="none"
    )
    plt.grid(False)
    plt.ylabel("% of cells")
    plt.xlabel("cell type")
    leg = plt.legend(loc=(1.01, 0.08), frameon=False, title="Subject rank:")
    leg._legend_box.align = "left"
    plt.show()

# # Store figures:

# Store:

matplotlib.rcParams["patch.edgecolor"] = "black"

for figname, fig in FIGURES.items():
    print("Storing", figname)
    fig.savefig(
        os.path.join(dir_figures, f"{figname}.png"),
        bbox_inches="tight",
        dpi=140,
        transparent=True,
    )
    plt.close()

# Or store a single figure:

# +
# figname = "ED6c_migratory_DCs_by_study_marker_violin"
# FIGURES[figname].savefig(
#     os.path.join(dir_figures, f"{figname}.png"),
#     bbox_inches="tight",
#     dpi=140,
#     transparent=True,
# )
# plt.close()
# -
