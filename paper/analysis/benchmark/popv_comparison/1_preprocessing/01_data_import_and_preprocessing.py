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

# ## HLCA PREPROCESSING
# This notebook reads in the raw data included in the HLCA core, then adds sample and celltype annotations, harmonizes those, and performs minimal filtering

# optional extension, for automatic pretty-formatting of code:

# %load_ext autoreload
# %autoreload 2

# %load_ext lab_black

# filter out warnings if wanted:

# +
import warnings

warnings.filterwarnings("ignore")
# -

# ### Import modules

# +
import numpy as np
import pandas as pd
import scanpy as sc
import sys

# plotting modules
import matplotlib.pyplot as plt

# self-written modules:
sys.path.append("../../scripts/")
import utils
import LCA_file_reading
import preprocessing
import reference_based_harmonizing
# -

# ### Set paths:

# raw data dir
with open("../../../raw_data_path.txt", "r") as file:
    dir_data = file.readline().strip()
# gene id to symbol mapping file:
path_ens_mapper = (
    "../../supporting_files/gene_info/Homo_sapiens_GRCh38_84_gene_ids_to_gene_symbol.csv"
)
# raw HLCA:
path_raw_HLCA = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_RAW.h5ad"
# subject-filtered sample-annotated HLCA:
path_subj_filt_samp_ann_HLCA = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_RAW_subjfilt_ann.h5ad"
# fully filtered sample-annotated and celltype annotated HLCA
path_filt_ann_HLCA = "../../data/HLCA_core_h5ads/HLCA_v1_intermediates/LCA_Bano_Barb_Jain_Kras_Lafy_Meye_Mish_MishBud_Nawi_Seib_Teic_RAW_filt_ann.h5ad"
# original to harmonized annotation mapping:
path_celltype_mapping = "../../supporting_files/metadata_harmonization/HLCA_cell_type_reference_mapping_20220712.csv"
# original to harmonized anatomical location mapping:
path_anatomical_loc_mapping = "../../supporting_files/metadata_harmonization/HLCA_anatomical_region_reference_mapping_20210521.csv"
# dir to sample level metadata:
dir_sample_metadata = "../../supporting_files/sample_level_metadata"

# #### version info

sc.logging.print_versions()

# #### import data:

studies = [
    "Banovich_Kropski_2020",
    "Barbry_Leroy_2020",
    "Jain_Misharin_2021",
    "Krasnow_2020",
    "Lafyatis_Rojas_2019",
    "Meyer_2019",
    "Misharin_2021",
    "Misharin_Budinger_2018",
    "Nawijn_2021",
    "Seibold_2020",
    "Teichmann_Meyer_2019",
]

studies_with_non_integer_values = list()
adatas = dict()
for study in studies:
    print("\nSTUDY:", study)
    project_dir = f"{dir_data}{study}/"
    # note that the two **args are path to project dir and verbose
    adata = getattr(LCA_file_reading, f"read_file_{study}")(project_dir, True)
    # check if count matrices have only integer values:
    print("checking if counts are integers...")
    test = np.sum(adata.X.toarray() % 1 == 0, axis=1)
    nonint_adata = adata[test != adata.shape[1], :].copy()
    nonint_adata.shape
    if nonint_adata.shape[0] != 0:
        print("WARNING: THIS DATASET HAS NON-INTEGER VALUES!!!")
        studies_with_non_integer_values.append(study)
    else:
        print("counts are integers.")
        adatas[study] = adata
del adata, test, nonint_adata, study

if len(studies_with_non_integer_values) != 0:
    print("WARNING: THERE WERE DATASETS WITH NON-INTEGER VALUES!")
    print(studies_with_non_integer_values)

# Now pool between datasets:

adata = sc.AnnData.concatenate(
    *adatas.values(),
    join="outer",
    batch_key=None,
    batch_categories=list(adatas.keys()),
    index_unique=None
)
print(adata.shape)

# remove gene symbol columns in adata.var (we will translate ids to symbols using ensembl 84 gtf file):

adata.var.drop(columns=adata.var.columns, inplace=True)

gene_id_to_symbol_mapper = pd.read_csv(path_ens_mapper, index_col=0)
# turn into dict:
gene_id_to_symbol_mapper = dict(
    zip(gene_id_to_symbol_mapper.index, gene_id_to_symbol_mapper.gene_name)
)

adata.var["gene_symbols"] = adata.var.index.map(gene_id_to_symbol_mapper)

# set NANs in adata.X to zero, and shuffle rows (for unbiased plotting etc.):

adata.X = np.nan_to_num(adata.X)
index_list = np.arange(adata.shape[0])
np.random.shuffle(index_list)
adata = adata[index_list].copy()

"nan" in adata.obs.original_celltype_ann

adata.obs.original_celltype_ann.isnull().any()

# store/load result:

# +
# adata.write(path_raw_HLCA)
# -

adata = sc.read(path_raw_HLCA)

# ### Add cell reference annotation and filter cells:

# Original cell type labeling (as provided by dataset providers) will now be translated to the current version of the cell type ontology, consisting of 5 levels.

# First, import the .csv that contains the translations:

harmonizing_df = reference_based_harmonizing.load_harmonizing_table(
    path_celltype_mapping
)

# Create a dataframe that contains each cell type name from the consensus ontology as indices, with their matching annotations at the other levels. This will simplify mapping:

consensus_df = reference_based_harmonizing.create_consensus_table(harmonizing_df)

# create a dataframe that for each original celltype annotation (from all datasets pooled) provides the translation to the consensus ontology at all levels:

celltype_translation_df = (
    reference_based_harmonizing.create_orig_ann_to_consensus_translation_df(
        adata, consensus_df, harmonizing_df, verbose=False
    )
)

# now translate the original annotations to the consensus in your AnnData:

adata = reference_based_harmonizing.consensus_annotate_anndata(
    adata, celltype_translation_df, verbose=True
)

# now remove unicorns and artifacts (cells annotated as low quality, doublets etc.)

print(
    "Number of unicorns and artifact cells to remove:",
    sum(adata.obs.original_ann_level_1 == "Unicorns and artifacts"),
)

adata.shape

adata = adata[adata.obs.original_ann_level_1 != "Unicorns and artifacts", :].copy()

adata.shape

# add "clean" annotations (with "None" for cells that do not have annotation at that level):

adata = reference_based_harmonizing.add_clean_annotation(adata,input_ann_type="original")

# ### add sample/donor annotations from LCA metadata tables:

# Note that the naming of the _samples_ is harmonized rather than the donor naming, so use sample names to copy metadata to AnnData object.

metadata = preprocessing.get_sample_annotation_table_LCA(dir_sample_metadata)

# remove subjects with lung condition that we think will affect lung significantly:

# first, check if there are any rows with "known lung disease" set to yes, but without specified condition. If there are, change the code in the cell below the next cell

for row in metadata.loc[metadata.known_lung_disease == "yes", :].index:
    matching_condition = metadata.loc[row, "condition"]
    if pd.isnull(matching_condition) or matching_condition == "nan":
        print(row, metadata.loc[row, "condition"])

# now check set of lung conditions, and make selection of which ones to remove:

lung_conditions = [x for x in (set(metadata.condition)) if not pd.isnull(x)]
lung_conditions_to_remove_from_data = [
    lc
    for lc in lung_conditions
    if lc
    # this is a list of the non-healthy subjects that we want to keep in,
    # as the tissue that was sampled should not have been (dramatically)
    # affected by the lung disease:
    not in [
        "carcinoid (non-tumor tissue)",
        "non-small cell lung cancer (non-tumor tissue)",
        "had TB as a child (fully treated over 30+ years)",
        "healthy",
        "unknown",
        "worsening respiratory function prior to arrest",
        "acute pneumonia, sample from unaffected tissue",
        "acute pneumonia, left lung, lower lobe, sample from unaffected tissue"
    ]
]
print("lung conditions to REMOVE from data:")
for i in lung_conditions_to_remove_from_data:
    print(i)
print("\nlung conditions to KEEP in data:")
for i in lung_conditions:
    if i not in lung_conditions_to_remove_from_data:
        print(i)
subjects_to_remove = sorted(
    set(
        metadata.loc[
            metadata.condition.isin(lung_conditions_to_remove_from_data), "subject_ID"
        ]
    )
)
print("\nnumber of subjects to remove:", len(subjects_to_remove))

subjects_to_remove

samples_in_adata = sorted(set(adata.obs["sample"]))
samples_in_metadata = metadata.index
print("n samples in adata:", len(samples_in_adata))
if len(samples_in_metadata) != len(set(samples_in_metadata)):
    print("WARNING: DUPLICATE SAMPLE NAMES IN METADATA TABLE! THIS SHOULD BE FIXED.")
for sample in samples_in_adata:
    if sample not in samples_in_metadata:
        print(sample, "is in AnnData object but not in metadata. Check this.")

metadata.index.value_counts().sort_values(ascending=False).head(3)

adata.obs.columns

metadata_columns_to_drop = [
    "IF_AVAILABLE/_APPLICABLE_-->",
    "Institute",
    "Study_PI",
    "library_ID",
    "publication_ID",
    "repository_ID",
    "library-construction_batch",
    "year_of_sample_collection",
    "relative_sample_collection_timepoint",
    "treatment_status",
    "number_of_cells_loaded",
    "tissue_dissociation_protocol"
]

metadata.drop(columns=metadata_columns_to_drop, inplace=True)

for cat in metadata.columns:
    sample_to_cat_dict = dict(zip(metadata.index, metadata[cat]))
    adata.obs[cat] = adata.obs["sample"].map(sample_to_cat_dict)

# now we can remove cells by subject ID:

filter_by_subject = ~adata.obs.subject_ID.isin(subjects_to_remove)
print("removing", sum(~filter_by_subject), "cells from adata based on lung condition.")
adata = adata[filter_by_subject, :].copy()

adata.shape

# check within-dataset diversity of technical covariates

adata.obs.groupby("study").agg(
    {
        "cell_ranger_version": "nunique",
        "disease_status": "nunique",
        "fresh_or_frozen": "nunique",
        "known_lung_disease": "nunique",
        "sample_type": "nunique",
        "sequencing_platform": "nunique",
        "single_cell_platform": "nunique",
        "subject_type": "nunique",
    }
)

# write/read file:

# +
# adata.write(path_subj_filt_samp_ann_HLCA)
# -

adata = sc.read(path_subj_filt_samp_ann_HLCA)

# ## splitting of datasets into separate batches, where necessary

# Three datasets should be split into seperate batches (note that this is based on the [study_splitting_by_batch_effect_assessment.ipynb](./study_splitting_by_dataset_assessment.ipynb) notebook, in which we check which studies should be split up into multiple datasets based on experimental conditions, such as 10X chemistry).
#     - lafyatis/rojas: different 10x versions
#     - seibold: includes both 10x_3'_v2 and 10x_3'_v3
#     - jain_misharin: 10x_5'_v1 and 10x_5'_v2

# generate sample to study dict, that will be updated with split datasets below

sample_to_study_df = adata.obs.groupby("sample").agg(
    {
        "sample": "first",
        "study": "first",
        "single_cell_platform": "first",
    }
)
sample_to_dataset_dict = dict(
    zip(sample_to_study_df["sample"], sample_to_study_df.study)
)

# lafyatis/rojas

samples_lafyatis_to_v1_dataset = {
    sample: f"{study}_10Xv1"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Lafyatis_Rojas_2019"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_3'_v1"
}
samples_lafyatis_to_v2_dataset = {
    sample: f"{study}_10Xv2"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Lafyatis_Rojas_2019"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_3'_v2"
}
sample_to_dataset_dict.update(samples_lafyatis_to_v1_dataset)
sample_to_dataset_dict.update(samples_lafyatis_to_v2_dataset)

# jain/misharin

samples_jain_to_v1_dataset = {
    sample: f"{study}_10Xv1"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Jain_Misharin_2021"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_5'_v1"
}
samples_jain_to_v2_dataset = {
    sample: f"{study}_10Xv2"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Jain_Misharin_2021"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_5'_v2"
}
sample_to_dataset_dict.update(samples_jain_to_v1_dataset)
sample_to_dataset_dict.update(samples_jain_to_v2_dataset)

# seibold

samples_seibold_to_v2_dataset = {
    sample: f"{study}_10Xv2"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Seibold_2020"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_3'_v2"
}
samples_seibold_to_v3_dataset = {
    sample: f"{study}_10Xv3"
    for sample, study in sample_to_dataset_dict.items()
    if study == "Seibold_2020"
    and sample_to_study_df.loc[sample, "single_cell_platform"] == "10x_3'_v3"
}
sample_to_dataset_dict.update(samples_seibold_to_v2_dataset)
sample_to_dataset_dict.update(samples_seibold_to_v3_dataset)

# now store resulting dataset assignments as dataset

adata.obs["dataset"] = adata.obs["sample"].map(sample_to_dataset_dict)

# ### Harmonize anatomical region:

# first add prefix to annotations from Barbry data, since their naming is inconsistent with other dataset's naming. Not adding prefix will result in mix-ups of translations.

# prefix barbry detailed annotation with coarse, because otherwise detailed
# mapping is many to one:
# make into list, so that we can freely add new categories
adata.obs["anatomical_region_detailed"] = adata.obs[
    "anatomical_region_detailed"
].tolist()
adata_barbry = adata[adata.obs["last_author_PI"] == "Barbry_Leroy", :].copy()
# check if prefixing was already done earlier:
if (
    sum(
        [
            fine.startswith(coarse)
            for coarse, fine in zip(
                adata_barbry.obs.anatomical_region_coarse,
                adata_barbry.obs.anatomical_region_detailed,
            )
        ]
    )
    == adata_barbry.n_obs
):
    # if "a" == "a":
    print("Fine anatomical regions barbry already prefixed.")
else:
    barbry_region_detailed_prefixed = [
        x + "_" + y
        for x, y in zip(
            adata_barbry.obs["anatomical_region_coarse"],
            adata_barbry.obs["anatomical_region_detailed"],
        )
    ]
    adata.obs.loc[
        adata_barbry.obs.index, "anatomical_region_detailed"
    ] = barbry_region_detailed_prefixed
del adata_barbry

# now harmonize anatomical region:

# read in harmonizing table:

harmonizing_df = reference_based_harmonizing.load_harmonizing_table(path_anatomical_loc_mapping)

# create translation table:

consensus_df = reference_based_harmonizing.create_consensus_table(
    harmonizing_df, max_level=3
)

# translate both levels (coarse and fine) to their harmonized counterpart:

for res in ["coarse", "fine"]:
    translation_df = (
        reference_based_harmonizing.create_orig_ann_to_consensus_translation_df(
            adata,
            consensus_df,
            harmonizing_df,
            verbose=False,
            ontology_type="anatomical_region_" + res,
        )
    )
    adata = reference_based_harmonizing.consensus_annotate_anndata(
        adata,
        translation_df,
        verbose=False,
        max_ann_level=3,
        ontology_type="anatomical_region_" + res,
    )

# merge coarse and fine annotations, so that we keep the finest annotation available for every sample:

adata = reference_based_harmonizing.merge_coarse_and_fine_anatomical_ontology_anns(
    adata, remove_harm_coarse_and_fine_original=True
)

# add ccf translation:

adata = reference_based_harmonizing.add_anatomical_region_ccf_score(
            adata, harmonizing_df
        )

# ### add age annotation (merging of age_in_years and age_range)

# add age as merger of 'age, in years' and 'age, range'
adata.obs["age"] = [
    preprocessing.age_converter(age, age_range)
    for age, age_range in zip(adata.obs["age_in_years"], adata.obs["age_range"])
]

# ### Change ethnicity to ancestry and map

# This is very imperfect mapping but the best we could do for now. SNP-based ancestry calling (using RNA-seq reads) is in the making...

ethnicity_remapper = {
    "asian": "asian",
    "black": "african",
    "latino": "american",
    "mixed": "mixed",
    "nan": "nan",
    "pacific islander": "pacific islander",
    "white": "european",
}

adata.obs['ancestry'] = adata.obs.ethnicity.map(ethnicity_remapper)

del adata.obs['ethnicity']

mixed_ethnicity_remapper = {
    "nan": np.nan,
    "white/asian": "european/asian",
    "Mixed race; South american": "american",
}

adata.obs["mixed_ancestry"] = adata.obs.mixed_ethnicity.map(mixed_ethnicity_remapper)

del adata.obs['mixed_ethnicity']

# ## Add digestion protocol information:

study_to_digest_prot = {
    "Barbry_Leroy_2020": "Cold protease 1h",
    "Jain_Misharin_2021": "Cold protease 1h",
    "Lafyatis_Rojas_2019": "Collagenase A + DNAse",
    "Misharin_Budinger_2018": "Collagenase D + DNAse",
    "Misharin_2021": "Collagenase D + DNAse",
    "Nawijn_2021": "Collagenase D + DNAse",
    "Teichmann_Meyer_2019": "Collagenase D + DNAse",
    "Meyer_2019": "Collagenase D + DNAse",
    "Banovich_Kropski_2020": "Dispase + collagenase",
    "Krasnow_2020": "Collagenase + Elastase + DNAse",
    "Seibold_2020": "Cold protease overnight",
}

adata.obs["tissue_dissociation_protocol"] = adata.obs.study.map(study_to_digest_prot)

# ## Sanity checks:

# ### check if all variables have values for all cells:

# we only expect that for some of them, but good to check

for cat in adata.obs.columns:
    if adata.obs[cat].isnull().any():
        print(cat, "has null values")
    elif "nan" in adata.obs[cat]:
        print(cat, "has 'nan' values")
#     print(cat, adata.obs[cat].isnull().any(), "nan" in adata.obs[cat].values)
#     if isinstance(adata.obs[cat].values, np.ndarray):
#         print(cat, np.nan in adata.obs[cat].values, "nan" in adata.obs[cat].values)
#     else:
#         print(cat, adata.obs[cat].values.isna().any(), "nan" in adata.obs[cat].values)

# ### Check if donor and sample names occur in only one dataset each:

temp = adata.obs.groupby("sample").agg({"dataset": "nunique"})
# check if sample names only occur in one dataset:
for sample in temp.index:
    if temp.loc[sample, "dataset"] != 1:
        print(str(sample) + ": this sample name occurs in multiple datasets")
temp = adata.obs.groupby("subject_ID").agg({"dataset": "nunique"})
for donor in temp.index:
    if temp.loc[donor, "dataset"] != 1:
        print(
            str(donor)
            + ": this subject_ID name occurs in "
            + str(temp.loc[donor, "dataset"])
            + " datasets!"
        )

# ### check if all values have only zeros as decimals:

# store remainders of division by 1, count for each row number of entries for which remainder is not 0 (they should all be zero if data are integers)

test = np.sum(adata.X.toarray() % 1 == 0, axis=1)

# select only those rows of adata that have non-integer values:

nonint_adata = adata[test != adata.shape[1], :].copy()

# check shape, it should have zero rows

nonint_adata.shape

# if it doesn't have zero rows, then check which datasets have non-integer values (in that case we received non-raw counts from them):

set(nonint_adata.obs.dataset)

# ### filter out cells with low numbers of genes expressed:

# filter out all cells with fewer than 200 genes expressed:

# check how many erythrocytes are present before filtering:
print(
    "Number of erythrocytes present before filtering:",
    np.sum(adata.obs.original_ann_level_3 == "Erythrocytes"),
)

n_cells_pre = adata.shape[0]
sc.pp.filter_cells(adata, min_genes=200)
n_cells_post = adata.shape[0]
print("Number of cells removed: " + str(n_cells_pre - n_cells_post))
print("Number of cells pre-filtering: " + str(n_cells_pre))
print("Number of cells post filtering: " + str(n_cells_post))
adata.shape

print(
    "Number of erythrocytes present after filtering:",
    np.sum(adata.obs.original_ann_level_3 == "Erythrocytes"),
)

# filter out remaining erythrocytes; they generally have too un-diverse transcriptomes to be analyzed properly:

n_cells_pre = adata.n_obs
adata = adata[adata.obs.original_ann_level_3 != "Erythrocytes", :].copy()
n_cells_post = adata.n_obs
print("Number of erythrocytes removed:", n_cells_pre - n_cells_post)

# ### Add QC annotations:

# calculate qc metrics, such as counts per cell, percentage of mitochondrial RNA etc.

# annotate with QC stuff:
adata = preprocessing.add_cell_annotations(adata, var_index="gene_ids")

# ### Filter out genes expressed in low number of cells:

n_genes_pre = adata.shape[1]
sc.pp.filter_genes(adata, min_cells=10)
n_genes_post = adata.shape[1]
print("Number of genes removed: " + str(n_genes_pre - n_genes_post))
print("Number of genes pre-filtering: " + str(n_genes_pre))
print("Number of genes post filtering: " + str(n_genes_post))

# ### harmonize nan/None/"nan" etc

# set all different types of None/NaN to np.nan
none_entries = adata.obs.applymap(utils.check_if_nan)
adata.obs = adata.obs.mask(none_entries.values)

# ### Store result:

adata.write(path_filt_ann_HLCA)
