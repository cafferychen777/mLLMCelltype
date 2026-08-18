import pandas as pd
import numpy as np
import scanpy as sc
import glob
from scipy import sparse

# HELPER FUNCTIONS:
def get_genes_to_keep(genes_to_keep_file_path):
    """cellrangers default gene filtering can be found here
    https://support.10xgenomics.com/single-cell-gene-expression/software/release-notes/build#grch38_1.2.0
    Here we generate a list of genes to keep according to ensembl84
    genome annotation and the matching cellranger filtering.
    Arguments:
        data_dir - LCA directory that has "references" folder. Including
                    trailing slash.
    Returns:
        list of genes ids to keep
    """
    if pd.isnull(genes_to_keep_file_path):
        genes_to_keep_file_path = "../references/Homo_sapiens_GRCh38_84_gene_ids_to_keep_cellrangerbased.txt"
    with open(
        genes_to_keep_file_path
    ) as file:
        gene_ids_to_keep = file.readlines()
    # strip off '\n' at end of strings:
    gene_ids_to_keep = [gene_id.strip() for gene_id in gene_ids_to_keep]
    return gene_ids_to_keep


def clean_10x_adata_var(adata):
    """cleans up adata.var (drops feature types columns that is Gene Expression
    for all, genome col, sets gene_ids to index and gene_symbols to a column of
    adata.var, remove index name.)"""

    columns_to_drop = [
        col for col in ["feature_types", "genome"] if col in adata.var.columns
    ]
    adata.var = (
        adata.var.drop(columns=columns_to_drop)
        .reset_index()
        .rename(columns={"index": "gene_symbols"})
        .set_index("gene_ids")
    )
    adata.var.index.name = None
    return adata



# IMPORT FUNCTIONS (ONE PER STUDY)
def read_file_Banovich_Kropski_2020(project_dir, verbose=True):
    # read in metadata:
    meta = pd.read_csv(
        f"{project_dir}01_Metadata/20210414_VUMC.TGen_control_annotation.csv",
        index_col=0,
    )
    # reformat barcodes to correspond with adata barcodes:
    def clean_bc(bc):
        split_bc = bc.strip("-1").split("_")
        new_bc = split_bc[1] + "_" + split_bc[0]
        return new_bc

    meta.index = [clean_bc(bc) for bc in meta.index]
    # remove doublets from meta:
    meta = meta.loc[meta["doublet/singlet"] == "Singlets", :].copy()
    # read in directory names
    dir_paths = glob.glob(
        f"{project_dir}04_Counts/IPF*"
    )  # "/outs/filtered_feature_bc_matrix.h5")
    sample_to_dir = dict()
    # extract sample names from paths
    for dir_path in dir_paths:
        sample_name = dir_path.split("/")[-1].split("_")[-2]
        sample_to_dir[sample_name] = dir_path
    # initiate dictionary to store data in:
    adatas = dict()
    # loop through samples and import data:
    for sample_name, sample_dir in sample_to_dir.items():
        if verbose:
            print("importing data for sample", sample_name)
        adata = sc.read_10x_h5(
            f"{sample_to_dir[sample_name]}/outs/filtered_feature_bc_matrix.h5"
        )
        adata = clean_10x_adata_var(adata)
        # if all barcodes end with -1, remove these suffices:
        if adata.obs.index.str.endswith("-1").all():
            adata.obs.index = adata.obs.index.str.strip("-1")
        adatas[sample_name] = adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique="_",
    )
    # subset to cells in metadata:
    adata = adata[meta.index, :].copy()
    if verbose:
        print("n cells in meta:", meta.shape[0])
        print("n cells from meta found in adata:", adata.n_obs)
    # add cell type annotations:
    adata.obs["original_celltype_ann"] = meta.loc[adata.obs.index, "celltype"]
    # add dataset annotations
    adata.obs["study_long"] = "TGen_VUMC_Banovich_Kropski_2020Habermann_and_unpubl"
    adata.obs["study"] = "Banovich_Kropski_2020"
    adata.obs["last_author_PI"] = "Banovich_Kropski"
    return adata



def read_file_Barbry_Leroy_2020(project_dir, verbose=True, use_souped=True):
    if verbose:
        print("importing all data at once")
    if use_souped:
        print("importing souped data...")
        adata = sc.read(project_dir + "04_Counts/HCA_Lung_Barbry_Raw_SoupX.h5")
        adata = clean_10x_adata_var(adata)
    elif not use_souped:
        print("importing non-souped data...")
        adata = sc.read(project_dir + "04_Counts/HCA_Lung_Barbry_Raw.h5")
        adata = clean_10x_adata_var(adata)
    # remove columns that we will use later anyway based on metadata
    adata.obs.drop(
        columns=["position", "method", "Sex", "Age", "donor", "batch"], inplace=True
    )
    adata.obs.rename(
        columns={
            "manip": "sample",
            "CellType": "original_celltype_ann",
        },
        inplace=True,
    )
    # remove obsm
    del adata.obsm
    # add dataset annotations
    adata.obs["study_long"] = "CNRS_Barbry_Leroy_2020Deprez"
    adata.obs["last_author_PI"] = "Barbry_Leroy"
    adata.obs["study"] = "Barbry_Leroy_2020"
    return adata


def read_file_Jain_Misharin_2021(
    project_dir, verbose=True
):
    # in the metadata file, samples are numbered 0-7
    # Here's how they translate to library IDs/sample names:
    scvibatch_to_sample_dict = {
        0: "SC172",
        1: "SC173",
        2: "SC174",
        3: "SC181",
        4: "SC182",
        5: "SC183",
        6: "SC184",
        7: "SC185",
    }
    adatas = dict()
    for sample_number, sample_name in scvibatch_to_sample_dict.items():
        if verbose:
            print("working on sample {}...".format(sample_name))
        adata = sc.read_10x_h5(
            "{}04_Counts/{}/filtered_feature_bc_matrix.h5".format(
                project_dir, sample_name
            )
        )
        # clean up adata.var
        adata = clean_10x_adata_var(adata)
        if verbose:
            print(adata.shape)
        # add sample number suffix so that barcodes correspond to metadata
        # barcodes:
        adata.obs.index = [f"{bc}-{sample_number}" for bc in adata.obs.index]
        # note that SC174 is a multiplexed sample with cells from two patients,
        # we'll split hem here:
        if sample_name == "SC174":
            if verbose:
                print("splitting SC174 into two")
            # read in file that specifies which cell belongs to which
            # sample/patient
            SC174_splitter = pd.read_csv(
                f"{project_dir}04_Counts/SC174/cell_patient.csv", index_col=1
            ).drop(columns="Unnamed: 0")
            SC174_SC172_cells = SC174_splitter.loc[
                SC174_splitter["patient"] == "SC172", :
            ].index
            SC174_SC173_cells = SC174_splitter.loc[
                SC174_splitter["patient"] == "SC173", :
            ].index
            # store split count matrix
            # add suffix -2 here for correspondence with metadata
            adatas["SC174_SC172"] = adata[
                [
                    f"{cell}-2"
                    for cell in SC174_SC172_cells
                    if f"{cell}-2" in adata.obs.index
                ],
                :,
            ].copy()
            if verbose:
                print("SC174_SC172 shape:", adatas["SC174_SC172"].shape)
            adatas["SC174_SC173"] = adata[
                [
                    f"{cell}-2"
                    for cell in SC174_SC173_cells
                    if f"{cell}-2" in adata.obs.index
                ],
                :,
            ].copy()
            if verbose:
                print("SC174_SC173 shape:", adatas["SC174_SC173"].shape)
        else:
            # store in dict
            adatas[sample_name] = adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique=None,
    )
    # import metadata
    metadata = pd.read_csv(f"{project_dir}01_Metadata/obs.csv", index_col=0)
    # check which cells are both in adata and in metadata
    n_cells_adata = adata.n_obs
    n_cells_metadata = metadata.shape[0]
    annotated_cells_from_adata = adata.obs.index[adata.obs.index.isin(metadata.index)]
    n_cells_annotated_and_in_count_matrix = len(annotated_cells_from_adata)
    if verbose:
        print("Number of cells in count matrices:", n_cells_adata)
        print("Number of cells in metadata:", n_cells_metadata)
        print("Number of cells in both:", n_cells_annotated_and_in_count_matrix)
    # subset adata
    adata = adata[annotated_cells_from_adata, :].copy()
    # copy cell type annotations:
    adata.obs["original_celltype_ann"] = metadata.loc[adata.obs.index, "cell_type"]
    # remove cells from two low-quality annotations:
    clusters_to_remove = ["Remove: doublets", "Remove: low quality", "Multiciliated, low quality"]
    adata = adata[~adata.obs.original_celltype_ann.isin(clusters_to_remove), :].copy()
    if verbose:
        print(
            n_cells_annotated_and_in_count_matrix - adata.n_obs,
            "cells from low quality clusters removed.",
        )
        print("final shape:", adata.shape)
    # add dataset level annotations:
    adata.obs["study_long"] = "Northwestern_Jain_Misharin_2021_unpubl"
    adata.obs["study"] = "Jain_Misharin_2021"
    adata.obs["last_author_PI"] = "Jain_Misharin"
    # return:
    return adata



def read_file_Krasnow_2020(project_dir, verbose=True):
    # read in metadata
    meta = pd.read_csv(
        f"{project_dir}01_Metadata/krasnow_hlca_10x_metadata.csv", index_col=0
    )
    # samples have multiple files (libraries), here's the translation
    sample_to_lib_dict = {
        "distal 1a": ["P1_2", "P1_3", "P1_4"],
        "distal 2": ["P2_1", "P2_2", "P2_5", "P2_7"],
        "medial 2": ["P2_3", "P2_4", "P2_6", "P2_8"],
        "distal 3": ["P3_5", "P3_6", "P3_7"],
        "proximal 3": ["P3_2", "P3_3", "P3_4"],
    }
    # initiate adata dict
    adatas = dict()
    for sample, libs in sample_to_lib_dict.items():
        if verbose:
            print("importing libraries from sample", sample)
        adatas_sample = dict()
        for lib_name in libs:
            if verbose:
                print("library", lib_name)
            adata = sc.read_10x_h5(
                f"{project_dir}04_Counts/krasnow_hlca_10x_84/pipelinerun_v1.0.0/cellranger/{lib_name}/count_matrices/raw_feature_bc_matrix.h5"
            )
            adata = clean_10x_adata_var(adata)
            if adata.obs.index.str.endswith("-1").all():
                adata.obs.index = adata.obs.index.str.strip("-1")
            # prefix lib name for correspondence with metadata names:
            adata.obs.index = [f"{lib_name}_{bc}" for bc in adata.obs.index]
            # check which cells are annotated and subset to those
            cells_from_lib = meta.index[meta.channel == lib_name]
            adata = adata[adata.obs.index.isin(cells_from_lib), :].copy()
            if verbose:
                print("Number of cells annotated in metadata:", len(cells_from_lib))
                print("Number of cells found in adata:", adata.shape)
            # copy cell type annotations:
            adata.obs["original_celltype_ann"] = meta.loc[
                adata.obs.index, "free_annotation"
            ]
            adatas_sample[lib_name] = adata
        # concatenate adatas from same sample into one
        adatas[sample] = sc.AnnData.concatenate(
            *adatas_sample.values(),
            join="outer",
            batch_key="sample",
            batch_categories=list(adatas_sample.keys()),
            index_unique=None,
        )
    # now concatenate all sample adatas into one adata
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique=None,
    )
    # add annotations:
    adata.obs["study_long"] = "Stanford_Krasnow_2020Travaglini"
    adata.obs["last_author_PI"] = "Krasnow"
    adata.obs["study"] = "Krasnow_2020"
    # return adata
    return adata



def read_file_Lafyatis_Rojas_2019(project_dir, verbose=True):
    # get file paths
    file_paths_long = glob.glob(f"{project_dir}04_Counts/*")
    # get file names:
    file_names = [x.split("/")[-1] for x in file_paths_long]
    # get sample names, i.e. remove end of file names ('_barcodes.tsv etc'):
    sample_names_all = sorted(
        list(set(["_".join(x.split("_")[:-1]) for x in file_names]))
    )
    # extract only normals:
    sample_names_long = [x for x in sample_names_all if "NOR" in x]
    # import metadata:
    meta = pd.read_csv(
        f"{project_dir}01_Metadata/CCA_NorControls_WorkspaceV2_activeIdent.csv",
        index_col=0,
    )
    # derive ID name from barcode prefixes:
    meta["ID"] = ["_".join(x.split("_")[:-1]) for x in meta.index]
    # store names of normal samples that are present in metadata df
    annotated_normal_samples = sorted([x for x in list(set(meta["ID"])) if "NOR" in x])
    # create a dictionary in which AnnData objects will be stored by sample name
    adatas = dict()
    # now loop through samples and import relevant files:
    for sample_name_long in sample_names_long:
        # take only what follows after the first underscore:
        sample_name = "_".join(sample_name_long.split("_")[1:])
        # store sample name clean (as in sample metadata file)
        sample_name_clean = sample_name.split("NOR")[0]

        if verbose:
            print("\n working on sample {} now...".format(sample_name))
        ### THIS NEEDS TO BE IMPROVED, METADATA SEEMS TO BE LACKING.
        # FOR NOW, WE HAVE METADATA FOR ONLY FOUR SAMPLES.
        # SO check if metadata is available:
        meta_sample_name = None
        for annotated_meta_sample in annotated_normal_samples:
            if annotated_meta_sample in sample_name:
                if verbose:
                    print(
                        "{} corresponds to meta-sample name {}".format(
                            sample_name, annotated_meta_sample
                        )
                    )
                meta_sample_name = annotated_meta_sample
        # if no metadata available, end this loop cycle and continue to next sample:
        if meta_sample_name == None:
            print("no corresponding meta-sample found. We will ignore this sample.")
            continue
        # now read in counts, barcodes and genes:
        adata = sc.read_10x_mtx(
            f"{project_dir}04_Counts",
            var_names="gene_ids",
            prefix=f"{sample_name_long}_",
        )
        # if all barcodes end with '-1', cut off this part:
        n_cells_unfiltered = adata.shape[0]
        if np.sum([x[-2:] == "-1" for x in adata.obs.index]) == n_cells_unfiltered:
            adata.obs.index = [x.strip("-1") for x in adata.obs.index]
        # import metadata:
        meta_sample = meta.iloc[[x == meta_sample_name for x in meta["ID"]], :]
        meta_sample.index = [x.split("_")[-1] for x in meta_sample.index]
        # check if indices are unique:
        if meta_sample.shape[0] != len(set(meta_sample.index)):
            print("WARNING: metadata indices are not unique!")
        # now subset only those cells from the AnnData object that are present
        # in the metadata df:
        adata = adata[meta_sample.index, :].copy()
        if verbose:
            print("Shape of adata after subsetting to annotated cells:", adata.shape)
        # and add annotation
        adata.obs["original_celltype_ann"] = meta_sample["norcca@active.ident"].values
        # store result in adatas dictionary
        adatas[sample_name_clean] = adata
    # concatenate samples into one adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique="-",
    )
    # add dataset level annotations:
    adata.obs["study_long"] = "Pittsburgh_Lafyatis_Rojas_2019Morse"
    adata.obs["last_author_PI"] = "Lafyatis_Rojas"
    adata.obs["study"] = "Lafyatis_Rojas_2019"
    # return adata
    return adata



def read_file_Meyer_2019(project_dir, verbose=True):
    if verbose:
        print("importing all data at once")
    # read in .h5ad
    adata_full = sc.read(f"{project_dir}04_Counts/lung.cellxgene.h5ad")
    # remove accessory annotation + clean up:
    adata = adata_full.copy()
    adata.var["gene_symbols"] = adata_full.var.index
    adata.var.index = adata_full.var["gene_ids-HCATisStab7509734"]
    adata.var.drop(columns=adata_full.var.columns, inplace=True)
    adata.var.index.name = None
    # drop .obs columns that are not of interest:
    adata.obs.drop(
        [
            "Time",
            "organ",
            "patient",
            "sample",
            "n_genes",
            "percent_mito",
            "leiden",
            "Donor",
        ],
        axis=1,
        inplace=True,
    )
    # rename remaining columns:
    adata.obs.rename(
        columns={
            "donor_time": "sample",
            "Celltypes": "original_celltype_ann",
        },
        inplace=True,
    )
    # remove uns and obsm
    del adata.uns
    del adata.obsm
    # undo normalization:
    adata.X = np.round(
        adata.X.toarray() * adata.obs["n_counts"].values[:, np.newaxis] / 10000
    )
    adata.X = sparse.csr_matrix(adata.X)
    # now remove total counts column:
    adata.obs.drop("n_counts", axis=1, inplace=True)
    # add annotation:
    adata.obs["last_author_PI"] = "Meyer"
    adata.obs["study_long"] = "Sanger_Meyer_2019Madissoon"
    adata.obs["study"] = "Meyer_2019"
    return adata


def read_file_Misharin_2021(project_dir, verbose=True):
    adatas = dict()
    # start with first set of samples
    sample_numbers_1 = list(range(84, 90))
    sample_names_1 = [f"SC{sn}" for sn in sample_numbers_1]
    # import data
    for sample_name in sample_names_1:
        if verbose:
            print("working on sample {}...".format(sample_name))
        adata = sc.read_10x_h5(
            "{}04_Counts/{}/filtered_feature_bc_matrix.h5".format(
                project_dir, sample_name
            )
        )
        # clean up adata.var
        adata = clean_10x_adata_var(adata)
        # if all barcodes end with '-1', cut off this part:
        n_cells = adata.shape[0]
        if np.sum([x[-2:] == "-1" for x in adata.obs.index]) == n_cells:
            adata.obs.index = [x[:-2] for x in adata.obs.index]
        metadata = pd.read_csv(
            f"{project_dir}01_Metadata/{sample_name}_idents.csv", index_col=0
        )
        # Take only the barcode part of the indices, not the prefixes or suffixes
        # That way the metadata indices will match with the adata.obs indices.
        metadata.index = [x.strip("-1")[-16:] for x in metadata.index]
        # check if uniqueness of indices is lost:
        if metadata.shape[0] != len(set(metadata.index)):
            print(
                "WARNING: Indices are not unique anymore after removing prefix! Look into this!"
            )
        # get cell names from metadata that are also present in adata.obs
        # print if not all cells were recovered:
        cells_to_keep = [cell for cell in metadata.index if cell in adata.obs.index]
        n_cells_not_recovered = metadata.shape[0] - len(cells_to_keep)
        if n_cells_not_recovered > 0:
            print(
                "WARNING: {} cells were in metadata, but not found in adata.obs!".format(
                    str(n_cells_not_recovered)
                )
            )
        # # subset adata to only annotated cells:
        adata = adata[cells_to_keep, :].copy()
        # now add annotations:
        adata.obs["original_celltype_ann"] = metadata.loc[
            adata.obs.index, sample_name + "_copy@active.ident"
        ]
        # two clusters are misnamed, and need a sample prefix. Will add it here:
        cells_to_change = [
            cell
            for cell, ann in zip(adata.obs.index, adata.obs.original_celltype_ann)
            if ann in ["C7_2_DC2_FCER1A", "C7_7_DC1_CLEC9A"]
        ]
        if len(cells_to_change) > 0:
            # add sample as prefix to these annotations:
            adata.obs.loc[cells_to_change, "original_celltype_ann"] = [
                sample_name + "_" + label
                for label in adata.obs.loc[cells_to_change, "original_celltype_ann"]
            ]
        # add sample to barcodes:
        adata.obs.index = [f"{bc}_{sample_name}" for bc in adata.obs.index]
        adatas[sample_name] = adata
    # now import second set of samples, it has its own metadata file
    # and formatting
    sample_numbers_2 = list(range(141, 145))
    sample_names_2 = [f"SC{sn}" for sn in sample_numbers_2]
    # import cell metadata for second set of samples:
    meta_2 = pd.read_csv(
        f"{project_dir}/01_Metadata/Bharat_samples_anno.csv.gz",
        index_col=0,
    )
    # rename index so that they correspond to anndata indices
    meta_2.index = [
        bc.split("_")[-1].split("-")[0] + "_" + sample
        for bc, sample in zip(meta_2.index, meta_2["Sample ID"])
    ]
    # filter out cells without annotations:
    meta_2 = meta_2.loc[~meta_2.cell_type_1.isnull(), :].copy()
    # filter out cells with low quality as annotation
    meta_2 = meta_2.loc[
        ~meta_2.cell_type_1.isin(["Immune,Low quality", "Low quality"]), :
    ].copy()
    for sample_name in sample_names_2:
        if verbose:
            print("working on sample {}...".format(sample_name))
        adata = sc.read_10x_h5(
            "{}04_Counts/{}/filtered_feature_bc_matrix.h5".format(
                project_dir, sample_name
            )
        )
        # clean up adata.var
        adata = clean_10x_adata_var(adata)
        # if all barcodes end with '-1', cut off this part:
        n_cells = adata.shape[0]
        if np.sum([x[-2:] == "-1" for x in adata.obs.index]) == n_cells:
            adata.obs.index = [x[:-2] for x in adata.obs.index]
        # append sample name to barcodes:
        adata.obs.index = [bc + "_" + sample_name for bc in adata.obs.index]
        # subset to only those cells that are found in metadata:
        if verbose:
            print(
                "n annotated cells from sample in meta:",
                np.sum(meta_2["Sample ID"] == sample_name),
            )
            print("n cells before filtering to annotated cells:", adata.n_obs)
        adata = adata[adata.obs.index.isin(meta_2.index), :].copy()
        if verbose:
            print("n cells after filtering to annotated cells:", adata.n_obs)
        # add cell type annotations:
        adata.obs["original_celltype_ann"] = meta_2.loc[adata.obs.index, "cell_type_5"]
        adatas[sample_name] = adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique=None,
    )
    # add annotations:
    adata.obs["study_long"] = "Northwestern_Misharin_2021_unpubl"
    adata.obs["study"] = "Misharin_2021"
    adata.obs["last_author_PI"] = "Misharin"
    # return adata
    return adata



def read_file_Misharin_Budinger_2018(
    project_dir,
    verbose=True,
):
    """Function to read in files from Reyfman et al. dataset.
    Returns: dictionary with anndatas
    matrix_type can be set for Reyfman dataset: either 'filtered'or 'raw'
    donor_type can be set for Reyf,an dataset: either 'Donor' or
    ....others, check file names."""
    # get file paths of files containing the correct matrix type (filtered or raw)
    # and correct donor type (donors, for now, since we want healthy only)
    # PART 1
    # READ IN FILES FROM SINGE-SAMPLE-DONORS
    file_paths_long = glob.glob(
        f"{project_dir}04_Counts/*Donor*filtered_gene_bc_matrices_h5.h5"
    )
    file_names = []
    donor_names = []
    for path in file_paths_long:
        file_name = path.split("/")[-1]
        donor_name_with_number = file_name.split("_filtered")[0]
        donor_name = donor_name_with_number[donor_name_with_number.find("Donor") :]
        file_names.append(file_name)
        donor_names.append(donor_name)
    # read in matching cell type annotations from single-sample donors:
    meta = pd.read_csv(
        f"{project_dir}01_Metadata/cell_type_annotation_coarse_20200225.csv",
        index_col=0,
    )
    # store sample of each barcode in a separate column. The sample is
    # specified in the barcode prefix:
    meta["sample"] = [x[:-17] for x in meta.index]
    # since sample names used in the meta-file are library ids, we'll need
    # a converter (this is based on Sasha's filled out metadata file):
    donor_to_sample_converter = {
        "Donor_01": "SC07",
        "Donor_02": "SC10",
        "Donor_03": "SC18",
        "Donor_04": "SC20",
        "Donor_05": "SC22",
        "Donor_06": "SC24",
        "Donor_07": "SC27",
        "Donor_08": "SC29",
    }
    # now read in files:
    adatas = dict()
    for file, donor in zip(file_names, donor_names):
        sample = donor_to_sample_converter[donor]
        adata = sc.read_10x_h5(f"{project_dir}04_Counts/{file}")
        adata = clean_10x_adata_var(adata)
        # if all barcodes end with '-1', cut off this part:
        n_cells = adata.n_obs
        if np.sum([bc.endswith("-1") for bc in adata.obs.index]) == n_cells:
            if verbose:
                print("all barcodes end with '-1', we will remove this suffix...")
            adata.obs.index = [x[:-2] for x in adata.obs.index]
        # WE ALSO HAVE ANNOTATIONS FOR THESE FILES, ALTHOUGH THEY MIGHT BE
        # IMPROVED LATER BY SASHA. FOR THE 6 LATER SAMPLES BELOW, I'M
        # STILL AWAITING ANNOTATIONS.
        # annotate cell types.
        # subset cell type metadata to cells from this sample:
        meta_sample = meta.iloc[[x == sample for x in meta["sample"]], :]
        # now clip off prefixes of barcodes, since they should be unique
        # within sample. This will allow for matching with anndata barcodes.
        meta_sample.index = [x[-16:] for x in meta_sample.index]
        # check if barcodes are still unique:
        if meta_sample.shape[0] != len(set(meta_sample.index)):
            print(
                "WARNING: clipping off of barcode prefixes resulted in loss\
of uniqueness of indices within sample {}. Look into this. Breaking out of loop\
now.".format(
                    sample
                )
            )
            break
        # check if all metadata cells are present in adata:
        if np.sum(meta_sample.index.isin(adata.obs.index)) != meta_sample.shape[0]:
            print(
                "WARNING: not all metadata cells can be found in anndata object.\
Look into this. We'll break out of loop now."
            )
            break
        # if all metadata cells could be retraced, add their annotation to
        # the anndata object now.
        # also, include only the cells that have metadata (the remainder
        # was not of sufficient quality according to Alexander Misharin's filtering)
        adata = adata[meta_sample.index, :].copy()
        adata.obs.loc[meta_sample.index, "original_celltype_ann"] = meta_sample.loc[
            :, "x"
        ].values
        # add annotations of interest:
        # store adata in adatas dict:
        adatas[sample] = adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique="_",
    )
    # add dataset-level annotations:
    adata.obs["last_author_PI"] = "Misharin_Budinger"
    adata.obs["study"] = "Misharin_Budinger_2018"
    adata.obs["study_long"] = "Northwestern_Misharin_Budinger_2018Reyfman"
    return adata


def read_file_Nawijn_2021(project_dir, verbose=True, genes_to_keep_file_path=None):
    # generate subject names:
    subjects = ["GRO-0" + str(s_n) for s_n in range(1, 10) if s_n != 5] + [
        "GRO-" + str(s_n) for s_n in [10, 11]
    ]
    # six subjects do not only have bronchial biopsies, but also nasal brushes
    # we will not include nasal sample from GRO-02, as this is a low quality
    # sample with only 79 cells
    subjects_with_nose_sample = ["GRO-0" + str(s_n) for s_n in [1,3,4]] + [
        "GRO-09",
        "GRO-10",
    ]
    # the biopsy data do not have standard cellranger gene filtering,
    # and therefore still contain about 60k genes.
    # I will filter the genes accordingly here. (see get_genes_to_keep function)
    gene_ids_to_keep = get_genes_to_keep(genes_to_keep_file_path)
    # initiate dictionary to store adatas in
    adatas = dict()
    # read in metadata. We'll use this first of all to filter to our cells
    # of interest.
    # metadata for all bronchial_biopsy samples:
    cell_meta_bronchial = pd.read_csv(
        project_dir
        + "01_Metadata/Bronchial_biopsy_annotation_metadata_Groningen_20210317.csv",
        index_col=0,
        sep=" ",
    )
    # metadata for all nasal_brush samples:
    cell_meta_nasal = pd.read_csv(
        project_dir + "01_Metadata/Nasal_brush_annotation_metadata_Groningen.csv",
        index_col=0,
        sep=" ",
    )
    # read in count matrices:
    for subj in subjects:
        if verbose:
            print(f"\nimporting data for {subj}")
        # read in the raw matrix, since the filtered one is too stringently
        # filtered. We will select cells based on provided annotations.
        path = project_dir + f"04_Counts/{subj}/raw_feature_bc_matrix/"
        adata = sc.read_10x_mtx(
            path, make_unique=False, gex_only=True, var_names="gene_ids"
        )
        # these data do not have standard cellranger gene filtering,
        # (see comments earlier), hence apply gene filtering now:
        adata = adata[:, gene_ids_to_keep].copy()
        # strip off '-1' from barcodes and add subject name
        # (for correspondence with metadata):
        adata.obs.index = [bc.strip("-1") + "-" + subj for bc in adata.obs.index]
        # select only those cells that are in the cell annotation data
        cells_from_sample = cell_meta_bronchial.index[
            cell_meta_bronchial["orig.ident"] == subj
        ]
        if verbose:
            print(f"Number of cells expected: {len(cells_from_sample)}")
        adata = adata[cells_from_sample, :].copy()
        # now copy cell annotations
        adata.obs["original_celltype_ann"] = cell_meta_bronchial.loc[
            cells_from_sample, "annotations"
        ]
        if verbose:
            print(f"Shape of count matrix: {adata.shape}")
        # remove subject again from barcodes, final suffices will
        # be added when concatenating all the datasets (that's cleaner)
        adata.obs.index = [bc.strip("-" + subj) for bc in adata.obs.index]
        # store in dict
        adatas[subj + "_biopsy"] = adata
        if subj in subjects_with_nose_sample:
            if verbose:
                print("importing nasal sample")
            path = (
                project_dir
                + f"04_Counts/nasal_brushes/{subj}/filtered_feature_bc_matrix/"
            )
            adata = sc.read_10x_mtx(
                path, make_unique=False, gex_only=True, var_names="gene_ids"
            )
            if adata.obs.index.str.endswith("-1").all():
                adata.obs.index = adata.obs.index.str.strip("-1")
            if subj in ["GRO-0" + str(s_n) for s_n in [1,3,4]]:
                # these samples have a different metadata file and formatting
                # from the other nasal samples.
                # select only cells with annotations
                cells_from_sample = cell_meta_nasal.index[
                    cell_meta_nasal["orig.ident"] == subj
                ]
                if verbose:
                    print(f"Number of cells expected: {len(cells_from_sample)}")
                # temporarily append sample names to adata barcodes, to
                # match with metadata sample naming
                adata.obs.index = [f"{bc}-{subj}" for bc in adata.obs.index]
                adata = adata[cells_from_sample, :].copy()
                if verbose:
                    print(f"Shape of count matrix: {adata.shape}")
                # now copy cell annotations
                adata.obs["original_celltype_ann"] = cell_meta_nasal.loc[
                    cells_from_sample, "annotations"
                ]
                # remove subject again from barcodes, final suffices will
                # be added when concatenating all the datasets (that's cleaner)
                adata.obs.index = [bc.strip("-" + subj) for bc in adata.obs.index]
            else:
                meta = pd.read_csv(
                    f"{project_dir}/01_Metadata/celltype_annotations_nasal_arms_09_10.csv",
                    index_col=0,
                )
                # subset to cells from subject:
                meta = meta.loc[meta["orig.ident"] == subj, :]
                # take only barcode and not prefixes, so that bcs correspond
                # to count matrix bcs:
                meta.index = [bc[-16:] for bc in meta.index]
                # subset to annotated cells and copy cell type annotations
                adata = adata[adata.obs.index.isin(meta.index), :].copy()
                adata.obs["original_celltype_ann"] = meta.loc[
                    adata.obs.index, "celltype"
                ]
                if verbose:
                    print("Number of cells expected:", meta.shape[0])
                    print("Shape of count matrix:", adata.n_obs)
            # store in dict
            adatas[subj + "_nasal_brush"] = adata
    # merge objects:
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique="_",
    )
    # remove feature_types column from adata.var:
    adata.var.drop(columns="feature_types", inplace=True)
    # add dataset-level metadata:
    adata.obs["study_long"] = "UMCG_Nawijn_2019VieiraBraga_and_unpubl"
    adata.obs["study"] = "Nawijn_2021"
    adata.obs["last_author_PI"] = "Nawijn"
    return adata



def read_file_Seibold_2020(
    project_dir,
    verbose=True,
):
    # import count matrix
    adatas = dict()
    v2_samples = ["T120", "T121", "T126", "T137", "T84", "T85", "T89"]
    v3_samples = ["T101", "T153", "T154", "T164", "T165", "T166", "T167", "T90"]
    for sample in v2_samples + v3_samples:
        if verbose:
            print("importing data for sample", sample)
        if sample in v2_samples:
            adata = sc.read_10x_mtx(
                f"{project_dir}04_Counts/V2_chemistry/{sample}/filtered_gene_bc_matrices/GRCh38/",
                var_names="gene_ids",
            )
        elif sample in v3_samples:
            # for some reason for v3 samples scanpy doesn't accept unzipped
            # files, so will import manually:
            adata = sc.read_mtx(
                f"{project_dir}04_Counts/V3_chemistry/{sample}/filtered_feature_bc_matrix/matrix.mtx"
            ).T
            barcodes = pd.read_table(
                f"{project_dir}04_Counts/V3_chemistry/{sample}/filtered_feature_bc_matrix/barcodes.tsv",
                index_col=0,
                header=None,
            )
            genes = pd.read_table(
                f"{project_dir}04_Counts/V3_chemistry/{sample}/filtered_feature_bc_matrix/features.tsv",
                index_col=0,
                header=None,
            )
            # clean up genes df and set adata.var to it
            genes.index.name = None
            genes = genes.drop(columns=2).rename(columns={1: "gene_symbols"})
            adata.var = genes
            # add barcodes/genes to anndata object
            adata.obs.index = list(barcodes.index.values)
            # if indices all end with "-1", remove from barcode
        # (for correspondence with metadata):
        if sum([bc.endswith("-1") for bc in adata.obs.index]) == adata.n_obs:
            adata.obs.index = [bc.strip("-1") for bc in adata.obs.index]
        adatas[sample] = adata
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique="_",
    )
    # import metadata as taken from GEO, update indices to match adata
    metadata = pd.read_table(
        f"{project_dir}01_Metadata/GSE134174_Processed_invivo_metadata.txt", index_col=0
    )
    new_meta_idc = [
        idc.split("_")[0] + "_" + donor
        for idc, donor in zip(metadata.index, metadata.Donor)
    ]
    metadata.index = new_meta_idc
    # get cells that are present both in metadata and adata
    cells_present_in_meta_and_adata = list(set(adata.obs.index) & set(metadata.index))
    if verbose:
        print("n cells in metadata:", metadata.shape[0])
        print(
            "n cells from metadata found in adata:",
            len(cells_present_in_meta_and_adata),
        )
    # subselect adata to only those cells
    adata = adata[cells_present_in_meta_and_adata, :].copy()
    # add cell type annotation:
    adata.obs["original_celltype_ann"] = metadata.loc[
        adata.obs.index, "subcluster_ident"
    ]
    adata.obs["last_author_PI"] = "Seibold"
    adata.obs["study_long"] = "NJH_Seibold_2020Goldfarbmuren"
    adata.obs["study"] = "Seibold_2020"
    # return
    return adata



def read_file_Teichmann_Meyer_2019(
    project_dir,
    verbose=True,
):
    # import fully processed adata object, use this for metadata:
    # this file was obtained from the COVID19 atlas website, taking the
    # parenchyma file of the VieiraBraga paper. It only has healthy.
    adata_proc = sc.read(
        project_dir
        + "03_Pipeline_output/vieira19_Alveoli_and_parenchyma_anonymised.processed.h5ad"
    )
    # shorten sample names to final 7 characters, these are the numbers that
    # are of relevance:
    adata_proc.obs["Sample"] = [x[-7:] for x in adata_proc.obs["Sample"]]
    samples = sorted(set(adata_proc.obs["Sample"]))
    # strip of prefixes (because they are of different lenghts, and varying with
    # and without underscore) and suffixes from indices:
    new_idc = [idx[-18:-2] for idx in adata_proc.obs.index]
    # now use sample number as prefix, to ensure that barcodes stay unique:
    new_idc = [
        sample + "_" + old_idx
        for sample, old_idx in zip(adata_proc.obs.Sample, new_idc)
    ]
    # update index names:
    adata_proc.obs.index = new_idc
    # store .obs as metadata dataframe
    meta = adata_proc.obs
    meta.drop(columns=["BroadCellType", "Donor", "Source", "Location"], inplace=True)
    adatas = dict()
    for sample in samples:
        if verbose:
            print("importing data for", sample)
        adata = sc.read_10x_h5(
            f"{project_dir}04_Counts/raw_counts/LungTranscriptome{sample}.h5"
        )
        adata = clean_10x_adata_var(adata)
        # if barcodes all have suffix -1, remove it so that barcodes
        # correspond with metadata barcodes. Also add sample name as prefix:
        if np.sum([x.endswith("-1") for x in adata.obs.index]) == adata.shape[0]:
            new_barcodes = [sample + "_" + idx[:16] for idx in adata.obs.index]
            adata.obs.index = new_barcodes
        else:
            print("not all barcodes end with -1. Update code!")
        # now check if all barcodes from this sample in metadata are present
        # in count matrices:
        cells_from_sample_in_meta = meta.index[meta["Sample"] == sample]
        cells_in_meta_and_adata = [
            cell for cell in cells_from_sample_in_meta if cell in adata.obs.index
        ]
        n_cells_in_meta = len(cells_from_sample_in_meta)
        n_cells_recovered = len(cells_in_meta_and_adata)
        if n_cells_recovered < n_cells_in_meta:
            print("Number of cells not recovered", n_cells_in_meta - n_cells_recovered)
        elif n_cells_recovered == n_cells_in_meta and verbose:
            print(f"All cells ({n_cells_recovered}) were recovered!")
        adata = adata[cells_in_meta_and_adata, :].copy()
        adata.obs["original_celltype_ann"] = meta.loc[adata.obs.index, "CellType"]
        adatas[sample] = adata
    # concatenate separate adatas
    adata = sc.AnnData.concatenate(
        *adatas.values(),
        join="outer",
        batch_key="sample",
        batch_categories=list(adatas.keys()),
        index_unique=None,
    )
    # add dataset and PI info:
    adata.obs["study_long"] = "Sanger_Teichman_Meyer_2019VieiraBraga"
    adata.obs["last_author_PI"] = "Teichmann_Meyer"
    adata.obs["study"] = "Teichmann_Meyer_2019"
    return adata



def read_file_Thienpont_2018(project_dir, verbose=True):
    adata = sc.read(project_dir + "04_Counts/matrix.h5ad")
    # import metadata
    meta = pd.read_csv(project_dir + "01_Metadata/LC_metadata.csv", index_col=0)
    # rename columns to match with LCA terminology:
    meta.rename(
        columns={
            "nGene": "n_genes_detected",
            "nUMI": "total_counts",
            "PatientNumber": "subject_ID",
            "CellType": "original_celltype_ann",
            "TumorType": "tumor_type",
            "TumorSite": "tumor_site",
            "CellFromTumor": "cell_from_tumor",
        },
        inplace=True,
    )
    # drop tumor type column (this is always "lung")
    meta.drop(columns=["tumor_type"], inplace=True)
    # change subject_ID so that they're not just numbers (could cause mix-ups
    # when combining with other datasets)
    meta.subject_ID = [
        "lambrechts_" + str(subj_number) for subj_number in meta.subject_ID
    ]
    # add metadata to adata.obs
    adata.obs = meta.loc[adata.obs.index, :]
    # add sample names based on barcodes:
    adata.obs["sample"] = [bc.split("_")[0] for bc in adata.obs.index]
    # add study name:
    adata.obs["study"] = "Thienpont_2018"
    # store dataset name:
    dataset_name = "KULeuven_Thienpont_2018Lambrechts"
    adata.obs["study_long"] = dataset_name
    # add dataset name (split into 10Xv1 and 10Xv2 dataset;
    # patient 1 and 2 were 10X v1, the rest 10Xv2):
    patient_to_dataset = dict()
    for patient_n in [1, 2]:
        patient_to_dataset["lambrechts_" + str(patient_n)] = dataset_name + "_v1"
    for patient_n in range(3, 9):
        patient_to_dataset["lambrechts_" + str(patient_n)] = dataset_name + "_v2"
    adata.obs["dataset"] = adata.obs.subject_ID.map(patient_to_dataset)
    # add last author info:
    adata.obs["last_author_PI"] = "Thienpont"
    # add lung vs nasal info:
    adata.obs["lung_vs_nasal"] = "lung"
    return adata
