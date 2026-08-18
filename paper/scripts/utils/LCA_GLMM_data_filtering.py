import scanpy as sc
import pandas as pd

def filter_based_on_n_cells_per_sample(data_per_sample_df, min_n_cells=50):
    """returns filtered data_per_sample_df with only samples that contain
    at least min_n_cells"""
    print("Number of samples before filtering:", data_per_sample_df.shape[0])
    samples_to_include = data_per_sample_df.iloc[
        (data_per_sample_df.n_cells >= min_n_cells).values, :
    ].index
    print(
        "number of samples to remove because of low number of cells:",
        data_per_sample_df.shape[0] - len(samples_to_include),
    )
    data_per_sample_df = data_per_sample_df.loc[samples_to_include, :].copy()
    print("Number of samples after filtering:", data_per_sample_df.shape[0])
    return data_per_sample_df


def filter_based_on_n_subjects_per_dataset(data_per_sample_df, min_n_subj):
    """filters data_per_sample_df based on number of subjects per dataset.
    Returns filtered data_per_sample_df."""
    data_per_dataset = (
        data_per_sample_df.groupby("dataset")
        .agg({"subject_ID": "nunique"})
        .rename(columns={"subject_ID": "n_subjects"})
    )
    print("Number of datasets before filtering:", data_per_dataset.shape[0])
    datasets_to_include = data_per_dataset.iloc[
        (data_per_dataset.n_subjects >= min_n_subj).values, :
    ].index
    print(
        "Number of datasets to remove because of fewer than",
        min_n_subj,
        "subjects:",
        data_per_dataset.shape[0] - len(datasets_to_include),
    )
    print(set(data_per_dataset.index) - set(datasets_to_include))
    print("Number of datasets after filtering:", len(datasets_to_include), "\n")
    samples_to_include = [
        sample
        for sample in data_per_sample_df.index
        if data_per_sample_df.loc[sample, "dataset"] in datasets_to_include
    ]
    print("Number of samples before filtering:", data_per_sample_df.shape[0])
    data_per_sample_df = data_per_sample_df.loc[samples_to_include, :].copy()
    print("Number of samples after filtering:", data_per_sample_df.shape[0])
    return data_per_sample_df


def annotated_proportion_checker(data_per_sample_df, min_prop_ann=0.8):
    # checks which of the columns in your data_per_sample_df are
    # fully annotated, which have at least min_prop_ann
    # annotated, and which are under-annotated.
    # Returns tuple:
    # cats_fully_annotated, cats_suff_annotated, cats_underannotated
    cats = [
        col
        for col in data_per_sample_df
        if col not in ["sample", "subject_ID", "dataset", "n_cells"]
    ]
    n_samples = data_per_sample_df.shape[0]
    # calculate proportion of samples that has NA
    na_prop = [data_per_sample_df[cat].isnull().sum() / n_samples for cat in cats]
    na_prop = pd.Series(na_prop, index=cats)
    # store samples that have an na_prop of 0
    cats_fully_annotated = na_prop[na_prop == 0].index.tolist()
    # store samples that have at least min_prop_ann annotated
    cats_suff_annotated = na_prop.iloc[(na_prop <= (1 - min_prop_ann)).values].index.tolist()
    # store samples that have less than min_prop_ann annotated:
    cats_underannotated = sorted(set(cats) - set(cats_suff_annotated))
    # print info
    print("Number of cats fully annotated:", len(cats_fully_annotated))
    print(sorted(cats_fully_annotated))
    print(
        "Number of cats with at least fraction {} annotated:".format(
            min_prop_ann
        ),
        len(cats_suff_annotated),
    )
    print(sorted(cats_suff_annotated))
    print("Number of cats not sufficiently annotated:", len(cats_underannotated))
    print(cats_underannotated)
    return cats_fully_annotated, cats_suff_annotated, cats_underannotated


def filter_unannotated_samples(data_per_sample_df, cats_to_include):
    print("Number of samples before filtering:", data_per_sample_df.shape[0])
    # include non-covariate categories we want to keep:
    for additional_cat in ["subject_ID", "sample", "n_cells", "dataset"]:
        if additional_cat in data_per_sample_df.columns:
            cats_to_include.append(additional_cat)
    data_per_sample_df = data_per_sample_df.loc[:, cats_to_include]
    # filter out all rows that have an NaN value:
    data_per_sample_df.dropna(axis=0, inplace=True)
    print("Number of samples after filtering:", data_per_sample_df.shape[0])
    return data_per_sample_df


def filter_underrepresented_groups(data_per_sample_df, min_inst=3):
    """filters out samples that come from groups that have fewer than
    min_inst instances, at subject_ID level! Only
    anatomical_region_level_1 is considered at sample level."""
    print("Number of samples before filtering:", data_per_sample_df.shape[0])
    data_per_subject_df = data_per_sample_df.groupby("subject_ID").agg(
        dict(zip(data_per_sample_df.columns, data_per_sample_df.shape[1] * ["first"]))
    )
    for cat in data_per_sample_df.columns:
        if (
            not data_per_sample_df[cat].dtype == float
            and not data_per_sample_df[cat].dtype == int
            and cat not in ["subject_ID", "dataset"]
        ):
            if cat == "anatomical_region_level_1":
                # count at sample level
                value_counts = data_per_sample_df[cat].value_counts()
            else:
                # count at subject level
                value_counts = data_per_subject_df[cat].value_counts()
            underrepr_conds = value_counts.iloc[(value_counts < min_inst).values]
            if len(underrepr_conds) > 0:
                print(cat, "\nunderrepresented conditions:")
                print(underrepr_conds, "\n")
                print("filetering out samples with underrepresented conditions...")
                samples_to_filter = [
                    sample
                    for sample in data_per_sample_df.index
                    if data_per_sample_df.loc[sample, cat] in underrepr_conds
                ]
                data_per_sample_df = data_per_sample_df.loc[
                    [
                        sample not in samples_to_filter
                        for sample in data_per_sample_df.index
                    ],
                    :,
                ]
    print("Number of samples after filtering:", data_per_sample_df.shape[0])
    return data_per_sample_df
