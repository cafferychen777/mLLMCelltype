import scanpy as sc
import pandas as pd

def get_unique_markers(
    adata,
    adata_expr_boolean,
    ct_key,
    rank_genes_groups_key="rank_genes_groups_filtered",
    min_n_markers_per_ct=1,
    max_n_markers_per_ct=3,
    max_out_ct_fraction=0.2,
    min_in_mean_pos_cell_fraction=0.3,
    verbose=True,
):
    """Function to get markers fulfilling defined criteria.
    Arguments:

    adata - (pseudobulk) anndata object on which a rank_genes_groups analysis
    has been performed
    ct_key - adata.obs column name containing cts on which dea was performed

    Returns:
    (final_markers, cts_to_redo)
    final_markers - dict with cts as keys and list of final markers as values
    for cts that passed filtering
    cts_to_redo - list of cts that did not pass filtering

    """
    cts_to_redo = list()
    final_markers = dict()
    for ct in adata.obs[ct_key].unique():
        ct_deg_df = sc.get.rank_genes_groups_df(
            adata,
            key=rank_genes_groups_key,
            group=ct,
        ).dropna(axis=0, how="any")
        total_marker_candidates = ct_deg_df.shape[0]
        if total_marker_candidates < min_n_markers_per_ct:
            if verbose:
                print(f"{ct}: Not sufficient marker candidates")
            cts_to_redo.append(ct)
        else:
            n_current_markers = 0
            current_markers = list()
            marker_candidate_i = 0
            while (
                n_current_markers < max_n_markers_per_ct
                and marker_candidate_i < total_marker_candidates
            ):
                # get gene name
                gene = ct_deg_df.names.values[marker_candidate_i]
                # CHECK 1: check in which cell types it is expressed:
                # first copy ct adata.obs column:
                single_marker_df = adata.obs.loc[:, [ct_key]].copy()
                # then copy matching (pseudobulk) gene counts
                single_marker_df[gene] = adata[:, gene].X.toarray()
                # check for every pseudobulk if it has counts for the gene
                single_marker_df[f"{gene}_notzero"] = single_marker_df[gene] > 0
                # calculate for every cell type the fraction of non-zero samples:
                marker_expr_frac_in_ct = single_marker_df.groupby(ct_key).agg(
                    {f"{gene}_notzero": "mean"}
                )
                # check if any other cell type has more than
                # max_out_group_fraction of samples expressing the gene
                expr_sparse_in_other_types = (
                    marker_expr_frac_in_ct.loc[
                        marker_expr_frac_in_ct.index != ct, f"{gene}_notzero"
                    ]
                    > max_out_ct_fraction
                ).sum() == 0
                # CHECK 2: check if the gene is frequently expressed within
                # the cell type of interest (i.e. not too sparsely expressed).
                # Note that this is different from mean in pseudobulks! We're
                # now look at fraction of positive single cells.
                single_marker_positive_df = adata_expr_boolean.obs.loc[
                    :, [ct_key]
                ].copy()
                # copy matching gene fraction of positive cells
                single_marker_positive_df[gene] = adata_expr_boolean[
                    :, gene
                ].X.toarray()
                # calculate for every cell type the mean fraction of positive cells per sample
                marker_pos_cell_fraction_in_ct = (
                    single_marker_positive_df.groupby(ct_key)
                    .agg({gene: "mean"})
                    .loc[ct, gene]
                )
                # check if a high enough fraction of the cells was positve:
                expr_dense_in_ct = (
                    marker_pos_cell_fraction_in_ct >= min_in_mean_pos_cell_fraction
                )
                if expr_sparse_in_other_types and expr_dense_in_ct:
                    # if criteria are fulfilled, store gene as marker:
                    current_markers.append(gene)
                    n_current_markers = len(current_markers)
                # ALTERNATIVE:
                # # check if any other cell type has more than max_out_group_fraction
                # of samples expressing the gene at a higher level than 10% of the median expression
                # within cell type:
                #                 ct_gene_median = single_marker_df

                marker_candidate_i += 1
            if n_current_markers >= min_n_markers_per_ct:
                if verbose:
                    print(f"{ct}: Found sufficient markers!")
                final_markers[ct] = current_markers
            else:
                cts_to_redo.append(ct)
                if verbose:
                    print(f"{ct}: Not sufficient markers fulfilled criteria.")
    return final_markers, cts_to_redo
