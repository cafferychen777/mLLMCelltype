import scanpy as sc
import anndata
import numpy as np
from scipy import sparse

# This script contains mostly functions taken from scib (https://github.com/theislab/scib)
# (see also https://www.biorxiv.org/content/10.1101/2020.05.22.111161v2), that are
#  copy-pasted here so that not the entire scib package needs to be installed



# from scIB.utils.py:

def checkAdata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')

def checkBatch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')

def splitBatches(adata, batch, hvg= None):
    split = []
    if hvg is not None:
        adata = adata[:, hvg]
    for i in adata.obs[batch].unique():
        split.append(adata[adata.obs[batch]==i].copy())
    return split

def checkHVG(hvg, adata_var):
    if type(hvg) is not list:
        raise TypeError('HVG list is not a list')
    else:
        if not all(i in adata_var.index for i in hvg):
            raise ValueError('Not all HVGs are in the adata object')

def checkSanity(adata, batch, hvg):
    checkAdata(adata)
    checkBatch(batch, adata.obs)
    if hvg is not None:
        checkHVG(hvg, adata.var)

def merge_adata(adata_list, sep='-'):
    """
    merge adatas from list and remove duplicated obs and var columns
    """

    if len(adata_list) == 1:
        return adata_list[0]

    adata = adata_list[0].concatenate(
        *adata_list[1:], index_unique=None, batch_key='tmp')
    del adata.obs['tmp']

    if len(adata.obs.columns) > 0:
        # if there is a column with separator
        if sum(adata.obs.columns.str.contains(sep)) > 0:
            columns_to_keep = [
            name.split(sep)[1] == '0' for name in adata.var.columns.values]
            clean_var = adata.var.loc[:, columns_to_keep]
        else:
            clean_var = adata.var

    if len(adata.var.columns) > 0:
        if sum(adata.var.columns.str.contains(sep)) > 0:
            adata.var = clean_var.rename(
                columns={
                name : name.split('-')[0] for name in clean_var.columns.values})

    return adata



# from scIB.preprocessing.py

def SCRAN_normalize(adata, min_mean = 0.1, n_pcs=50, counts_per_cell = 1e4,
    louvain_r=0.5, ignore_R_warnings=False, log_transform=False):
    """adapted from scIB, returns normalized and log1p transformed adata"""
    # import R-related packages:
    import anndata2ri
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks

    if ignore_R_warnings == True:
        # Ignore R warning messages
        rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)


    checkAdata(adata)

    # massive speedup when working with sparse matrix
    if not sparse.issparse(adata.X): # quick fix: HVG doesn't work on dense mtx
        adata.X = sparse.csr_matrix(adata.X)

    anndata2ri.activate()
    ro.r('library("scran")')

    # keep raw counts
    adata.layers["counts"] = adata.X.copy()

    # Preliminary clustering for differentiated normalisation
    adata_pp = adata.copy()
    sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=counts_per_cell)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=n_pcs, svd_solver='arpack')
    sc.pp.neighbors(adata_pp)
    sc.tl.louvain(adata_pp, key_added='groups', resolution=louvain_r)

    ro.globalenv['data_mat'] = adata.X.T
    ro.globalenv['input_groups'] = adata_pp.obs['groups']
    # size_factors = ro.r(
    #    f'computeSumFactors(data_mat, clusters = input_groups, min.mean = {min_mean})')
    size_factors = ro.r('sizeFactors(computeSumFactors(SingleCellExperiment('
                            'list(counts=data_mat)), clusters = input_groups,'
                            f' min.mean = {min_mean}))')
    del adata_pp

    # modify adata
    adata.obs['size_factors'] = size_factors
    adata.X /= adata.obs['size_factors'].values[:,None]
    if log_transform == True:
        print("log1p-transforming data")
        sc.pp.log1p(adata)
    # convert to sparse, bc operation always converts to dense
    adata.X = sparse.csr_matrix(adata.X)
    # adata.raw = adata # Store the full data set in 'raw' as log-normalised data for statistical testing
    return adata

def hvg_batch(
    adata,
    batch_key=None,
    target_genes=2000,
    flavor='cell_ranger',
    n_bins=20,
    adataOut=False
    ):
    """
    Method to select HVGs based on mean dispersions of genes that are highly
    variable genes in all batches. Using a the top target_genes per batch by
    average normalize dispersion. If target genes still hasn't been reached,
    then HVGs in all but one batches are used to fill up. This is continued
    until HVGs in a single batch are considered.
    """

    checkAdata(adata)
    if batch_key is not None:
        checkBatch(batch_key, adata.obs)

    adata_hvg = adata if adataOut else adata.copy()

    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    # Calculate double target genes per dataset
    sc.pp.highly_variable_genes(adata_hvg,
                                flavor=flavor,
                                n_top_genes=target_genes,
                                n_bins=n_bins,
                                batch_key=batch_key)

    nbatch1_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches >
                                                           len(adata_hvg.obs[batch_key].cat.categories)-1]

    nbatch1_dispersions.sort_values(ascending=False, inplace=True)

    if len(nbatch1_dispersions) > target_genes:
        hvg = nbatch1_dispersions.index[:target_genes]

    else:
        enough = False
        print(f'Using {len(nbatch1_dispersions)} HVGs from full intersect set')
        hvg = nbatch1_dispersions.index[:]
        not_n_batches = 1

        while not enough:
            target_genes_diff = target_genes - len(hvg)

            tmp_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches ==
                                                                (n_batches-not_n_batches)]

            if len(tmp_dispersions) < target_genes_diff:
                print(f'Using {len(tmp_dispersions)} HVGs from n_batch-{not_n_batches} set')
                hvg = hvg.append(tmp_dispersions.index)
                not_n_batches += 1

            else:
                print(f'Using {target_genes_diff} HVGs from n_batch-{not_n_batches} set')
                tmp_dispersions.sort_values(ascending=False, inplace=True)
                hvg = hvg.append(tmp_dispersions.index[:target_genes_diff])
                enough=True

    print(f'Using {len(hvg)} HVGs')

    if not adataOut:
        del adata_hvg
        return hvg.tolist()
    else:
        return adata_hvg[:,hvg].copy

def scale_batch(adata, batch):
    """
    Function to scale the gene expression values of each batch separately.
    """

    checkAdata(adata)
    checkBatch(batch, adata.obs)

    # Store layers for after merge (avoids vstack error in merge)
    adata_copy = adata.copy()
    tmp = dict()
    for lay in list(adata_copy.layers):
        tmp[lay] = adata_copy.layers[lay]
        del adata_copy.layers[lay]

    split = splitBatches(adata_copy, batch)

    for i in split:
        sc.pp.scale(i)

    adata_scaled = merge_adata(split)

    # Reorder to original obs_name ordering
    adata_scaled = adata_scaled[adata.obs_names]

    # Add layers again
    for key in tmp:
        adata_scaled.layers[key] = tmp[key]

    del tmp
    del adata_copy

    return adata_scaled


# From integration.py:

def runScanorama(adata, batch, hvg = None):
    import scanorama
    checkSanity(adata, batch, hvg)
    split = splitBatches(adata.copy(), batch)
    emb, corrected = scanorama.correct_scanpy(split, return_dimred=True)
    corrected = corrected[0].concatenate(corrected[1:])
    emb = np.concatenate(emb, axis=0)
    corrected.obsm['X_emb']= emb
    #corrected.uns['emb']=True

    return corrected

def runBBKNN(adata, batch, hvg=None, n_pcs=50):
    import bbknn
    checkSanity(adata, batch, hvg)
    sc.pp.pca(adata, svd_solver='arpack', n_comps = n_pcs)
    if adata.n_obs <1e5:
        corrected = bbknn.bbknn(adata, batch_key=batch,trim=20, neighbors_within_batch=15, copy=True)
    if adata.n_obs >=1e5:
        corrected = bbknn.bbknn(adata, batch_key=batch,trim=30, neighbors_within_batch=30, copy=True)
    return corrected

# from itertools:
def ifilterfalse(predicate, iterable):
    # ifilterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

# written by LISA (i.e. not from scIB)
def reconstruct_indices_after_Scanorama(adata_pre, adata_corrected, batch_key):
    """note that scanorama re-orders and re-names the rows of the corrected
    anndata. It also drops all the annotation (.obs., .var etc.).
    To reconstruct the names and annotations, we use this function."""

    # the rows are re-ordered by batch, in order of batch-appearance in the
    # original anndata
    # therefore get the original order of the batches:
    batch_names = list(unique_everseen(adata_pre.obs[batch_key]))
    # start an empty list for the new indices (i.e. original cell names)
    new_indices = []
    # now for each batch, store the names of the cells that match with it,
    # in order of appearance.
    # append these cell names to the new-index-list
    for i, batch_name in enumerate(batch_names):
        cell_names = list(
            adata_pre[[x == batch_name for x in adata_pre.obs[batch_key]]].obs.index
        )
        new_indices = new_indices + cell_names
    # rename the index of the corrected anndata
    adata_corrected.obs.index = new_indices
    # now order rows as they were ordered in original anndata
    adata_corrected = adata_corrected[adata_pre.obs_names, :]
    # and copy the obs
    adata_corrected.obs = adata_pre.obs
    # now re-order the var as the original
    adata_corrected = adata_corrected[:, adata_pre.var_names]
    # and copy the var df
    adata_corrected.var = adata_pre.var

    return adata_corrected
