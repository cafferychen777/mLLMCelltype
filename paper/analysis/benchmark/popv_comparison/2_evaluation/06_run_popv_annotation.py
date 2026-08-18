import popv
import anndata
import scanpy as sc
import os
import joblib
import numpy as np
import scipy.sparse as scp
import pandas as pd
from popv.algorithms import Support_Vector, XGboost, ONCLASS, CELLTYPIST

# 设置输出目录
output_dir = "results/benchmark/popv_comparison/2_evaluation"
os.makedirs(output_dir, exist_ok=True)

print("Loading processed data...")
query_adata = anndata.read_h5ad("data/processed/tcell_lifespan.h5ad")
print(f"Query data shape: {query_adata.shape}")
print(f"First few genes in query data: {list(query_adata.var_names[:5])}")


# 保存原始的 celltype（如果存在）
if 'cell_type' in query_adata.obs:
    original_celltype = query_adata.obs['cell_type'].copy()
else:
    original_celltype = pd.Series('Unknown', index=query_adata.obs.index)

# 确保数据是非负整数
X = query_adata.X.copy()
if isinstance(X, scp.spmatrix):
    X.data = np.round(X.data).astype(int)
    X.data[X.data < 0] = 0
else:
    X = np.round(X).astype(int)
    X[X < 0] = 0

# 保存原始计数数据
query_adata.layers["raw_counts"] = X

# 基本预处理
print("\nPerforming basic preprocessing...")
sc.pp.normalize_total(query_adata)
sc.pp.log1p(query_adata)

# 确保数据中有 Sample 列
if 'Sample' not in query_adata.obs:
    query_adata.obs['Sample'] = 'sample1'

# 加载minified reference data
print("\nLoading minified reference data...")
ref_adata = anndata.read_h5ad("data/pbmc/immune/minified_ref_adata.h5ad")
print(f"Reference data shape: {ref_adata.shape}")

# 创建基因名称到Ensembl ID的映射
gene_to_ensembl = dict(zip(ref_adata.var['feature_name'], ref_adata.var_names))
print(f"Total number of genes with mapping: {len(gene_to_ensembl)}")

# 找到查询数据集中可以映射的基因
mappable_genes = [gene for gene in query_adata.var_names if gene in gene_to_ensembl]
print(f"Number of query genes that can be mapped: {len(mappable_genes)}")

# 创建新的查询数据集，只包含可以映射的基因
query_adata = query_adata[:, mappable_genes].copy()

# 将基因名称转换为Ensembl ID
ensembl_ids = [gene_to_ensembl[gene] for gene in query_adata.var_names]
query_adata.var_names = ensembl_ids

# 检查基因重叠
common_genes = set(query_adata.var_names).intersection(set(ref_adata.var_names))
print(f"Number of common genes after mapping: {len(common_genes)}")

# 打印 ref_adata 的信息
print("\nReference data information:")
print("Available columns in ref_adata.obs:", list(ref_adata.obs.columns))
print("Available keys in ref_adata.uns:", list(ref_adata.uns.keys()))

print("\nInitializing PopV processor with fast mode...")
processor = popv.preprocessing.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,  # Use minified reference data
    ref_labels_key='cell_type',  # Use cell_type as label
    ref_batch_key='Sample',  # Use Sample as batch key
    prediction_mode='fast',  # Fast mode for quick prediction
    pretrained_scvi_path='data/pbmc/immune/scvi',  # Path to pretrained model
    cl_obo_folder='data/pbmc/immune/ontology',  # Path to cell ontology folder
    save_path_trained_models='data/pbmc/immune',  # Directory for pretrained models
    query_batch_key='Sample',  # Use Sample as query batch key
    query_layer_key='raw_counts',  # Use raw_counts as input data
    hvg=None  # Disable highly variable gene selection
)

# 添加必要的信息到 adata.uns
processor.adata.uns["unknown_celltype_label"] = "unknown"
processor.adata.uns["label_categories"] = np.array(list(ref_adata.obs['cell_type'].unique()) + ["unknown"])

print("\nRunning annotation...")
# 使用所有可用的算法进行注释
methods = [
    "Support_Vector",  # 支持向量机分类器
    "XGboost",  # XGBoost分类器
    "ONCLASS",  # OnClass分类器
    "CELLTYPIST"  # CellTypist分类器
]
popv.annotation.annotate_data(
    adata=processor.adata,
    methods=methods,  # 使用指定的算法
    save_path=output_dir
)

print("\nSaving results...")
# 删除无法序列化的对象
if '_ontology_graph' in processor.adata.uns:
    del processor.adata.uns['_ontology_graph']

# 保存结果
# 先处理特殊列
special_cols = ['_reference_labels_annotation', '_labels_annotation']
for col in special_cols:
    if col in processor.adata.obs.columns:
        del processor.adata.obs[col]

# 确保所有的标签列都是字符串类型
for col in processor.adata.obs.columns:
    try:
        if isinstance(processor.adata.obs[col].dtype, pd.CategoricalDtype):
            processor.adata.obs[col] = processor.adata.obs[col].astype(str)
        elif pd.api.types.is_numeric_dtype(processor.adata.obs[col]) or \
             pd.api.types.is_bool_dtype(processor.adata.obs[col]):
            processor.adata.obs[col] = processor.adata.obs[col].astype(str)
    except:
        print(f"Warning: Could not convert column {col} to string")
        continue

# 保存结果
processor.adata.write_h5ad(os.path.join(output_dir, "tcell_lifespan_popv_fast.h5ad"))

# 提取并保存预测结果
predictions = processor.adata.obs[[col for col in processor.adata.obs.columns if 'prediction' in col]]
# 添加原始的 celltype
predictions['original_celltype'] = original_celltype
predictions.to_csv(os.path.join(output_dir, "tcell_lifespan_popv_fast_results.csv"))

# 打印一些基本统计信息
print("\nAnnotation Statistics:")
print("\nPopV Majority Vote Prediction Distribution:")
print(processor.adata.obs['popv_majority_vote_prediction'].value_counts())

print("\nPrediction Score Statistics:")
print(processor.adata.obs['popv_majority_vote_score'].describe())

# 如果有原始的 celltype，打印一下分布
if 'cell_type' in query_adata.obs:
    print("\nOriginal Cell Type Distribution:")
    print(original_celltype.value_counts())

print(f"\nResults saved to: {output_dir}")
print("- Full annotated data: tcell_lifespan_popv_fast.h5ad")
print("- Prediction results: tcell_lifespan_popv_fast_results.csv")
