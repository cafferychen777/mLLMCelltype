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

# 加载原始数据
print("Loading raw data...")
adata = anndata.read_h5ad("data/raw/InfantGut.h5ad")
print(f"Original data shape: {adata.shape}")

# 分离小肠和大肠数据
small_intestine_mask = adata.obs['Organ'].isin(['duojejunum', 'ileum'])
large_intestine_mask = adata.obs['Organ'] == 'colon'

# 创建小肠和大肠的AnnData对象
small_intestine_adata = adata[small_intestine_mask].copy()
large_intestine_adata = adata[large_intestine_mask].copy()

print(f"\nSmall intestine data shape: {small_intestine_adata.shape}")
print(f"Large intestine data shape: {large_intestine_adata.shape}")

# 保存分割后的数据
processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

small_intestine_adata.write_h5ad(os.path.join(processed_dir, "small_intestine.h5ad"))
large_intestine_adata.write_h5ad(os.path.join(processed_dir, "large_intestine.h5ad"))

# 函数用于处理数据
def process_dataset(adata, dataset_name):
    print(f"\nProcessing {dataset_name}...")

    # 保存原始的 celltype（如果存在）
    if 'cell_name' in adata.obs:
        original_celltype = adata.obs['cell_name'].copy()
    else:
        original_celltype = pd.Series('Unknown', index=adata.obs.index)

    # 确保数据是非负整数
    X = adata.X.copy()
    if isinstance(X, scp.spmatrix):
        X.data = np.round(X.data).astype(int)
        X.data[X.data < 0] = 0
    else:
        X = np.round(X).astype(int)
        X[X < 0] = 0

    # 保存原始计数数据
    adata.layers["raw_counts"] = X

    # 基本预处理
    print("\nPerforming basic preprocessing...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # 确保数据中有 Sample 列
    if 'Sample' not in adata.obs:
        adata.obs['Sample'] = 'sample1'

    return adata, original_celltype

# 分别处理小肠和大肠数据
small_intestine_adata, small_intestine_celltype = process_dataset(small_intestine_adata, "small intestine")
large_intestine_adata, large_intestine_celltype = process_dataset(large_intestine_adata, "large intestine")


# 加载小肠和大肠的参考数据
print("\nLoading small intestine reference data...")
small_intestine_ref = anndata.read_h5ad("data/intestine/tabula_sapiens_Small_Intestine/minified_ref_adata.h5ad")
print(f"Small intestine reference data shape: {small_intestine_ref.shape}")

print("\nLoading large intestine reference data...")
large_intestine_ref = anndata.read_h5ad("data/intestine/tabula_sapiens_Large_Intestine/minified_ref_adata.h5ad")
print(f"Large intestine reference data shape: {large_intestine_ref.shape}")

# 函数用于运行 PopV 模型
def run_popv_model(query_adata, ref_adata, model_name, output_prefix):
    print(f"\nRunning PopV model for {model_name}...")

    try:
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

        # 初始化 PopV processor
        print("\nInitializing PopV processor with fast mode...")
        base_path = f"data/intestine/{model_name}"
        processor = popv.preprocessing.Process_Query(
            query_adata=query_adata,
            ref_adata=ref_adata,
            ref_labels_key='cell_type',
            ref_batch_key='Sample',
            prediction_mode='fast',
            pretrained_scvi_path=os.path.join(base_path, 'scvi'),
            cl_obo_folder=os.path.join(base_path, 'ontology'),
            save_path_trained_models=base_path,
            query_batch_key='Sample',
            query_layer_key='raw_counts',
            hvg=None
        )

        # 添加必要的信息到 adata.uns
        processor.adata.uns["unknown_celltype_label"] = "unknown"
        all_labels = list(ref_adata.obs['cell_type'].unique())
        if "unknown" not in all_labels:
            all_labels.append("unknown")
        processor.adata.uns["label_categories"] = np.array(all_labels)

        # 设置预测模式
        processor.adata.uns["_prediction_mode"] = "fast"
        processor.adata.uns["_save_path_trained_models"] = base_path
        processor.adata.obs["_predict_cells"] = "relabel"

        # 运行注释
        print("\nRunning annotation...")
        methods = [
            "Support_Vector",
            "XGboost",
            "ONCLASS",
            "CELLTYPIST"
        ]

        # 打印标签信息
        print(f"Number of unique labels: {len(all_labels)}")
        print(f"Labels: {all_labels}")

        # 初始化算法
        algorithms = {
            "Support_Vector": popv.algorithms.Support_Vector(),
            "XGboost": popv.algorithms.XGboost(classifier_dict={
                "tree_method": "hist",
                "device": "cpu",
                "objective": "multi:softprob",
                "max_depth": 6,
                "eta": 0.3,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "num_class": len(all_labels),
                "verbosity": 2  # 增加日志输出
            }),
            "ONCLASS": popv.algorithms.ONCLASS(),
            "CELLTYPIST": popv.algorithms.CELLTYPIST()
        }

        # 添加必要的信息到 adata.uns
        processor.adata.uns["_prediction_mode"] = "fast"
        processor.adata.uns["_save_path_trained_models"] = base_path
        processor.adata.uns["_ref_subsample"] = ref_adata.obs.index
        processor.adata.uns["_ref_labels"] = ref_adata.obs['cell_type']
        processor.adata.uns["label_categories"] = all_labels  # 添加这一行
        processor.adata.uns["unknown_celltype_label"] = "unknown"  # 添加这一行

        # 添加 Sample 列
        if 'Sample' not in ref_adata.obs.columns:
            ref_adata.obs['Sample'] = 'unknown'
        processor.adata.uns["_ref_batch"] = ref_adata.obs['Sample']
        processor.adata.uns["_ref_layer"] = None

        # 逐个运行算法
        for method in methods:
            try:
                print(f"Running {method}...")
                algorithms[method].predict(processor.adata)
            except Exception as e:
                print(f"Error during {method} annotation: {str(e)}")

        # 保存结果
        print("\nSaving results...")
        predictions = {}
        for method_lower in ['svm', 'xgboost', 'onclass', 'celltypist']:
            try:
                key = f'popv_{method_lower}_prediction'
                if key in processor.adata.obs:
                    predictions[method_lower] = processor.adata.obs[key]
                else:
                    print(f"Warning: {key} not found in results")
                    predictions[method_lower] = pd.Series('unknown', index=processor.adata.obs.index)
            except Exception as e:
                print(f"Error getting {method_lower} results: {str(e)}")
                predictions[method_lower] = pd.Series('unknown', index=processor.adata.obs.index)

        # 保存预测结果
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(output_dir, f"{output_prefix}_popv_fast_results.csv"))

        print(f"\nResults saved to: {output_dir}")
        print(f"- Prediction results: {output_prefix}_popv_fast_results.csv")

        return predictions_df
    except Exception as e:
        print(f"Error in PopV processing: {str(e)}")
        # 返回空结果
        return pd.DataFrame({
            'svm': pd.Series('unknown', index=query_adata.obs.index),
            'xgboost': pd.Series('unknown', index=query_adata.obs.index),
            'onclass': pd.Series('unknown', index=query_adata.obs.index),
            'celltypist': pd.Series('unknown', index=query_adata.obs.index)
        })

# 分别处理小肠和大肠数据
print("\nProcessing small intestine data...")
small_intestine_results = run_popv_model(
    small_intestine_adata,
    small_intestine_ref,
    "tabula_sapiens_Small_Intestine",
    "small_intestine"
)

print("\nProcessing large intestine data...")
large_intestine_results = run_popv_model(
    large_intestine_adata,
    large_intestine_ref,
    "tabula_sapiens_Large_Intestine",
    "large_intestine"
)

print("\nAll done!")
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
