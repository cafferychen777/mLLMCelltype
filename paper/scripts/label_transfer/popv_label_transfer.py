import scanpy as sc
import popv
import popv.annotation as pa
from popv.preprocessing import Process_Query
import anndata as ad
import numpy as np
import pandas as pd
import os

# 设置popv参数
popv.settings.mode = "retrain"  # 使用retrain模式
popv.settings.compute_embedding = True  # 计算UMAP嵌入
popv.settings.return_probabilities = True  # 返回预测概率

print("Loading processed data...")

# 加载处理后的数据
ts_lung = sc.read_h5ad("data/processed/TS_Lung_processed.h5ad")
hlca = sc.read_h5ad("data/processed/HLCA_processed.h5ad")

print("\nData shapes:")
print(f"Tabula Sapiens lung: {ts_lung.shape}")
print(f"HLCA: {hlca.shape}")

# 设置必要的参数
ref_labels_key = 'cell_type'  # Tabula Sapiens的细胞类型标签
ref_batch_key = None  # 不使用批次校正
query_batch_key = None  # 不使用批次校正
unknown_celltype_label = 'unknown'
output_folder = 'trained_models'
os.makedirs(output_folder, exist_ok=True)

# 计算每个细胞类型的最小数量
min_celltype_size = np.min(ts_lung.obs.groupby(ref_labels_key).size())
n_samples_per_label = np.max((min_celltype_size, 500))

print("\nRunning PopV...")
# 使用Process_Query处理数据
combined = Process_Query(
    query_adata=hlca,
    ref_adata=ts_lung,
    ref_labels_key=ref_labels_key,
    ref_batch_key=ref_batch_key,
    cl_obo_folder=False,  # 不使用细胞本体论
    query_batch_key=query_batch_key,
    unknown_celltype_label=unknown_celltype_label,
    save_path_trained_models=output_folder,
    prediction_mode='retrain',  # 使用retrain模式
    n_samples_per_label=n_samples_per_label,
    hvg=4000  # 使用4000个高变异基因
).adata

# 运行PopV预测
pa.annotate_data(
    combined,
    methods=None,  # 使用默认的快速方法
    save_path=None,  # 不保存到文件
)

# 将预测结果添加到HLCA数据中
hlca.obs['predicted_cell_type'] = combined[combined.obs._dataset == 'query'].obs['popv_majority_vote_prediction']

# 保存结果
print("\nSaving results...")
os.makedirs("results", exist_ok=True)
hlca.write_h5ad("results/hlca_with_predictions.h5ad")
print("Done!")
