#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：使用Schiller_2020.h5ad数据集测试LLMCelltype的迭代讨论功能
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入LLMCelltype的函数
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

# 加载环境变量
load_dotenv()

def run_schiller_dataset_test():
    """使用Schiller_2020数据集测试LLMCelltype的迭代讨论功能"""
    
    print("\n=== Schiller 2020 Dataset Test with Iterative Discussion ===\n")
    
    # 加载数据集
    print("Loading Schiller 2020 dataset...")
    data_path = "/Users/apple/Research/LLMCelltype/data/raw/Schiller_2020.h5ad"
    adata = sc.read_h5ad(data_path)
    
    print(f"Dataset loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # 检查数据集中是否已有聚类结果
    if 'leiden' not in adata.obs.columns:
        print("\nPreprocessing dataset...")
        # 标准化数据
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # 选择高变异基因
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        print(f"Selected {sum(adata.var.highly_variable)} highly variable genes")
        
        # 主成分分析
        sc.pp.pca(adata, svd_solver='arpack')
        
        # 计算邻居图
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # 使用Leiden算法进行聚类
        print("Performing clustering...")
        sc.tl.leiden(adata, resolution=0.8)
        print(f"Found {len(adata.obs['leiden'].unique())} clusters")
        
        # 计算UMAP嵌入以便可视化
        sc.tl.umap(adata)
        
        # 选择一部分细胞进行注释（为了加快处理速度）
        print("Sampling cells for annotation...")
        # 从每个聚类中随机选择一些细胞
        np.random.seed(42)  # 设置随机种子以确保可重复性
        
        # 如果聚类数量过多，只选择前10个聚类进行测试
        clusters_to_use = sorted(adata.obs['leiden'].unique())[:10]
        print(f"Using clusters: {', '.join(clusters_to_use)}")
        
        # 创建一个新的AnnData对象，只包含选定的聚类
        adata_subset = adata[adata.obs['leiden'].isin(clusters_to_use)].copy()
        print(f"Subset created with {adata_subset.shape[0]} cells from {len(clusters_to_use)} clusters")
        
        # 使用子集进行后续分析
        adata = adata_subset
    
    # 获取API密钥
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # 使用的模型
    models = ['gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro', 'qwen2.5-72b-instruct']
    print(f"Using models: {', '.join(models)}")
    
    # 不需要创建实例，直接使用函数
    
    # 提取标记基因
    print("\nExtracting marker genes...")
    
    # 创建基因ID到基因名称的映射
    gene_id_to_name = dict(zip(adata.var.index, adata.var['original_gene_symbols']))
    
    # 计算每个聚类的标记基因
    print("Computing marker genes...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # 为每个聚类提取前10个标记基因，并转换为基因名称
    marker_genes = {}
    for cluster in adata.obs['leiden'].unique():
        try:
            # 获取基因ID
            gene_ids = sc.get.rank_genes_groups_df(adata, group=cluster).head(10)['names'].tolist()
            # 将基因ID转换为基因名称
            gene_names = [gene_id_to_name.get(gene_id, gene_id) for gene_id in gene_ids]
            marker_genes[cluster] = gene_names
        except Exception as e:
            print(f"Error extracting marker genes for cluster {cluster}: {e}")
            # 如果出错，使用一些基本的标记基因
            marker_genes[cluster] = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'CD14', 'FCGR3A', 'FCGR3B']
    
    # 打印标记基因
    print("\nMarker genes for each cluster:")
    for cluster, genes in marker_genes.items():
        print(f"  Cluster {cluster}: {', '.join(genes[:5])}...")
    
    # 运行共识注释
    print("\nRunning consensus annotation with iterative discussion...")
    result = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species='human',
        tissue='lung',
        models=models,
        api_keys=api_keys,
        consensus_threshold=0.9,  # 高共识阈值，使其更难达成共识
        max_discussion_rounds=3,  # 最多3轮讨论
        use_cache=True,
        verbose=True
    )
    
    # 打印结果摘要
    print_consensus_summary(result)
    
    # 打印详细的讨论日志
    controversial_clusters = result.get("controversial_clusters", [])
    if controversial_clusters:
        print("\n=== Detailed Discussion Logs for Controversial Clusters ===\n")
        for cluster in controversial_clusters:
            if "discussion_logs" in result and cluster in result["discussion_logs"]:
                print(f"\n{'='*80}")
                print(f"Cluster {cluster} Discussion:")
                print(f"{'='*80}")
                
                # 获取完整的讨论日志
                logs = result["discussion_logs"][cluster]
                
                # 直接打印完整的讨论日志
                if isinstance(logs, list):
                    print("\n".join(logs))
                else:
                    print(logs)
                
                # 尝试提取并显示共识指标
                logs_text = logs if isinstance(logs, str) else "\n".join(logs)
                
                print("\n=== Extracted Consensus Metrics ===\n")
                
                # 提取初始指标
                initial_metrics_found = False
                if "Initial votes" in logs_text:
                    initial_section = logs_text.split("Round 1")[0] if "Round 1" in logs_text else logs_text
                    print("Initial metrics:")
                    for line in initial_section.split("\n"):
                        if "Consensus Proportion (CP):" in line or "Shannon Entropy (H):" in line:
                            print(f"  {line.strip()}")
                            initial_metrics_found = True
                    
                    if not initial_metrics_found:
                        print("  No initial metrics found")
                
                # 提取每轮讨论的指标
                rounds = logs_text.split("Round ")
                for i in range(1, len(rounds)):
                    round_text = rounds[i]
                    round_metrics_found = False
                    
                    print(f"\nRound {i} metrics:")
                    # 提取提议的细胞类型
                    for line in round_text.split("\n"):
                        if "Proposed cell type:" in line:
                            print(f"  {line.strip()}")
                            round_metrics_found = True
                        elif "Consensus Proportion (CP):" in line or "Shannon Entropy (H):" in line:
                            print(f"  {line.strip()}")
                            round_metrics_found = True
                    
                    if not round_metrics_found:
                        print("  No metrics found for this round")
    
    # 返回结果
    return result

if __name__ == "__main__":
    run_schiller_dataset_test()
