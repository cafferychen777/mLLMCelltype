#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from adjustText import adjust_text
import matplotlib.colors as mcolors
import re
from difflib import SequenceMatcher
import networkx as nx

# Set up the plotting style
sc.settings.set_figure_params(dpi=300, frameon=False)
plt.rcParams['figure.figsize'] = (21, 7)  # Increased width for three plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.labelsize'] = 16  # Size for axis labels
plt.rcParams['axes.titlesize'] = 20  # Size for subplot titles

# Nature Methods color palette
# Using a professional, color-blind friendly palette based on Nature publishing guidelines
# Main categories have distinct base colors, subcategories use different shades
nm_colors = {
    # T细胞和NK细胞 - 蓝色系 (使用ColorBrewer Blues)
    'Cytotoxic T cell': '#08519C',  # 深蓝色
    'CD8-positive, alpha-beta T cell': '#2171B5',  # 蓝色
    'CD4-positive, alpha-beta T cell': '#4292C6',  # 中蓝色
    'Naive T cells': '#6BAED6',  # 浅蓝色
    'Activated T cell': '#9ECAE1',  # 更浅蓝色
    'Proliferating T cells': '#C6DBEF',  # 最浅蓝色
    'T cell': '#DEEBF7',  # 极浅蓝色
    'naive thymus-derived CD8-positive': '#6BAED6',  # 浅蓝色 - 与Naive T cells相同
    'naive thymus-derived CD4-positive': '#6BAED6',  # 浅蓝色 - 与Naive T cells相同

    # B细胞和浆细胞 - 紫色系 (使用ColorBrewer Purples)
    'Natural killer cells': '#54278F',  # 深紫色
    'B cell': '#756BB1',  # 中紫色
    'Plasma cells': '#9E9AC8',  # 浅紫色

    # 树突状细胞和单核细胞 - 橙色系 (使用ColorBrewer Oranges)
    'CD1c-positive myeloid dendritic cell': '#A63603',  # 深橙色
    'Macrophages': '#E6550D',  # 橙红色
    'alveolar macrophage': '#FD8D3C',  # 橙色
    'Monocytes': '#FDAE6B',  # 浅橙色
    'non-classical monocyte': '#FDD0A2',  # 更浅橙色
    'intermediate monocyte': '#FEE6CE',  # 最浅橙色

    # 其他免疫细胞 - 粉色系 (使用ColorBrewer RdPu)
    'Neutrophils': '#AE017E',  # 深粉色
    'Mast cells': '#DD3497',  # 中粉色
    'basophil': '#F768A1',  # 浅粉色
    'myeloid leukocyte': '#FA9FB5',  # 更浅粉色
    'lymphocyte': '#FCC5C0',  # 淡粉色

    # 上皮细胞 AT1 - 深绿色系 (使用ColorBrewer Greens)
    'Alveolar epithelial cells type I': '#00441B',  # 深绿色 - AT1细胞
    'pulmonary alveolar type 1 cell': '#006D2C',  # 中深绿色 - AT1细胞
    'Alveolar Epithelial Type I Cells': '#238B45',  # 中绿色 - AT1细胞

    # 上皮细胞 AT2 - 浅绿色系
    'Alveolar epithelial cells type II': '#41AB5D',  # 中浅绿色 - AT2细胞
    'pulmonary alveolar type 2 cell': '#74C476',  # 浅绿色 - AT2细胞
    'Alveolar type II pneumocytes': '#A1D99B',  # 更浅绿色 - AT2细胞

    # 其他上皮细胞 - 青绿色系 (使用ColorBrewer GnBu)
    'Club cells': '#084081',  # 深青绿色
    'respiratory goblet cell': '#0868AC',  # 中青绿色
    'lung neuroendocrine cell': '#43A2CA',  # 浅青绿色
    'pulmonary ionocyte': '#7BCCC4',  # 更浅青绿色
    'serous cell of epithelium of bronchus': '#A8DDB5',  # 最浅青绿色
    'Ciliated Epithelial Cells': '#CCEBC5',  # 非常浅青绿色
    'Multiciliated Cells': '#E0F3DB',  # 最浅青绿色

    # 内皮细胞 - 红色系 (使用ColorBrewer Reds)
    'endothelial cell of lymphatic vessel': '#99000D',  # 深红色
    'Endothelial cell': '#CB181D',  # 红色
    'Capillary Endothelial Cell': '#EF3B2C',  # 中红色
    'Arterial endothelial cell': '#FB6A4A',  # 浅红色
    'Capillary endothelial cell': '#FC9272',  # 更浅红色
    'Lymphatic endothelial cells': '#FCBBA1',  # 粉红色
    'vein endothelial cell': '#FEE0D2',  # 浅粉红色

    # 间质细胞 - 棕色和黄色系 (使用ColorBrewer YlOrBr)
    'myofibroblast cell': '#8C2D04',  # 深棕色
    'Fibroblasts': '#D94801',  # 棕色
    'fibroblast': '#F16913',  # 中棕色
    'pulmonary interstitial fibroblast': '#FD8D3C',  # 浅棕色
    'adventitial cell': '#FDAE6B',  # 更浅棕色
    'alveolar adventitial fibroblast': '#FDD0A2',  # 黄棕色
    'Pericytes': '#FEE6CE',  # 浅黄色
    'vascular associated smooth muscle cell': '#FFF5EB',  # 最浅黄色

    # 其他细胞类型 - 灰色和青色系 (使用ColorBrewer PuBuGn)
    'mesothelial cell': '#1B7837',  # 深青色
    'mesothelial cell of pleura': '#5AAE61',  # 中青色
    'megakaryocyte': '#A6DBA0',  # 浅青色
    'Platelets': '#D9F0D3',  # 最浅青色
    'Extracellular Matrix': '#E5E5E5',  # 灰色
}

# Define semantic standardization functions
def normalize_cell_type(name):
    """标准化细胞类型名称"""
    # 转换为小写
    name = name.lower()

    # 删除常见的无信息后缀
    name = re.sub(r'\s+cells?$', '', name)
    name = re.sub(r'\s+populations?$', '', name)

    # 标准化常见的缩写和变体
    name = re.sub(r'\bdc\b', 'dendritic cell', name)
    name = re.sub(r'\bnk\b', 'natural killer', name)
    name = re.sub(r'\bmac\b', 'macrophage', name)
    name = re.sub(r'\bcd4\+\b', 'cd4-positive', name)
    name = re.sub(r'\bcd8\+\b', 'cd8-positive', name)
    name = re.sub(r'\bt-?cell\b', 't cell', name)
    name = re.sub(r'\bb-?cell\b', 'b cell', name)

    # 标准化肺泡上皮细胞类型
    name = re.sub(r'\bat1\b', 'alveolar type 1', name)
    name = re.sub(r'\bat2\b', 'alveolar type 2', name)
    name = re.sub(r'type i', 'type 1', name)
    name = re.sub(r'type ii', 'type 2', name)

    # 删除括号及其内容
    name = re.sub(r'\(.*?\)', '', name)

    # 删除多余的空格
    name = re.sub(r'\s+', ' ', name).strip()

    return name

def get_cell_type_tokens(name):
    """获取细胞类型名称的标记"""
    # 标准化名称
    name = normalize_cell_type(name)

    # 分割为标记
    tokens = name.split()

    # 过滤掉停用词
    stop_words = {'and', 'or', 'the', 'of', 'in', 'with', 'without', 'by', 'from', 'to', 'a', 'an'}
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def calculate_name_similarity(name1, name2):
    """计算两个细胞类型名称之间的相似度"""
    # 常见缩写映射
    abbreviations = {
        'nk': 'natural killer',
        'dc': 'dendritic cell',
        'aec': 'alveolar epithelial cell',
        'at1': 'alveolar type 1',
        'at2': 'alveolar type 2',
        'smc': 'smooth muscle cell',
        'ec': 'endothelial cell',
        'pdc': 'plasmacytoid dendritic cell',
        'mdc': 'myeloid dendritic cell',
        't cell': 't lymphocyte',
        'cd4': 't helper',
        'cd8': 'cytotoxic t'
    }

    # 使用序列匹配算法
    norm1 = normalize_cell_type(name1)
    norm2 = normalize_cell_type(name2)

    # 处理缩写
    for abbr, full in abbreviations.items():
        norm1 = norm1.replace(abbr, full)
        norm2 = norm2.replace(abbr, full)

    # 如果标准化后完全相同，返回1.0
    if norm1 == norm2:
        return 1.0

    # 计算序列相似度
    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()

    # 获取标记并计算交集比例
    tokens1 = set(get_cell_type_tokens(name1))
    tokens2 = set(get_cell_type_tokens(name2))

    # 处理缩写的标记
    expanded_tokens1 = set()
    expanded_tokens2 = set()

    for token in tokens1:
        expanded_tokens1.add(token)
        if token.lower() in abbreviations:
            for expanded_token in abbreviations[token.lower()].split():
                expanded_tokens1.add(expanded_token)

    for token in tokens2:
        expanded_tokens2.add(token)
        if token.lower() in abbreviations:
            for expanded_token in abbreviations[token.lower()].split():
                expanded_tokens2.add(expanded_token)

    # 使用扩展后的标记计算Jaccard相似度
    if not expanded_tokens1 or not expanded_tokens2:
        return seq_sim

    intersection = len(expanded_tokens1.intersection(expanded_tokens2))
    union = len(expanded_tokens1.union(expanded_tokens2))
    jaccard = intersection / union if union > 0 else 0

    # 特殊情况处理

    # 检查是否属于不同的主要细胞类型，如果是则降低相似度
    major_types = {
        'epithelial': ['epithelial', 'basal', 'goblet', 'ciliated', 'secretory', 'club', 'alveolar', 'pneumocyte'],
        'endothelial': ['endothelial', 'capillary', 'artery', 'vein', 'lymphatic'],
        'immune': ['t cell', 'b cell', 'nk', 'natural killer', 'macrophage', 'monocyte', 'dendritic', 'neutrophil', 'mast'],
        'stromal': ['fibroblast', 'smooth muscle', 'pericyte', 'adventitial']
    }

    # 检查两个名称是否属于不同的主要类型
    def get_major_type(name):
        name_lower = name.lower()
        for major_type, keywords in major_types.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return major_type
        return None

    major_type1 = get_major_type(norm1)
    major_type2 = get_major_type(norm2)

    # 如果两个名称属于不同的主要类型，则降低相似度
    if major_type1 and major_type2 and major_type1 != major_type2:
        return min(0.3, seq_sim, jaccard)  # 显著降低相似度

    # 特别处理一些容易混淆的类型
    # 分泌细胞与内皮细胞
    if (('secretory' in norm1.lower() or 'secreting' in norm1.lower()) and
        ('endothelial' in norm2.lower() or 'capillary' in norm2.lower() or 'artery' in norm2.lower() or 'vein' in norm2.lower())):
        return min(0.3, seq_sim, jaccard)

    if (('endothelial' in norm1.lower() or 'capillary' in norm1.lower() or 'artery' in norm1.lower() or 'vein' in norm1.lower()) and
        ('secretory' in norm2.lower() or 'secreting' in norm2.lower())):
        return min(0.3, seq_sim, jaccard)

    # 基底细胞与上皮细胞
    if ('basal' in norm1.lower() and 'epithelial' in norm2.lower() and 'basal' not in norm2.lower()):
        return min(0.5, seq_sim, jaccard)

    if ('epithelial' in norm1.lower() and 'basal' not in norm1.lower() and 'basal' in norm2.lower()):
        return min(0.5, seq_sim, jaccard)

    # NK cells 和 natural killer cell
    if ('nk' in norm1.lower() and 'natural killer' in norm2.lower()) or \
       ('natural killer' in norm1.lower() and 'nk' in norm2.lower()):
        return max(0.9, seq_sim, jaccard)

    # T细胞相关类型
    if (('cd4' in norm1.lower() or 'cd8' in norm1.lower() or 't cell' in norm1.lower() or 't lymphocyte' in norm1.lower()) and
        ('cd4' in norm2.lower() or 'cd8' in norm2.lower() or 't cell' in norm2.lower() or 't lymphocyte' in norm2.lower())):
        # 检查是否是相同类型的T细胞
        if ('cd4' in norm1.lower() and 'cd4' in norm2.lower()) or \
           ('cd8' in norm1.lower() and 'cd8' in norm2.lower()) or \
           ('naive' in norm1.lower() and 'naive' in norm2.lower()) or \
           ('memory' in norm1.lower() and 'memory' in norm2.lower()) or \
           ('cytotoxic' in norm1.lower() and 'cytotoxic' in norm2.lower()) or \
           ('helper' in norm1.lower() and 'helper' in norm2.lower()) or \
           ('regulatory' in norm1.lower() and 'regulatory' in norm2.lower()):
            return max(0.85, seq_sim, jaccard)
        else:
            return max(0.7, seq_sim, jaccard)

    # 肺泡上皮细胞类型 - 强化区分AT1和AT2
    if (('alveolar' in norm1.lower() or 'pneumocyte' in norm1.lower()) and
        ('alveolar' in norm2.lower() or 'pneumocyte' in norm2.lower())):

        # 检查是否是AT1细胞
        is_at1_1 = ('type 1' in norm1.lower() or 'type i' in norm1.lower() or 'at1' in norm1.lower())
        is_at1_2 = ('type 1' in norm2.lower() or 'type i' in norm2.lower() or 'at1' in norm2.lower())

        # 检查是否是AT2细胞
        is_at2_1 = ('type 2' in norm1.lower() or 'type ii' in norm1.lower() or 'at2' in norm1.lower())
        is_at2_2 = ('type 2' in norm2.lower() or 'type ii' in norm2.lower() or 'at2' in norm2.lower())

        # 如果都是AT1或都是AT2，则高度相似
        if (is_at1_1 and is_at1_2) or (is_at2_1 and is_at2_2):
            return max(0.9, seq_sim, jaccard)
        # 如果一个是AT1，一个是AT2，则显著降低相似度
        elif (is_at1_1 and is_at2_2) or (is_at2_1 and is_at1_2):
            return min(0.3, seq_sim, jaccard)
        # 如果一个明确指定了类型，另一个没有，则中等相似度
        elif (is_at1_1 or is_at2_1) and not (is_at1_2 or is_at2_2):
            return min(0.5, seq_sim, jaccard)
        elif (is_at1_2 or is_at2_2) and not (is_at1_1 or is_at2_1):
            return min(0.5, seq_sim, jaccard)
        # 如果都没有明确指定类型，则中等相似度
        else:
            return max(0.7, seq_sim, jaccard)

    # 内皮细胞类型
    if ('endothelial' in norm1.lower() and 'endothelial' in norm2.lower()):
        # 检查是否是相同类型的内皮细胞
        if ('lymphatic' in norm1.lower() and 'lymphatic' in norm2.lower()):
            return max(0.9, seq_sim, jaccard)
        elif ('artery' in norm1.lower() and 'artery' in norm2.lower()):
            return max(0.9, seq_sim, jaccard)
        elif ('capillary' in norm1.lower() and 'capillary' in norm2.lower()):
            return max(0.9, seq_sim, jaccard)
        elif ('vein' in norm1.lower() and 'vein' in norm2.lower()):
            return max(0.9, seq_sim, jaccard)
        else:
            return max(0.8, seq_sim, jaccard)

    # 上皮细胞类型
    if ('epithelial' in norm1.lower() and 'epithelial' in norm2.lower()):
        return max(0.8, seq_sim, jaccard)

    # 基底细胞
    if ('basal' in norm1.lower() and 'basal' in norm2.lower()):
        return max(0.85, seq_sim, jaccard)

    # 杯状细胞
    if ('goblet' in norm1.lower() and 'goblet' in norm2.lower()):
        return max(0.85, seq_sim, jaccard)

    # 纤毛细胞类型
    if ('ciliated' in norm1.lower() and 'ciliated' in norm2.lower()):
        return max(0.85, seq_sim, jaccard)

    # 纤维细胞类型
    if ('fibroblast' in norm1.lower() and 'fibroblast' in norm2.lower()):
        return max(0.85, seq_sim, jaccard)

    # 树突状细胞类型
    if ('dendritic' in norm1.lower() and 'dendritic' in norm2.lower()):
        if (('myeloid' in norm1.lower() and 'myeloid' in norm2.lower()) or
            ('plasmacytoid' in norm1.lower() and 'plasmacytoid' in norm2.lower())):
            return max(0.9, seq_sim, jaccard)
        else:
            return max(0.8, seq_sim, jaccard)

    # 平滑肌细胞类型
    if ('smooth muscle' in norm1.lower() and 'smooth muscle' in norm2.lower()):
        return max(0.85, seq_sim, jaccard)

    # 分泌细胞类型
    if (('secretory' in norm1.lower() or 'secreting' in norm1.lower() or 'serous' in norm1.lower()) and
        ('secretory' in norm2.lower() or 'secreting' in norm2.lower() or 'serous' in norm2.lower())):
        return max(0.85, seq_sim, jaccard)

    # 综合考虑序列相似度和Jaccard相似度
    return max(seq_sim, jaccard)

def build_similarity_groups(cell_types, threshold=0.8):
    """基于名称相似度构建细胞类型组"""
    # 创建图，节点是细胞类型，边是相似度高于阈值的关系
    G = nx.Graph()

    # 添加所有节点
    for ct in cell_types:
        G.add_node(ct)

    # 添加边
    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types[i+1:], i+1):
            # 特殊处理NK cells和natural killer cell
            if (('nk' in ct1.lower() and 'natural killer' in ct2.lower()) or
                ('natural killer' in ct1.lower() and 'nk' in ct2.lower())):
                G.add_edge(ct1, ct2, weight=0.9)
                continue

            # 特殊处理内皮细胞和分泌细胞，避免混淆
            if (('secretory' in ct1.lower() or 'secreting' in ct1.lower()) and
                ('endothelial' in ct2.lower() or 'capillary' in ct2.lower() or 'artery' in ct2.lower() or 'vein' in ct2.lower())):
                continue

            if (('endothelial' in ct1.lower() or 'capillary' in ct1.lower() or 'artery' in ct1.lower() or 'vein' in ct1.lower()) and
                ('secretory' in ct2.lower() or 'secreting' in ct2.lower())):
                continue

            # 特殊处理AT1和AT2肺泡上皮细胞，避免混淆
            if (('type 1' in ct1.lower() or 'type i' in ct1.lower() or 'at1' in ct1.lower()) and
                ('type 2' in ct2.lower() or 'type ii' in ct2.lower() or 'at2' in ct2.lower())):
                continue

            if (('type 2' in ct1.lower() or 'type ii' in ct1.lower() or 'at2' in ct1.lower()) and
                ('type 1' in ct2.lower() or 'type i' in ct2.lower() or 'at1' in ct2.lower())):
                continue

            sim = calculate_name_similarity(ct1, ct2)
            if sim >= threshold:
                G.add_edge(ct1, ct2, weight=sim)

    # 找出连通分量（即相似的细胞类型组）
    groups = list(nx.connected_components(G))

    # 为每个组选择代表性名称
    group_representatives = {}
    for i, group in enumerate(groups):
        # 选择组内最具代表性的名称作为代表
        # 优先选择更具描述性的名称（通常是较长的名称）
        if len(group) == 1:
            # 如果只有一个成员，直接使用它
            representative = list(group)[0]
        else:
            # 对于多成员组，选择最具代表性的名称
            candidates = list(group)

            # 肺泡上皮细胞特殊处理：优先选择明确指定了类型的名称
            at1_candidates = [c for c in candidates if ('type 1' in c.lower() or 'type i' in c.lower() or 'at1' in c.lower())]
            at2_candidates = [c for c in candidates if ('type 2' in c.lower() or 'type ii' in c.lower() or 'at2' in c.lower())]

            if at1_candidates:
                # 如果有AT1细胞，选择最具描述性的AT1名称
                representative = max(at1_candidates, key=len)
                # 确保AT1细胞有正确的标准化名称
                if 'alveolar' in representative.lower() or 'pneumocyte' in representative.lower():
                    representative = 'Alveolar epithelial cells type I'
            elif at2_candidates:
                # 如果有AT2细胞，选择最具描述性的AT2名称
                representative = max(at2_candidates, key=len)
                # 确保AT2细胞有正确的标准化名称
                if 'alveolar' in representative.lower() or 'pneumocyte' in representative.lower():
                    representative = 'Alveolar epithelial cells type II'
            else:
                # 其他情况，优先选择某些关键词的名称
                priority_keywords = ['cell', 'type', 'specific']
                priority_candidates = [c for c in candidates if any(k in c.lower() for k in priority_keywords)]

                if priority_candidates:
                    # 如果有包含优先关键词的候选项，从中选择
                    representative = max(priority_candidates, key=len)
                else:
                    # 否则选择最长的名称
                    representative = max(candidates, key=len)

        for ct in group:
            # 特殊处理AT1和AT2细胞
            if ('type 1' in ct.lower() or 'type i' in ct.lower() or 'at1' in ct.lower()) and ('alveolar' in ct.lower() or 'pneumocyte' in ct.lower()):
                group_representatives[ct] = 'Alveolar epithelial cells type I'
            elif ('type 2' in ct.lower() or 'type ii' in ct.lower() or 'at2' in ct.lower()) and ('alveolar' in ct.lower() or 'pneumocyte' in ct.lower()):
                group_representatives[ct] = 'Alveolar epithelial cells type II'
            else:
                group_representatives[ct] = representative

    return group_representatives

# Function to calculate improved centroids for each cluster
def calculate_centroids(coords, labels):
    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            points = coords[mask]

            # 1. 计算中位数 - 减少离群点影响
            median_centroid = np.median(points, axis=0)

            # 2. 计算密度峰值
            # 使用KDE估计密度
            if len(points) > 10:  # 只有当点数足够多时才使用KDE
                try:
                    from scipy.stats import gaussian_kde
                    from scipy.optimize import minimize

                    # 使用高斯核密度估计
                    kde = gaussian_kde(points.T)

                    # 寻找密度最大的点
                    def neg_density(p):
                        return -kde([p[0], p[1]])[0]

                    # 从中位数开始优化
                    density_peak = minimize(neg_density, median_centroid, method='L-BFGS-B').x

                    # 3. 计算加权平均
                    # 根据到密度峰值的距离计算权重
                    distances = np.sqrt(np.sum((points - density_peak)**2, axis=1))
                    weights = np.exp(-distances / distances.mean())  # 距离越近权重越大
                    weighted_centroid = np.average(points, weights=weights, axis=0)

                    # 4. 综合三种方法
                    # 根据点的数量和分布调整权重
                    if len(points) > 100:
                        # 大群体优先使用密度峰值
                        final_centroid = density_peak * 0.6 + weighted_centroid * 0.3 + median_centroid * 0.1
                    else:
                        # 小群体优先使用加权平均和中位数
                        final_centroid = density_peak * 0.2 + weighted_centroid * 0.4 + median_centroid * 0.4

                    centroids[label] = final_centroid
                except:
                    # 如果KDE失败，回退到中位数
                    centroids[label] = median_centroid
            else:
                # 点数太少时使用中位数
                centroids[label] = median_centroid

    return centroids

# Function to add text (no bold, consistent with supplementary_figure8)
def add_text(ax, x, y, text, fontsize=10):
    # Add text with black color, no bold
    text_obj = ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
                      color='black', zorder=5)

    return text_obj

# Load the data
print("Loading the data...")
adata = sc.read_h5ad('data/processed/LCA_with_umap.h5ad')
results_df = pd.read_csv('results/benchmark/reference_comparison/2_evaluation/LCA_results.csv')
popv_df = pd.read_csv('results/benchmark/popv_comparison/2_evaluation/LCA_popv_fast_results.csv')

print("Processing the data...")

# Create output directory if it doesn't exist
output_dir = 'manuscript/figures'
os.makedirs(output_dir, exist_ok=True)

# Create name standardization mapping for human cell types
name_standardization = {
    'myeloid dendritic cell, human': 'myeloid dendritic cell',
    'plasmacytoid dendritic cell, human': 'plasmacytoid dendritic cell',
    'effector memory CD4-positive, alpha-beta T cell': 'effector memory CD4-positive',
    'effector memory CD8-positive, alpha-beta T cell': 'effector memory CD8-positive',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'naive thymus-derived CD4-positive',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'naive thymus-derived CD8-positive'
}

# Create a mapping from reference_name to final_consensus
ref_to_consensus = dict(zip(results_df['reference_name'], results_df['final_consensus']))

# Apply initial name standardization
adata.obs['cell_type_norm'] = adata.obs['cell_type'].map(lambda x: name_standardization.get(x, x))
consensus_norm = [ref_to_consensus.get(name_standardization.get(x, x), "Unknown") for x in adata.obs['cell_type']]
adata.obs['final_consensus_norm'] = consensus_norm
popv_df['popv_prediction_norm'] = popv_df['popv_prediction']
adata.obs['popv_prediction_norm'] = popv_df.set_index('index')['popv_prediction_norm']

# Collect all unique cell type names from all annotation methods
all_cell_types = set(adata.obs['cell_type_norm'].unique()).union(
    set(adata.obs['final_consensus_norm'].unique())).union(
    set(adata.obs['popv_prediction_norm'].dropna().unique()))

# Remove 'Unknown' if present
if 'Unknown' in all_cell_types:
    all_cell_types.remove('Unknown')

# Build semantic similarity groups
print("Building semantic similarity groups...")
semantic_groups = build_similarity_groups(list(all_cell_types), threshold=0.8)  # 提高阈值以减少过度聚类

# Apply semantic standardization
adata.obs['cell_type_std'] = adata.obs['cell_type_norm'].map(lambda x: semantic_groups.get(x, x))
adata.obs['final_consensus_std'] = adata.obs['final_consensus_norm'].map(lambda x: semantic_groups.get(x, x))
adata.obs['popv_prediction_std'] = adata.obs['popv_prediction_norm'].map(lambda x: semantic_groups.get(x, x) if pd.notna(x) else x)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

# Set figure style for each subplot
for ax in [ax1, ax2, ax3]:
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

# Get UMAP coordinates
umap_coords = adata.obsm['X_umap']

# Function to get color for cell type
def get_cell_type_color(cell_type):
    if cell_type in nm_colors:
        return nm_colors[cell_type]
    # Generate a random but consistent color for unknown types
    hash_val = hash(cell_type) % 100000
    return plt.cm.tab20(hash_val / 100000)

# Plot 1: Original annotations
cell_types = adata.obs['cell_type_std']
for ct in np.unique(cell_types):
    mask = cell_types == ct
    color = get_cell_type_color(ct)
    ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               c=[color], s=5, alpha=0.7, rasterized=True)

# Add text labels for original annotations
centroids = calculate_centroids(umap_coords, cell_types)
texts = []
for ct, centroid in centroids.items():
    text_obj = add_text(ax1, centroid[0], centroid[1], ct, fontsize=12)
    texts.append(text_obj)

# Adjust text positions to avoid overlap
adjust_text(texts, ax=ax1)
ax1.set_title('Reference', fontsize=20)

# Plot 2: Consensus annotations
consensus_types = adata.obs['final_consensus_std']
for ct in np.unique(consensus_types):
    mask = consensus_types == ct
    color = get_cell_type_color(ct)
    ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               c=[color], s=5, alpha=0.7, rasterized=True)

# Add text labels for consensus annotations
centroids = calculate_centroids(umap_coords, consensus_types)
texts = []
for ct, centroid in centroids.items():
    text_obj = add_text(ax2, centroid[0], centroid[1], ct, fontsize=12)
    texts.append(text_obj)

# Adjust text positions to avoid overlap
adjust_text(texts, ax=ax2)
ax2.set_title('Our Framework', fontsize=20)

# Plot 3: PopV predictions
popv_types = adata.obs['popv_prediction_std']
for ct in np.unique(popv_types):
    mask = popv_types == ct
    color = get_cell_type_color(ct)
    ax3.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               c=[color], s=5, alpha=0.7, rasterized=True)

# Add text labels for PopV predictions
centroids = calculate_centroids(umap_coords, popv_types)
texts = []
for ct, centroid in centroids.items():
    text_obj = add_text(ax3, centroid[0], centroid[1], ct, fontsize=12)
    texts.append(text_obj)

# Adjust text positions to avoid overlap
adjust_text(texts, ax=ax3)
ax3.set_title('popV', fontsize=20)

# Customize plots
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('UMAP1', fontsize=16, labelpad=10)
    ax.set_ylabel('UMAP2', fontsize=16, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=14)

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'supplementary_figure3.pdf'),
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(output_dir, 'supplementary_figure3.png'),
            dpi=300, bbox_inches='tight', format='png')
plt.close()

print(f"Visualization saved to: {os.path.join(output_dir, 'supplementary_figure3.pdf')}")
print(f"Visualization saved to: {os.path.join(output_dir, 'supplementary_figure3.png')}")

# Save semantic groups to CSV for reference
semantic_groups_df = pd.DataFrame([(cell_type, group_rep)
                                  for cell_type, group_rep in semantic_groups.items()],
                                 columns=['Original', 'Standardized'])
semantic_groups_df.to_csv(os.path.join(output_dir, 'semantic_standardization_groups.csv'), index=False)
print(f"Semantic standardization groups saved to: {os.path.join(output_dir, 'semantic_standardization_groups.csv')}")

# Print statistics about standardization
print("\nNumber of unique cell types before and after semantic standardization:")
print(f"Original (before): {len(adata.obs['cell_type_norm'].unique())}")
print(f"Original (after): {len(adata.obs['cell_type_std'].unique())}")
print(f"Consensus (before): {len(adata.obs['final_consensus_norm'].unique())}")
print(f"Consensus (after): {len(adata.obs['final_consensus_std'].unique())}")
print(f"PopV (before): {len(adata.obs['popv_prediction_norm'].dropna().unique())}")
print(f"PopV (after): {len(adata.obs['popv_prediction_std'].dropna().unique())}")

# Print the semantic groups
print("\nSemantic similarity groups:")
group_to_members = {}
for cell_type, group_rep in semantic_groups.items():
    if group_rep not in group_to_members:
        group_to_members[group_rep] = []
    group_to_members[group_rep].append(cell_type)

# Print groups with more than one member
print(f"Total groups: {len(group_to_members)}")
print("Groups with multiple members:")
for group_rep, members in sorted(group_to_members.items(), key=lambda x: len(x[1]), reverse=True):
    if len(members) > 1:
        print(f"  Group '{group_rep}' has {len(members)} members:")
        for member in sorted(members):
            print(f"    - {member}")

# Print statistics about standardization
print("\nNumber of unique cell types after standardization:")
print(f"Original: {len(np.unique(adata.obs['cell_type_std']))}")
print(f"Consensus: {len(np.unique(adata.obs['final_consensus_std']))}")
print(f"PopV: {len(np.unique(adata.obs['popv_prediction_std']))}")

# Print the standardized names
print("\nStandardized cell types in each annotation:")
print("\nOriginal:")
print(sorted(np.unique(adata.obs['cell_type_std'])))
print("\nConsensus:")
print(sorted(np.unique(adata.obs['final_consensus_std'])))
print("\nPopV:")
print(sorted(np.unique(adata.obs['popv_prediction_std'])))

# 确保所有标准化后的细胞类型都有颜色
for cell_type in all_cell_types:
    if cell_type not in nm_colors:
        # 根据细胞类型分配默认颜色
        if 'T cell' in cell_type or 'CD4' in cell_type or 'CD8' in cell_type:
            nm_colors[cell_type] = '#9ECAE1'  # 浅蓝色 (T细胞)
        elif 'B cell' in cell_type or 'plasma' in cell_type.lower():
            nm_colors[cell_type] = '#9E9AC8'  # 浅紫色 (B细胞)
        elif 'NK' in cell_type or 'natural killer' in cell_type.lower():
            nm_colors[cell_type] = '#BCBDDC'  # 最浅紫色 (NK细胞)
        elif 'macrophage' in cell_type.lower() or 'monocyte' in cell_type.lower():
            nm_colors[cell_type] = '#FDAE6B'  # 浅橙色 (巨噬细胞/单核细胞)
        elif 'dendritic' in cell_type.lower():
            nm_colors[cell_type] = '#F768A1'  # 粉色 (树突状细胞)
        elif 'alveolar type 1' in cell_type.lower() or 'at1' in cell_type.lower() or 'type i' in cell_type.lower():
            nm_colors[cell_type] = '#00441B'  # 深绿色 (AT1细胞)
        elif 'alveolar type 2' in cell_type.lower() or 'at2' in cell_type.lower() or 'type ii' in cell_type.lower():
            nm_colors[cell_type] = '#74C476'  # 浅绿色 (AT2细胞)
        elif 'epithelial' in cell_type.lower():
            nm_colors[cell_type] = '#A1D99B'  # 浅绿色 (上皮细胞)
        elif 'endothelial' in cell_type.lower() or 'capillary' in cell_type.lower():
            nm_colors[cell_type] = '#FC9272'  # 浅红色 (内皮细胞)
        elif 'fibroblast' in cell_type.lower() or 'stromal' in cell_type.lower():
            nm_colors[cell_type] = '#FEC44F'  # 浅黄棕色 (纤维细胞)
        else:
            nm_colors[cell_type] = '#CCCCCC'  # 灰色 (其他)

# 分析标准化后的唯一配对
print("\n分析标准化后的唯一配对:")
# 收集所有标准化后的唯一细胞类型
std_cell_types = set(adata.obs['cell_type_std'].unique()).union(
    set(adata.obs['final_consensus_std'].unique())).union(
    set(adata.obs['popv_prediction_std'].dropna().unique()))

# 创建从原始到标准化的映射
original_to_std = {}
for original, std in semantic_groups.items():
    if std not in original_to_std:
        original_to_std[std] = []
    original_to_std[std].append(original)

# 输出每个标准化类型对应的原始类型
for std_type in sorted(original_to_std.keys()):
    originals = original_to_std[std_type]
    if len(originals) > 1:  # 只输出有多个原始类型的标准化类型
        print(f"标准化类型 '{std_type}' 对应的原始类型:")
        for orig in sorted(originals):
            print(f"  - {orig}")

# 特别检查NK cells和natural killer cell
print("\n特别检查NK cells和natural killer cell:")
nk_related = [ct for ct in all_cell_types if 'nk' in ct.lower() or 'natural killer' in ct.lower()]
for ct in sorted(nk_related):
    std = semantic_groups.get(ct, ct)
    print(f"'{ct}' 标准化为 '{std}'")

# 输出标准化后的唯一配对
unique_pairs = {}
for og_type in adata.obs['cell_type_norm'].unique():
    std_type = semantic_groups.get(og_type, og_type)
    if std_type not in unique_pairs:
        unique_pairs[std_type] = set()
    unique_pairs[std_type].add(og_type)

for cons_type in adata.obs['final_consensus_norm'].unique():
    std_type = semantic_groups.get(cons_type, cons_type)
    if std_type not in unique_pairs:
        unique_pairs[std_type] = set()
    unique_pairs[std_type].add(cons_type)

for popv_type in adata.obs['popv_prediction_norm'].dropna().unique():
    std_type = semantic_groups.get(popv_type, popv_type)
    if std_type not in unique_pairs:
        unique_pairs[std_type] = set()
    unique_pairs[std_type].add(popv_type)

# 将唯一配对保存到CSV
unique_pairs_data = []
for std_type, orig_types in unique_pairs.items():
    for orig_type in orig_types:
        unique_pairs_data.append({
            'Standardized': std_type,
            'Original': orig_type
        })

unique_pairs_df = pd.DataFrame(unique_pairs_data)
unique_pairs_df.to_csv(os.path.join(output_dir, 'unique_standardization_pairs.csv'), index=False)
print(f"\n唯一标准化配对保存到: {os.path.join(output_dir, 'unique_standardization_pairs.csv')}")

# Print statistics about standardization
print("\nNumber of unique cell types before and after semantic standardization:")
print(f"Original (before): {len(adata.obs['cell_type_norm'].unique())}")
print(f"Original (after): {len(adata.obs['cell_type_std'].unique())}")
print(f"Consensus (before): {len(adata.obs['final_consensus_norm'].unique())}")
print(f"Consensus (after): {len(adata.obs['final_consensus_std'].unique())}")
print(f"PopV (before): {len(adata.obs['popv_prediction_norm'].dropna().unique())}")
print(f"PopV (after): {len(adata.obs['popv_prediction_std'].dropna().unique())}")

# Print the semantic groups
print("\nSemantic similarity groups:")
group_to_members = {}
for cell_type, group_rep in semantic_groups.items():
    if group_rep not in group_to_members:
        group_to_members[group_rep] = []
    group_to_members[group_rep].append(cell_type)

# Print groups with more than one member
print(f"Total groups: {len(group_to_members)}")
print("Groups with multiple members:")
for group_rep, members in sorted(group_to_members.items(), key=lambda x: len(x[1]), reverse=True):
    if len(members) > 1:
        print(f"  Group '{group_rep}' has {len(members)} members:")
        for member in sorted(members):
            print(f"    - {member}")

# Print statistics about standardization
print("\nNumber of unique cell types after standardization:")
print(f"Original: {len(np.unique(adata.obs['cell_type_std']))}")
print(f"Consensus: {len(np.unique(adata.obs['final_consensus_std']))}")
print(f"PopV: {len(np.unique(adata.obs['popv_prediction_std']))}")

# Print the standardized names
print("\nStandardized cell types in each annotation:")
print("\nOriginal:")
print(sorted(np.unique(adata.obs['cell_type_std'])))
print("\nConsensus:")
print(sorted(np.unique(adata.obs['final_consensus_std'])))
print("\nPopV:")
print(sorted(np.unique(adata.obs['popv_prediction_std'])))
