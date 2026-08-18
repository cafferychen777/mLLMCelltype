import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import time
from tqdm import tqdm
import pickle
import re
from typing import Dict, Set, Optional, List, Tuple

class CellOntologyMatcher:
    def __init__(self, cache_file=None):
        """
        初始化匹配器

        Args:
            cache_file: 缓存文件路径，如果存在则加载，否则创建新的
        """
        self.cache_file = cache_file or "results/evaluation/ontology_cache.pkl"
        self.load_cache()

        # 初始化同义词映射
        self.synonyms = {
            'alpha beta': 'alpha-beta',
            'gamma delta': 'gamma-delta',
            'cd4 positive': 'CD4-positive',
            'cd8 positive': 'CD8-positive',
            'cd4positive': 'CD4-positive',
            'cd8positive': 'CD8-positive',
            't cell': 'T cell',
            'b cell': 'B cell',
            'nk cell': 'natural killer cell',
            'dc': 'dendritic cell',
            'tec': 'thymic epithelial cell',
            'mtec': 'medullary thymic epithelial cell',
            'ctec': 'cortical thymic epithelial cell',
            'double positive': 'double-positive',
            'double negative': 'double-negative',
            'single positive': 'single-positive',
            'dp': 'double-positive',
            'dn': 'double-negative',
            'sp': 'single-positive'
        }

    def load_cache(self):
        """加载或初始化缓存"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Loaded cache with {len(self.cache)} entries")
        except FileNotFoundError:
            self.cache = {}
            print("Created new cache")

    def save_cache(self):
        """保存缓存到文件"""
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"Saved cache with {len(self.cache)} entries")

    def normalize_cell_type(self, cell_type: str) -> str:
        """
        标准化细胞类型名称

        Args:
            cell_type: 原始细胞类型名称

        Returns:
            标准化后的名称
        """
        if not cell_type:
            return cell_type

        # 转换为小写进行处理
        cell_type = cell_type.lower()

        # 基本清理
        cell_type = re.sub(r'[^\w\s-]', ' ', cell_type)
        cell_type = ' '.join(cell_type.split())

        # 应用同义词映射
        for old, new in self.synonyms.items():
            cell_type = re.sub(r'\b' + old + r'\b', new, cell_type)

        # 特殊处理：确保某些词的大小写正确
        words_to_capitalize = ['t', 'b', 'nk', 'cd4', 'cd8']
        for word in words_to_capitalize:
            cell_type = re.sub(r'\b' + word + r'\b', word.upper(), cell_type)

        return cell_type

    def query_ontology(self, cell_type: str) -> Optional[Dict]:
        """
        查询 Cell Ontology

        Args:
            cell_type: 细胞类型名称

        Returns:
            包含 ID、IRI 等信息的字典，如果未找到则返回 None
        """
        cache_key = f"term_{cell_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        normalized_type = self.normalize_cell_type(cell_type)

        # 首先尝试精确匹配
        params = {
            'q': normalized_type,
            'ontology': 'cl',
            'exact': 'true',
            'queryFields': 'label,synonym'
        }

        try:
            response = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=params)
            if response.status_code == 200:
                data = response.json()
                if data['response']['numFound'] > 0:
                    # 找到CL本体中的术语
                    cl_term = None
                    for doc in data['response']['docs']:
                        if doc['ontology_prefix'] == 'CL':
                            cl_term = doc
                            break

                    if cl_term:
                        result = {
                            'id': cl_term['obo_id'],
                            'iri': cl_term['iri'],
                            'label': cl_term['label']
                        }
                        self.cache[cache_key] = result
                        return result

            # 如果精确匹配失败，尝试模糊匹配
            params['exact'] = 'false'
            response = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=params)
            if response.status_code == 200:
                data = response.json()
                if data['response']['numFound'] > 0:
                    # 找到CL本体中的术语
                    cl_term = None
                    for doc in data['response']['docs']:
                        if doc['ontology_prefix'] == 'CL':
                            cl_term = doc
                            break

                    if cl_term:
                        result = {
                            'id': cl_term['obo_id'],
                            'iri': cl_term['iri'],
                            'label': cl_term['label']
                        }
                        self.cache[cache_key] = result
                        return result

        except Exception as e:
            print(f"Error querying ontology for {cell_type}: {e}")

        self.cache[cache_key] = None
        return None

    def get_term_relations(self, term_info: Dict) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        """
        获取术语的关系

        Args:
            term_info: 术语信息字典

        Returns:
            (parents, children, ancestors, descendants) 的集合元组
        """
        if not term_info:
            return set(), set(), set(), set()

        cache_key = f"relations_{term_info['id']}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        encoded_iri = requests.utils.quote(term_info['iri'], safe='')

        # 获取术语详情
        try:
            response = requests.get(f"https://www.ebi.ac.uk/ols4/api/terms?iri={encoded_iri}")
            if response.status_code != 200:
                return set(), set(), set(), set()

            data = response.json()
            if '_embedded' not in data or 'terms' not in data['_embedded']:
                return set(), set(), set(), set()

            # 找到CL本体中的术语
            cl_term = None
            for term in data['_embedded']['terms']:
                if term.get('ontology_prefix') == 'CL':
                    cl_term = term
                    break

            if not cl_term:
                return set(), set(), set(), set()

            # 获取关系链接
            links = cl_term.get('_links', {})
            parents = set()
            children = set()
            ancestors = set()
            descendants = set()

            # 获取父节点
            if 'parents' in links:
                parents_url = links['parents']['href']
                response = requests.get(parents_url)
                if response.status_code == 200:
                    data = response.json()
                    if '_embedded' in data and 'terms' in data['_embedded']:
                        for term in data['_embedded']['terms']:
                            if 'obo_id' in term:
                                parents.add(term['obo_id'])

            # 获取子节点
            if 'children' in links:
                children_url = links['children']['href']
                response = requests.get(children_url)
                if response.status_code == 200:
                    data = response.json()
                    if '_embedded' in data and 'terms' in data['_embedded']:
                        for term in data['_embedded']['terms']:
                            if 'obo_id' in term:
                                children.add(term['obo_id'])

            # 获取祖先
            if 'ancestors' in links:
                ancestors_url = links['ancestors']['href']
                response = requests.get(ancestors_url)
                if response.status_code == 200:
                    data = response.json()
                    if '_embedded' in data and 'terms' in data['_embedded']:
                        for term in data['_embedded']['terms']:
                            if 'obo_id' in term:
                                ancestors.add(term['obo_id'])

            # 获取后代
            if 'descendants' in links:
                descendants_url = links['descendants']['href']
                response = requests.get(descendants_url)
                if response.status_code == 200:
                    data = response.json()
                    if '_embedded' in data and 'terms' in data['_embedded']:
                        for term in data['_embedded']['terms']:
                            if 'obo_id' in term:
                                descendants.add(term['obo_id'])

            result = (parents, children, ancestors, descendants)
            self.cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error getting relations for {term_info['id']}: {e}")
            return set(), set(), set(), set()

def evaluate_matches(pairs_file: str, output_dir: str = "results/evaluation"):
    """
    评估细胞类型对的匹配关系

    Args:
        pairs_file: 包含细胞类型对的CSV文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化匹配器
    matcher = CellOntologyMatcher(cache_file=output_dir/"ontology_cache.pkl")

    # 读取细胞类型对
    print(f"Reading pairs from {pairs_file}")
    pairs_df = pd.read_csv(pairs_file)

    # 初始化结果列表
    results = []

    # 评估每个对
    print("Evaluating matches...")
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        true_type = row['original_celltype']
        pred_type = row['popv_majority_vote_prediction']
        frequency = row['frequency']

        # 查询本体
        true_info = matcher.query_ontology(true_type)
        pred_info = matcher.query_ontology(pred_type)

        # 初始化匹配结果
        match_result = {
            'true_type': true_type,
            'predicted_type': pred_type,
            'frequency': frequency,
            'true_id': true_info['id'] if true_info else None,
            'pred_id': pred_info['id'] if pred_info else None,
            'match_type': 'No Match'
        }

        # 如果两个类型都能在本体中找到
        if true_info and pred_info:
            if true_info['id'] == pred_info['id']:
                match_result['match_type'] = 'Exact Match'
            else:
                # 获取层级关系
                true_parents, true_children, true_ancestors, true_descendants = matcher.get_term_relations(true_info)
                pred_parents, pred_children, pred_ancestors, pred_descendants = matcher.get_term_relations(pred_info)

                # 检查父子关系
                if pred_info['id'] in true_parents:
                    match_result['match_type'] = 'Parent Match'
                elif pred_info['id'] in true_children:
                    match_result['match_type'] = 'Child Match'
                # 检查祖先后代关系
                elif pred_info['id'] in true_ancestors:
                    match_result['match_type'] = 'Ancestor Match'
                elif pred_info['id'] in true_descendants:
                    match_result['match_type'] = 'Descendant Match'
                # 检查兄弟关系（共享父节点）
                elif any(parent in pred_parents for parent in true_parents):
                    match_result['match_type'] = 'Sibling Match'

        results.append(match_result)

    # 保存缓存
    matcher.save_cache()

    # 转换结果为DataFrame
    results_df = pd.DataFrame(results)

    # 保存详细结果
    results_df.to_csv(output_dir/"detailed_matches.csv", index=False)

    # 计算统计信息
    total_cells = pairs_df['frequency'].sum()
    stats = results_df.groupby('match_type').agg({
        'frequency': 'sum'
    }).reset_index()
    stats['percentage'] = stats['frequency'] / total_cells * 100

    # 保存统计信息
    stats.to_csv(output_dir/"match_statistics.csv", index=False)

    # 打印统计信息
    print("\nMatch Statistics:")
    print(stats)

    return results_df, stats

if __name__ == "__main__":
    # 设置输入输出路径
    pairs_file = "results/evaluation/unique_celltype_pairs.csv"
    output_dir = "results/evaluation"

    # 评估匹配
    results_df, stats = evaluate_matches(pairs_file, output_dir)
