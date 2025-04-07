#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查看LLMCelltype讨论日志的完整内容
"""

import os
import sys
import json

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入LLMCelltype的函数
from llmcelltype import interactive_consensus_annotation

def view_latest_discussion_logs():
    """查看最新的讨论日志"""
    
    # 查找缓存目录中最新的结果文件
    cache_dir = os.path.expanduser("~/.llmcelltype/cache")
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # 获取所有json文件并按修改时间排序
    json_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.json')]
    if not json_files:
        print("No cache files found")
        return
    
    # 按修改时间排序，最新的在前
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 查看前10个最新的文件，直到找到包含讨论日志的文件
    found_logs = False
    data = None
    
    for i, file_path in enumerate(json_files[:10]):
        print(f"Checking cache file {i+1}: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 检查是否包含讨论日志
            if 'discussion_logs' in data and data['discussion_logs']:
                print(f"Found discussion logs in file: {file_path}")
                found_logs = True
                break
            else:
                print("No discussion logs found in this file")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    if not found_logs:
        print("No discussion logs found in any of the recent cache files")
        return
    
    # 打印每个讨论日志
    for cluster, logs in data['discussion_logs'].items():
        print(f"\n{'='*80}")
        print(f"Cluster {cluster} Discussion:")
        print(f"{'='*80}")
        
        if isinstance(logs, list):
            logs_text = "\n".join(logs)
        else:
            logs_text = logs
        
        print(logs_text)
        
        # 尝试提取CP值和H值
        print("\n--- Extracted Metrics ---")
        for line in logs_text.split('\n'):
            if "Consensus Proportion (CP):" in line:
                print(line)
            if "Shannon Entropy (H):" in line:
                print(line)

if __name__ == "__main__":
    view_latest_discussion_logs()
