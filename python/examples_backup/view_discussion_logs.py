#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
View the complete content of LLMCelltype discussion logs
"""

import os
import sys
import json

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from LLMCelltype
from mllmcelltype import interactive_consensus_annotation

def view_latest_discussion_logs():
    """View the latest discussion logs"""
    
    # Find the latest result file in the cache directory
    cache_dir = os.path.expanduser("~/.mllmcelltype/cache")
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Get all json files and sort by modification time
    json_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.json')]
    if not json_files:
        print("No cache files found")
        return
    
    # Sort by modification time, newest first
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Look at the 10 most recent files until finding one containing discussion logs
    found_logs = False
    data = None
    
    for i, file_path in enumerate(json_files[:10]):
        print(f"Checking cache file {i+1}: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it contains discussion logs
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
    
    # Print each discussion log
    for cluster, logs in data['discussion_logs'].items():
        print(f"\n{'='*80}")
        print(f"Cluster {cluster} Discussion:")
        print(f"{'='*80}")
        
        if isinstance(logs, list):
            logs_text = "\n".join(logs)
        else:
            logs_text = logs
        
        print(logs_text)
        
        # Try to extract CP and H values
        print("\n--- Extracted Metrics ---")
        for line in logs_text.split('\n'):
            if "Consensus Proportion (CP):" in line:
                print(line)
            if "Shannon Entropy (H):" in line:
                print(line)

if __name__ == "__main__":
    view_latest_discussion_logs()
