#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the cache functionality and API call efficiency in LLMCelltype
"""

import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import hashlib
import glob

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import LLMCelltype functions
from mllmcelltype import annotate_clusters
from mllmcelltype.utils import format_results, create_cache_key

# Load environment variables
load_dotenv()

def test_cache_efficiency():
    """Test the efficiency of the caching mechanism in LLMCelltype"""
    
    print("\n=== Testing Cache Efficiency ===\n")
    
    # Define a simple test dataset
    marker_genes = {
        "1": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],  # T cells
        "2": ["CD19", "MS4A1", "CD79A", "CD79B", "CD22"],  # B cells
        "3": ["CD14", "LYZ", "CSF1R", "MARCO", "CD68"]  # Macrophages
    }
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # Test model
    provider = 'openai'
    model = 'gpt-4o'
    api_key = api_keys[provider]
    
    # Clear cache directory first (optional)
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Run 1: Without cache
    print("Running annotation without cache...")
    start_time = time.time()
    result_no_cache = annotate_clusters(
        marker_genes=marker_genes,
        species='human',
        tissue='blood',
        provider=provider,
        model=model,
        api_key=api_key,
        use_cache=False
    )
    no_cache_time = time.time() - start_time
    print(f"Time without cache: {no_cache_time:.2f} seconds")
    
    # Run 2: With cache (first time, should create cache)
    print("\nRunning annotation with cache (first time)...")
    start_time = time.time()
    result_cache_1 = annotate_clusters(
        marker_genes=marker_genes,
        species='human',
        tissue='blood',
        provider=provider,
        model=model,
        api_key=api_key,
        use_cache=True
    )
    cache_time_1 = time.time() - start_time
    print(f"Time with cache (first time): {cache_time_1:.2f} seconds")
    
    # Run 3: With cache (second time, should use cache)
    print("\nRunning annotation with cache (second time)...")
    start_time = time.time()
    result_cache_2 = annotate_clusters(
        marker_genes=marker_genes,
        species='human',
        tissue='blood',
        provider=provider,
        model=model,
        api_key=api_key,
        use_cache=True
    )
    cache_time_2 = time.time() - start_time
    print(f"Time with cache (second time): {cache_time_2:.2f} seconds")
    
    # Calculate speedup
    if no_cache_time > 0:
        speedup = no_cache_time / cache_time_2
        print(f"\nCache speedup: {speedup:.2f}x faster")
    
    # The results are already formatted as dictionaries
    # No need to call format_results again
    
    # Check if results are consistent
    
    print("\nResults consistency check:")
    for cluster in marker_genes.keys():
        print(f"  Cluster {cluster}:")
        print(f"    Without cache: {result_no_cache.get(cluster, 'N/A')}")
        print(f"    With cache (1st): {result_cache_1.get(cluster, 'N/A')}")
        print(f"    With cache (2nd): {result_cache_2.get(cluster, 'N/A')}")
    
    # Plot timing comparison
    labels = ['No Cache', 'First Run (Cache Creation)', 'Second Run (Cache Use)']
    times = [no_cache_time, cache_time_1, cache_time_2]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['salmon', 'skyblue', 'lightgreen'])
    
    # Add time values on top of bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.ylabel('Time (seconds)')
    plt.title('Cache Performance Comparison')
    plt.tight_layout()
    plt.savefig('cache_performance.png')
    print("\nChart saved as 'cache_performance.png'")
    
    return {
        'no_cache_time': no_cache_time,
        'cache_time_1': cache_time_1,
        'cache_time_2': cache_time_2,
        'speedup': speedup if no_cache_time > 0 else 0
    }

def test_cache_key_generation():
    """Test the cache key generation mechanism"""
    
    print("\n=== Testing Cache Key Generation ===\n")
    
    # Test different prompts, models, and providers
    test_cases = [
        {
            'prompt': "Annotate cell types for markers CD3D, CD3E in human blood",
            'provider': 'openai',
            'model': 'gpt-4o'
        },
        {
            'prompt': "Annotate cell types for markers CD3D, CD3E in mouse blood",  # Changed species
            'provider': 'openai',
            'model': 'gpt-4o'
        },
        {
            'prompt': "Annotate cell types for markers CD3D, CD3E, CD4 in human blood",  # Changed markers
            'provider': 'openai',
            'model': 'gpt-4o'
        },
        {
            'prompt': "Annotate cell types for markers CD3D, CD3E in human blood",
            'provider': 'anthropic',  # Changed provider
            'model': 'claude-3-5-sonnet-20241022'
        },
        {
            'prompt': "Annotate cell types for markers CD3D, CD3E in human blood",
            'provider': 'openai',
            'model': 'gpt-3.5-turbo'  # Changed model
        }
    ]
    
    # Generate cache keys
    cache_keys = []
    for i, case in enumerate(test_cases):
        key = create_cache_key(
            prompt=case['prompt'],
            provider=case['provider'],
            model=case['model']
        )
        cache_keys.append(key)
        print(f"Case {i+1}: {key}")
    
    # Check if keys are different
    unique_keys = set(cache_keys)
    print(f"\nNumber of unique keys: {len(unique_keys)} (expected: {len(test_cases)})")
    if len(unique_keys) == len(test_cases):
        print("All keys are unique as expected")
    else:
        print("WARNING: Some keys are duplicated!")
    
    return cache_keys

def examine_cache_files():
    """Examine the cache files in the cache directory"""
    
    print("\n=== Examining Cache Files ===\n")
    
    # Find cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return {}
    
    # List cache files
    cache_files = glob.glob(os.path.join(cache_dir, "*.json"))
    print(f"Found {len(cache_files)} cache files")
    
    # Analyze cache files
    cache_data = {}
    for file_path in cache_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        # Read cache file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            provider = data.get('provider', 'unknown')
            model = data.get('model', 'unknown')
            timestamp = data.get('timestamp', 'unknown')
            
            cache_data[file_name] = {
                'size_kb': file_size,
                'provider': provider,
                'model': model,
                'timestamp': timestamp
            }
            
            print(f"File: {file_name}")
            print(f"  Size: {file_size:.2f} KB")
            print(f"  Provider: {provider}")
            print(f"  Model: {model}")
            print(f"  Timestamp: {timestamp}")
            print()
            
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    # Analyze cache usage by model
    model_sizes = {}
    for file_info in cache_data.values():
        model = file_info['model']
        size = file_info['size_kb']
        
        if model in model_sizes:
            model_sizes[model] += size
        else:
            model_sizes[model] = size
    
    if model_sizes:
        print("Cache usage by model:")
        for model, size in model_sizes.items():
            print(f"  {model}: {size:.2f} KB")
        
        # Plot cache usage by model
        plt.figure(figsize=(10, 6))
        plt.bar(model_sizes.keys(), model_sizes.values(), color='lightblue')
        plt.xlabel('Model')
        plt.ylabel('Cache Size (KB)')
        plt.title('Cache Usage by Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('cache_usage_by_model.png')
        print("\nChart saved as 'cache_usage_by_model.png'")
    
    return cache_data

def test_api_call_efficiency():
    """Test the efficiency of API calls with different parameters"""
    
    print("\n=== Testing API Call Efficiency ===\n")
    
    # Define a simple test dataset
    marker_genes = {
        "1": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
        "2": ["CD19", "MS4A1", "CD79A", "CD79B", "CD22"],
        "3": ["CD14", "LYZ", "CSF1R", "MARCO", "CD68"]
    }
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # Test different models
    models_to_test = [
        {'provider': 'openai', 'model': 'gpt-4o'},
        {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'},
        {'provider': 'gemini', 'model': 'gemini-1.5-pro'},
        {'provider': 'qwen', 'model': 'qwen2.5-72b-instruct'}
    ]
    
    results = {}
    
    for model_info in models_to_test:
        provider = model_info['provider']
        model = model_info['model']
        api_key = api_keys.get(provider)
        
        if not api_key:
            print(f"Skipping {model} (no API key found for {provider})")
            continue
        
        print(f"\nTesting {model}...")
        
        # Run annotation
        start_time = time.time()
        try:
            result = annotate_clusters(
                marker_genes=marker_genes,
                species='human',
                tissue='blood',
                provider=provider,
                model=model,
                api_key=api_key,
                use_cache=False  # Disable cache to measure actual API call time
            )
            
            elapsed_time = time.time() - start_time
            success = True
            error = None
            
            # The results are already formatted as dictionaries
            formatted_result = result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            success = False
            error = str(e)
            formatted_result = {}
        
        # Store results
        results[model] = {
            'time': elapsed_time,
            'success': success,
            'error': error,
            'predictions': formatted_result
        }
        
        # Print results
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Success: {success}")
        if error:
            print(f"  Error: {error}")
        else:
            print("  Predictions:")
            for cluster, prediction in formatted_result.items():
                print(f"    Cluster {cluster}: {prediction}")
    
    # Plot timing comparison
    if results:
        models = []
        times = []
        colors = []
        
        for model, data in results.items():
            if data['success']:
                models.append(model)
                times.append(data['time'])
                colors.append('lightgreen')
            else:
                models.append(f"{model} (failed)")
                times.append(data['time'])
                colors.append('salmon')
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, times, color=colors)
        
        # Add time values on top of bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.ylabel('Time (seconds)')
        plt.title('API Call Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('api_call_time.png')
        print("\nChart saved as 'api_call_time.png'")
    
    return results

def run_all_tests():
    """Run all cache and API tests"""
    
    print("=== Running All Cache and API Tests ===")
    
    # Test cache efficiency
    cache_efficiency = test_cache_efficiency()
    
    # Test cache key generation
    cache_keys = test_cache_key_generation()
    
    # Examine cache files
    cache_files = examine_cache_files()
    
    # Test API call efficiency
    api_efficiency = test_api_call_efficiency()
    
    print("\n=== All Tests Completed ===")
    
    return {
        'cache_efficiency': cache_efficiency,
        'cache_keys': cache_keys,
        'cache_files': cache_files,
        'api_efficiency': api_efficiency
    }

if __name__ == "__main__":
    run_all_tests()
