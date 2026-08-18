import pandas as pd
import os
import re

# 结果文件目录
results_dir = 'results/benchmark/reference_comparison/2_evaluation'

# 获取所有以 results.csv 结尾的文件
csv_files = [f for f in os.listdir(results_dir) if f.endswith('results.csv')]

# 需要处理的列名
columns_to_clean = [
    'final_consensus',
    'initial_claude-3-5-sonnet-latest',
    'initial_claude-3-5-haiku-latest',
    'initial_gemini-1.5-pro',
    'initial_gemini-2.0-flash-exp',
    'initial_qwen-max-2025-01-25',
    'initial_gpt-4o'
]

def clean_cell_type(value):
    # 移除 "Cluster X:" 格式
    cleaned = re.sub(r'^Cluster\s+\d+:?\s*', '', str(value))
    # 移除数字前缀，如 "1: " 或 "1. " 或 "1.1: "
    cleaned = re.sub(r'^\d+\.?\d*:?\s*', '', cleaned)
    # 移除前后空格
    cleaned = cleaned.strip()
    return cleaned

# 处理每个CSV文件
for csv_file in csv_files:
    print(f"\nProcessing {csv_file}...")
    file_path = os.path.join(results_dir, csv_file)

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 检查哪些列存在于当前文件中
    columns_to_process = [col for col in columns_to_clean if col in df.columns]

    if not columns_to_process:
        print(f"No relevant columns found in {csv_file}")
        continue

    # 清理每个相关列
    for col in columns_to_process:
        if col in df.columns:
            df[col] = df[col].apply(clean_cell_type)

    # 保存修改后的文件
    df.to_csv(file_path, index=False)
    print(f"Cleaned and saved {csv_file}")

print("\nAll files processed successfully!")
