import pandas as pd
import shutil
from pathlib import Path

# 定义路径
source_dir = Path("results/benchmark/reference_comparison/2_evaluation")
target_dir = source_dir / "small_datasets"

# 创建目标文件夹
target_dir.mkdir(exist_ok=True)

# 读取所有结果文件
for file_path in source_dir.glob("*_results_with_match.csv"):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 如果总行数小于10，移动文件
    if len(df) < 11:
        print(f"Moving {file_path.name} (entries: {len(df)})")
        shutil.move(str(file_path), str(target_dir / file_path.name))

print("\nDone! Files with less than 10 entries have been moved to:", target_dir)
