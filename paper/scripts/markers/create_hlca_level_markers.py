import pandas as pd
import os

def create_level_marker_files(df, output_dir):
    """
    为每个层级创建标记基因文件

    参数：
    df: 包含标记基因的DataFrame
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有细胞类型的marker_for信息
    cell_types = [col.replace('_marker', '') for col in df.columns if not pd.isna(col) and col.endswith('_marker')]

    # 创建层级字典
    level_markers = {}  # {level_name: {cell_type: [genes]}}

    for cell_type in cell_types:
        marker_col = f"{cell_type}_marker"
        marker_for_col = f"{cell_type}_marker_for"

        # 获取该细胞类型的所有标记基因和它们的层级信息
        markers = df[marker_col].dropna().tolist()
        markers_for = df[marker_for_col].dropna().tolist()

        # 确保markers_for列表至少和markers一样长
        if len(markers_for) < len(markers):
            markers_for.extend([markers_for[-1]] * (len(markers) - len(markers_for)))

        # 将标记基因按层级分组
        for marker, level in zip(markers, markers_for):
            if level not in level_markers:
                level_markers[level] = {}
            if cell_type not in level_markers[level]:
                level_markers[level][cell_type] = []
            level_markers[level][cell_type].append(marker)

    # 为每个层级创建CSV文件
    for level, cell_type_markers in level_markers.items():
        # 创建该层级的所有行
        rows = []
        for cell_type, genes in cell_type_markers.items():
            genes_str = ','.join(genes)
            rows.append(f"{cell_type},{genes_str}")

        # 写入文件
        level_file = os.path.join(output_dir, f'HLCA_level_{level.lower().replace(" ", "_")}_markers.csv')
        with open(level_file, 'w') as f:
            f.write('cluster,gene\n')
            for row in rows:
                f.write(f"{row}\n")
        print(f"Created {level_file}")

    # 创建一个汇总文件，显示每个层级包含的细胞类型
    summary_rows = []
    for level in level_markers.keys():
        cell_types = list(level_markers[level].keys())
        summary_rows.append({
            'level': level,
            'cell_types': ', '.join(cell_types),
            'cell_type_count': len(cell_types)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_output = os.path.join(output_dir, 'HLCA_level_summary.csv')
    summary_df.to_csv(summary_output, index=False)
    print(f"Created {summary_output}")

# 读取marker genes表
file_path = "data/processed/41591_2023_2327_MOESM3_ESM.xlsx"
df = pd.read_excel(file_path, sheet_name='6 - marker genes')

# 获取第一行作为列名
header = df.iloc[0]
df = df.iloc[1:]  # 删除第一行
df.columns = header  # 使用第一行作为列名

# 创建输出文件
output_dir = "data/reference"
create_level_marker_files(df, output_dir)
