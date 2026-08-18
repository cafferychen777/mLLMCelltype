import pandas as pd
import os

def create_hierarchical_marker_files(df, output_dir):
    """
    创建包含层级信息的标记基因文件

    参数：
    df: 包含标记基因的DataFrame
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有细胞类型
    cell_types = [col.replace('_marker', '') for col in df.columns if not pd.isna(col) and col.endswith('_marker')]

    # 1. 创建详细的层级CSV文件
    detailed_rows = []
    for cell_type in cell_types:
        marker_col = f"{cell_type}_marker"
        marker_for_col = f"{cell_type}_marker_for"
        reference_col = f"{cell_type}_marker_reference"

        # 获取该细胞类型的所有信息
        markers = df[marker_col].dropna().tolist()
        markers_for = df[marker_for_col].dropna().tolist()
        references = df[reference_col].dropna().tolist()

        # 对每个标记基因创建一行
        for i, marker in enumerate(markers):
            marker_for = markers_for[i] if i < len(markers_for) else ""
            reference = references[i] if i < len(references) else ""
            detailed_rows.append({
                'cell_type': cell_type,
                'marker_gene': marker,
                'marker_for': marker_for,
                'reference': reference
            })

    # 写入详细的层级CSV文件
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_output = os.path.join(output_dir, 'HLCA_detailed_markers.csv')
    detailed_df.to_csv(detailed_output, index=False)
    print(f"Created {detailed_output}")

    # 2. 创建按组织区室分组的标记基因文件
    compartment_groups = {}
    for _, row in detailed_df.iterrows():
        compartment = row['marker_for']
        if pd.notna(compartment):
            if compartment not in compartment_groups:
                compartment_groups[compartment] = set()
            compartment_groups[compartment].add((row['cell_type'], row['marker_gene']))

    # 为每个组织区室创建一个文件
    for compartment, markers in compartment_groups.items():
        compartment_rows = []
        for cell_type, gene in markers:
            compartment_rows.append(f"{cell_type},{gene}")

        # 写入文件
        compartment_file = os.path.join(output_dir, f'HLCA_{compartment.replace(" ", "_").lower()}_markers.csv')
        with open(compartment_file, 'w') as f:
            f.write('cluster,gene\n')
            for row in compartment_rows:
                f.write(f"{row}\n")
        print(f"Created {compartment_file}")

    # 3. 创建一个汇总文件，显示细胞类型的层级关系
    hierarchy_rows = []
    for cell_type in cell_types:
        marker_for_col = f"{cell_type}_marker_for"
        hierarchy = df[marker_for_col].dropna().unique().tolist()
        hierarchy_rows.append({
            'cell_type': cell_type,
            'hierarchy': ' -> '.join(hierarchy)
        })

    hierarchy_df = pd.DataFrame(hierarchy_rows)
    hierarchy_output = os.path.join(output_dir, 'HLCA_cell_type_hierarchy.csv')
    hierarchy_df.to_csv(hierarchy_output, index=False)
    print(f"Created {hierarchy_output}")

# 读取marker genes表
file_path = "data/processed/41591_2023_2327_MOESM3_ESM.xlsx"
df = pd.read_excel(file_path, sheet_name='6 - marker genes')

# 获取第一行作为列名
header = df.iloc[0]
df = df.iloc[1:]  # 删除第一行
df.columns = header  # 使用第一行作为列名

# 创建输出文件
output_dir = "data/reference"
create_hierarchical_marker_files(df, output_dir)
