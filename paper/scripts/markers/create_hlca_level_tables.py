import pandas as pd
import os
from collections import defaultdict

def get_five_level_hierarchy(hierarchy):
    """获取5层层级结构，保留重复的层级名称"""
    # 移除括号中的注释并去除空白
    clean_hierarchy = [item.split(" (")[0].strip() for item in hierarchy if item.strip()]

    if not clean_hierarchy:
        return []

    # 如果层级数量超过5，取最后5个层级
    if len(clean_hierarchy) >= 5:
        return clean_hierarchy[-5:]

    # 如果层级数量不足5，用最后一个层级填充
    while len(clean_hierarchy) < 5:
        clean_hierarchy.insert(0, clean_hierarchy[0])

    return clean_hierarchy

def get_level_markers(df):
    """获取每个层级的细胞类型及其标记基因"""
    # 存储每个层级的细胞类型和标记基因
    level_markers = defaultdict(lambda: defaultdict(list))

    # 获取所有细胞类型列
    marker_cols = [col for col in df.columns if isinstance(col, str) and col.endswith('_marker')]

    for col in marker_cols:
        cell_type = col.replace('_marker', '')
        marker_for_col = f"{cell_type}_marker_for"

        # 获取标记基因和层级信息
        markers = df[col].dropna().tolist()
        hierarchy = df[marker_for_col].dropna().tolist()

        if not markers or not hierarchy:
            continue

        # 获取5层层级结构
        five_level_hierarchy = get_five_level_hierarchy(hierarchy)

        if not five_level_hierarchy:
            continue

        # 将标记基因添加到相应的层级
        for level, cell_name in enumerate(five_level_hierarchy, 1):
            level_markers[level][cell_name].extend(markers)

    # 对每个层级的标记基因进行去重
    for level in level_markers:
        for cell_type in level_markers[level]:
            level_markers[level][cell_type] = list(dict.fromkeys(level_markers[level][cell_type]))

    return level_markers

def create_level_tables(df, output_dir):
    """为每个层级创建标记基因表"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取每个层级的标记基因
    level_markers = get_level_markers(df)

    # 为每个层级创建CSV文件
    for level in range(1, 6):  # 创建5个层级的文件
        if not level_markers[level]:  # 如果这个层级没有数据，跳过
            print(f"\n=== Level {level} 没有数据 ===")
            continue

        rows = []
        for cell_type, genes in level_markers[level].items():
            # 准备基因列表
            final_genes = genes[:min(9, len(genes))]  # 只取前9个基因，因为还要加上cell_type，总共10列

            # 如果基因数量不足9个，用空字符串填充
            while len(final_genes) < 9:
                final_genes.append('')

            if final_genes:  # 只有当有基因时才添加这一行
                genes_str = ','.join(final_genes)
                rows.append(f"{cell_type},{genes_str}")

        if rows:  # 只有当有数据时才创建文件
            # 写入文件
            output_file = os.path.join(output_dir, f'HLCA_L{level}_markers.csv')
            with open(output_file, 'w') as f:
                f.write('cluster,gene\n')
                for row in sorted(rows):  # 按细胞类型名称排序
                    f.write(f"{row}\n")
            print(f"Created {output_file}")

            # 打印文件内容预览
            print(f"\n=== Level {level} 标记基因预览 ===")
            print("细胞类型数量:", len(rows))
            print("前几个细胞类型示例:")
            with open(output_file, 'r') as f:
                lines = f.readlines()[:6]  # 显示标题和前5个细胞类型
                print(''.join(lines))

def analyze_hierarchy(df):
    """分析层级结构"""
    print("\n=== 层级结构分析 ===")

    # 分析层级长度
    hierarchy_lengths = defaultdict(int)
    examples = defaultdict(list)

    for col in df.columns:
        if isinstance(col, str) and col.endswith('_marker_for'):
            hierarchy = df[col].dropna().tolist()
            if hierarchy:
                clean_hierarchy = [item.split(" (")[0].strip() for item in hierarchy if item.strip()]
                length = len(clean_hierarchy)
                hierarchy_lengths[length] += 1
                if len(examples[length]) < 3:  # 为每个长度保存最多3个例子
                    examples[length].append((col, clean_hierarchy))

    print("各层级数量的细胞类型统计：")
    for length, count in sorted(hierarchy_lengths.items()):
        print(f"层级数量为 {length} 的细胞类型有 {count} 个")
        if examples[length]:
            print("示例：")
            for col, hierarchy in examples[length]:
                print(f"  {col}:")
                print(f"    {' -> '.join(hierarchy)}")

# 读取marker genes表
file_path = "data/processed/41591_2023_2327_MOESM3_ESM.xlsx"
df = pd.read_excel(file_path, sheet_name='6 - marker genes')

# 获取第一行作为列名
header = df.iloc[0]
df = df.iloc[1:]
df.columns = header

# 分析层级结构
analyze_hierarchy(df)

# 创建输出文件
output_dir = "data/reference"
create_level_tables(df, output_dir)
