"""
Color utilities for cell type visualization.
This module provides utilities for standardizing cell type names and colors.
"""

def get_standardized_name(cell_type):
    """
    Get standardized name for a cell type
    Args:
        cell_type (str): Original cell type name
    Returns:
        str: Standardized cell type name
    """
    # 处理空值或非字符串输入
    if not isinstance(cell_type, str) or not cell_type:
        return 'Unknown'

    # 先尝试直接从映射中获取标准化名称
    standardized = name_mapping.get(cell_type, cell_type)

    # 如果没有直接匹配，尝试小写匹配
    if standardized == cell_type:
        standardized = name_mapping.get(cell_type.lower(), cell_type)

    # 如果仍然没有匹配，尝试去除前后空格后匹配
    if standardized == cell_type:
        standardized = name_mapping.get(cell_type.strip(), cell_type)

    # 如果标准化后的名称不在颜色字典中，只有在原始名称不是标准化名称时才发出警告
    if standardized not in cell_colors and cell_type != standardized:
        # 不打印警告，因为这会产生太多输出
        pass

    return standardized



def get_expanded_cell_type_hierarchy():
    """
    Get an expanded cell type hierarchy with more comprehensive relationships

    Returns:
        dict: Dictionary mapping parent cell types to lists of child cell types
    """
    return {
        # 免疫细胞 - T细胞家族
        't cell': [
            'cd4+ t cell', 'cd8+ t cell', 'regulatory t cell', 'gamma delta t cell',
            'naive t cell', 'memory t cell', 'effector t cell', 'helper t cell', 'cytotoxic t cell',
            'naive cd4+ t cell', 'naive cd8+ t cell', 'memory cd4+ t cell', 'memory cd8+ t cell',
            'effector cd4+ t cell', 'effector cd8+ t cell', 'effector memory cd4+ t cell',
            'effector memory cd8+ t cell', 'central memory cd4+ t cell', 'central memory cd8+ t cell',
            'th1 cell', 'th2 cell', 'th17 cell', 'follicular helper t cell', 'exhausted t cell',
            'resident memory t cell', 'naive thymus-derived cd4+ t cell', 'naive thymus-derived cd8+ t cell',
            'activated cd4+ t cell', 'activated cd8+ t cell', 'mait cell', 'mucosal associated invariant t cell',
            'cd4-positive, alpha-beta t cell', 'cd8-positive, alpha-beta t cell',
            'cd4-positive, alpha-beta memory t cell', 'cd8-positive, alpha-beta memory t cell',
            'cd4-positive, alpha-beta thymocyte', 'cd8-positive, alpha-beta thymocyte',
            'cd8+ tissue-resident memory t cell', 'tissue-resident memory t cell',
            'effector memory cd4-positive, alpha-beta t cell', 'effector memory cd8-positive, alpha-beta t cell'
        ],

        # CD4+ T细胞家族
        'cd4+ t cell': [
            'naive cd4+ t cell', 'memory cd4+ t cell', 'effector cd4+ t cell',
            'effector memory cd4+ t cell', 'central memory cd4+ t cell',
            'th1 cell', 'th2 cell', 'th17 cell', 'follicular helper t cell', 'regulatory t cell',
            'activated cd4+ t cell', 'cd4-positive, alpha-beta t cell',
            'cd4-positive, alpha-beta memory t cell', 'cd4-positive, alpha-beta thymocyte',
            'effector memory cd4-positive, alpha-beta t cell', 'naive thymus-derived cd4-positive, alpha-beta t cell'
        ],

        # CD8+ T细胞家族
        'cd8+ t cell': [
            'naive cd8+ t cell', 'memory cd8+ t cell', 'effector cd8+ t cell',
            'effector memory cd8+ t cell', 'central memory cd8+ t cell', 'cytotoxic t cell',
            'activated cd8+ t cell', 'cd8+ tissue-resident memory t cell',
            'cd8-positive, alpha-beta t cell', 'cd8-positive, alpha-beta memory t cell',
            'cd8-positive, alpha-beta thymocyte', 'effector memory cd8-positive, alpha-beta t cell',
            'naive thymus-derived cd8-positive, alpha-beta t cell'
        ],

        # 免疫细胞 - B细胞家族
        'b cell': [
            'naive b cell', 'memory b cell', 'plasma cell', 'plasmablast',
            'germinal center b cell', 'marginal zone b cell', 'follicular b cell',
            'b1 cell', 'b2 cell', 'regulatory b cell', 'antibody-secreting cell',
            'activated b cell', 'immature b cell', 'mature b cell', 'transitional b cell',
            'pre-b cell', 'pro-b cell'
        ],

        # 免疫细胞 - 先天性淋巴细胞
        'innate lymphoid cell': [
            'natural killer cell', 'nk cell', 'ilc1', 'ilc2', 'ilc3',
            'lymphoid tissue inducer cell', 'mature nk t cell', 'nkt cell',
            'group 1 innate lymphoid cell', 'group 2 innate lymphoid cell', 'group 3 innate lymphoid cell'
        ],

        # 自然杀手细胞家族
        'natural killer cell': [
            'nk cell', 'cd56bright nk cell', 'cd56dim nk cell', 'cd16+ nk cell', 'cd16- nk cell',
            'mature natural killer cell', 'immature natural killer cell'
        ],

        # 免疫细胞 - 骨髓细胞
        'myeloid cell': [
            'monocyte', 'macrophage', 'dendritic cell', 'neutrophil', 'eosinophil',
            'basophil', 'mast cell', 'classical monocyte', 'non-classical monocyte',
            'intermediate monocyte', 'alveolar macrophage', 'interstitial macrophage',
            'kupffer cell', 'microglia', 'langerhans cell', 'myeloid dendritic cell',
            'plasmacytoid dendritic cell', 'conventional dendritic cell',
            'cd1c-positive myeloid dendritic cell', 'cd141+ dendritic cell',
            'cd103+ dendritic cell', 'cd11b+ dendritic cell', 'cd14+ monocyte', 'cd16+ monocyte',
            'tissue resident macrophage', 'inflammatory macrophage', 'm1 macrophage', 'm2 macrophage'
        ],

        # 单核细胞家族
        'monocyte': [
            'classical monocyte', 'non-classical monocyte', 'intermediate monocyte',
            'cd14+ monocyte', 'cd16+ monocyte', 'inflammatory monocyte', 'patrolling monocyte'
        ],

        # 巨噩细胞家族
        'macrophage': [
            'alveolar macrophage', 'interstitial macrophage', 'kupffer cell', 'microglia',
            'tissue resident macrophage', 'inflammatory macrophage', 'm1 macrophage', 'm2 macrophage',
            'tumor-associated macrophage', 'red pulp macrophage', 'white pulp macrophage',
            'peritoneal macrophage', 'bone marrow macrophage', 'splenic macrophage'
        ],

        # 树突细胞家族
        'dendritic cell': [
            'myeloid dendritic cell', 'plasmacytoid dendritic cell', 'conventional dendritic cell',
            'cd1c-positive myeloid dendritic cell', 'cd141+ dendritic cell', 'cd103+ dendritic cell',
            'cd11b+ dendritic cell', 'langerhans cell', 'follicular dendritic cell',
            'interdigitating dendritic cell', 'inflammatory dendritic cell', 'tolerogenic dendritic cell',
            'cdc1', 'cdc2', 'mature dendritic cell', 'immature dendritic cell'
        ],

        # 上皮细胞
        'epithelial cell': [
            'alveolar epithelial cell', 'bronchial epithelial cell', 'ciliated epithelial cell',
            'basal epithelial cell', 'club cell', 'pulmonary ionocyte', 'alveolar type 1 cell',
            'alveolar type 2 cell', 'pulmonary alveolar type 1 cell', 'pulmonary alveolar type 2 cell',
            'lung ciliated cell', 'respiratory basal cell', 'secretory cell', 'mucus cell',
            'goblet cell', 'lung goblet cell', 'respiratory goblet cell', 'mucus secreting cell',
            'neuroendocrine cell', 'pulmonary neuroendocrine cell', 'tuft cell', 'brush cell',
            'type i alveolar cell', 'type ii alveolar cell', 'type 1 alveolar cell', 'type 2 alveolar cell',
            'alveolar type i cell', 'alveolar type ii cell', 'at1', 'at2',
            'basal cell', 'clara cell', 'serous cell', 'hillock cell', 'ionocyte',
            'multiciliated cell', 'pulmonary secretory cell', 'airway secretory cell'
        ],

        # 肺泵细胞家族
        'pneumocyte': [
            'alveolar type 1 cell', 'alveolar type 2 cell', 'pulmonary alveolar type 1 cell',
            'pulmonary alveolar type 2 cell', 'type i alveolar cell', 'type ii alveolar cell',
            'type 1 alveolar cell', 'type 2 alveolar cell', 'alveolar type i cell',
            'alveolar type ii cell', 'at1', 'at2'
        ],

        # I型肺泵细胞
        'alveolar type 1 cell': [
            'pulmonary alveolar type 1 cell', 'type i alveolar cell', 'type 1 alveolar cell',
            'alveolar type i cell', 'at1'
        ],

        # II型肺泵细胞
        'alveolar type 2 cell': [
            'pulmonary alveolar type 2 cell', 'type ii alveolar cell', 'type 2 alveolar cell',
            'alveolar type ii cell', 'at2'
        ],

        # 内皮细胞
        'endothelial cell': [
            'capillary endothelial cell', 'lymphatic endothelial cell', 'arterial endothelial cell',
            'vein endothelial cell', 'endothelial cell of artery', 'endothelial cell of lymphatic vessel',
            'pulmonary capillary endothelial cell', 'pulmonary arterial endothelial cell',
            'pulmonary venous endothelial cell', 'microvascular endothelial cell',
            'vascular endothelial cell', 'blood vessel endothelial cell', 'high endothelial venule cell',
            'sinusoidal endothelial cell', 'continuous endothelial cell', 'fenestrated endothelial cell',
            'discontinuous endothelial cell', 'tip cell', 'stalk cell', 'phalanx cell'
        ],

        # 间质细胞
        'stromal cell': [
            'fibroblast', 'myofibroblast', 'adventitial fibroblast', 'pulmonary interstitial fibroblast',
            'pericyte', 'mesenchymal stem cell', 'mesenchymal stromal cell',
            'smooth muscle cell', 'vascular associated smooth muscle cell', 'bronchial smooth muscle cell',
            'cardiac fibroblast', 'dermal fibroblast', 'lung fibroblast', 'activated fibroblast',
            'quiescent fibroblast', 'perivascular fibroblast', 'lipofibroblast', 'stellate cell',
            'hepatic stellate cell', 'pancreatic stellate cell', 'mesothelial cell'
        ],

        # 成纤维细胞家族
        'fibroblast': [
            'myofibroblast', 'adventitial fibroblast', 'pulmonary interstitial fibroblast',
            'cardiac fibroblast', 'dermal fibroblast', 'lung fibroblast', 'activated fibroblast',
            'quiescent fibroblast', 'perivascular fibroblast', 'lipofibroblast'
        ],

        # 神经系统细胞
        'neural cell': [
            'neuron', 'glial cell', 'astrocyte', 'oligodendrocyte', 'schwann cell',
            'satellite glial cell', 'olfactory ensheathing cell', 'radial glial cell',
            'microglia', 'ependymal cell', 'tanycyte', 'neural stem cell', 'neural progenitor cell',
            'neuroblast', 'motor neuron', 'sensory neuron', 'interneuron', 'excitatory neuron',
            'inhibitory neuron', 'pyramidal neuron', 'dopaminergic neuron', 'serotonergic neuron',
            'cholinergic neuron', 'gabaergic neuron', 'glutamatergic neuron'
        ],

        # 神经胺细胞家族
        'glial cell': [
            'astrocyte', 'oligodendrocyte', 'schwann cell', 'satellite glial cell',
            'olfactory ensheathing cell', 'radial glial cell', 'microglia', 'ependymal cell',
            'tanycyte', 'oligodendrocyte precursor cell', 'myelinating oligodendrocyte',
            'non-myelinating oligodendrocyte', 'protoplasmic astrocyte', 'fibrous astrocyte',
            'bergmann glia', 'muller glia', 'reactive astrocyte'
        ],

        # 造血系统细胞
        'hematopoietic cell': [
            'hematopoietic stem cell', 'hematopoietic progenitor cell', 'common myeloid progenitor',
            'common lymphoid progenitor', 'megakaryocyte-erythroid progenitor', 'granulocyte-monocyte progenitor',
            'erythrocyte', 'platelet', 'megakaryocyte', 'granulocyte', 'neutrophil', 'eosinophil',
            'basophil', 'mast cell', 'monocyte', 'macrophage', 'dendritic cell', 't cell', 'b cell',
            'natural killer cell', 'innate lymphoid cell', 'plasma cell'
        ],

        # 粒细胞家族
        'granulocyte': [
            'neutrophil', 'eosinophil', 'basophil', 'mast cell', 'segmented neutrophil',
            'band neutrophil', 'hypersegmented neutrophil', 'immature neutrophil',
            'mature neutrophil', 'activated neutrophil', 'tissue neutrophil'
        ],

        # 淋巴细胞家族
        'lymphocyte': [
            't cell', 'b cell', 'natural killer cell', 'innate lymphoid cell',
            'plasma cell', 'memory lymphocyte', 'naive lymphocyte', 'effector lymphocyte',
            'regulatory lymphocyte', 'cytotoxic lymphocyte', 'helper lymphocyte'
        ]
    }

def evaluate_cell_type_match(reference_type, predicted_type):
    """
    Evaluate the matching degree between two cell types based on Cell Ontology relationships

    Args:
        reference_type (str): Reference cell type
        predicted_type (str): Predicted cell type

    Returns:
        float: Matching score (1.0 = full match, 0.5 = partial match, 0.0 = no match)
    """
    # 处理空值或非字符串输入
    if not isinstance(reference_type, str) or not isinstance(predicted_type, str):
        return 0.0

    # 标准化细胞类型名称以便更好地匹配
    ref_std = get_standardized_name(reference_type)
    pred_std = get_standardized_name(predicted_type)

    # 处理未知预测
    if ref_std.lower() == 'unknown' or pred_std.lower() == 'unknown':
        return 0.0

    # 完全匹配情况 - 标准化后的精确匹配
    if ref_std.lower() == pred_std.lower():
        return 1.0

    # 获取扩展的细胞类型层次结构
    cell_type_hierarchy = get_expanded_cell_type_hierarchy()

    # 转换为小写以进行比较
    ref_lower = ref_std.lower()
    pred_lower = pred_std.lower()

    # 检查部分匹配使用扩展的层次结构
    for parent, children in cell_type_hierarchy.items():
        # 检查参考是父类型，预测是子类型
        if ref_lower == parent and pred_lower in children:
            return 0.5

        # 检查预测是父类型，参考是子类型
        if pred_lower == parent and ref_lower in children:
            return 0.5

        # 检查参考和预测都是同一父类型的子类型
        if ref_lower in children and pred_lower in children:
            return 0.5

        # 检查参考包含父类型术语，预测是子类型
        if parent in ref_lower and pred_lower in children:
            return 0.5

        # 检查预测包含父类型术语，参考是子类型
        if parent in pred_lower and ref_lower in children:
            return 0.5

    # 检查词汇相似性（如果一个是另一个的子字符串）
    if ref_lower in pred_lower or pred_lower in ref_lower:
        return 0.25  # 较低的匹配分数，因为这只是基于文本的匹配

    # 无匹配
    return 0.0

def get_cell_type_color(cell_type):
    """
    Get color for a cell type
    Args:
        cell_type (str): Cell type name
    Returns:
        str: Hex color code
    """
    # Return color for the cell type name
    if cell_type in cell_colors:
        return cell_colors[cell_type]
    else:
        print(f"WARNING: No color found for: {cell_type}, using default gray")
        return cell_colors.get('Unknown', '#BDBDBD')  # Default to light gray

# Define standardized cell type colors using Nature Methods-suitable palette
# Colors are organized by major cell type categories with consistent color families
# Each category uses a distinct color spectrum with varying shades for subtypes
cell_colors = {
    'early T lineage precursor': '#90CAF9',
    'Cortical Thymocytes': '#1976D2',
    'CD4+ T cells': '#1976D2',
    'CD4+ Memory T cells': '#1E88E5',
    'CD4+ Naive T cells': '#42A5F5',
    'CD4+ effector T cells': '#0D47A1',
    'CD4+ effector memory T cells': '#1565C0',
    'CD4+ central memory T cells': '#1E88E5',
    'CD4-positive, alpha-beta T cell': '#1565C0',
    'CD4-positive, alpha-beta memory T cell': '#1E88E5',
    'CD4-positive, alpha-beta thymocyte': '#42A5F5',
    'naive thymus-derived CD4-positive, alpha-beta T cell': '#90CAF9',
    'CD8+ T cells': '#3949AB',
    'Activated CD8+ T cells': '#1976D2',
    'CD8+ Memory T cells': '#5C6BC0',
    'CD8+ Naive T cells': '#7986CB',
    'CD8+ effector T cells': '#283593',
    'CD8+ effector memory T cells': '#3F51B5',
    'CD8+ central memory T cells': '#5C6BC0',
    'CD8+ tissue-resident memory T cells': '#4527A0',
    'CD8+ T cell': '#3949AB',
    'CD8+ T cells': '#3949AB',
    'CD8-positive, alpha-beta T cell': '#3949AB',
    'CD8-positive, alpha-beta memory T cell': '#5C6BC0',
    'CD8-positive, alpha-beta thymocyte': '#7986CB',
    'CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell': '#64B5F6',
    'naive thymus-derived CD8-positive, alpha-beta T cell': '#90CAF9',
    'CD8+ Naive T cell': '#7986CB',
    'Effector Memory CD8+ T cell (TEM)': '#3F51B5',
    'T cells': '#00838F',
    'T cell': '#00838F',
    'alpha-beta T cell': '#00838F',
    'Developing T cells': '#42A5F5',
    'Activated T cells': '#1976D2',
    'Activated T cell': '#1976D2',
    'T follicular helper cell': '#1E88E5',
    'T cells (developing)': '#42A5F5',
    'Regulatory T cells': '#1565C0',
    'regulatory T cell': '#1565C0',
    'MAIT cells': '#1976D2',
    'Double-positive thymocytes': '#0D47A1',
    'Double-negative thymocytes': '#42A5F5',
    'thymocyte': '#1976D2',
    'Thymocytes': '#1976D2',
    'double negative thymocyte': '#42A5F5',
    'double-positive, alpha-beta thymocyte': '#0D47A1',
    'Double positive thymocyte': '#0D47A1',
    'Cortical Thymocyte': '#1976D2',
    'Early T cells/Thymocyte': '#90CAF9',
    'Early T cells/Thymocytes': '#90CAF9',
    'γδ T cells': '#00BCD4',
    'Gamma delta T cell': '#00BCD4',
    'Gamma delta T cells': '#00BCD4',
    'Cytotoxic T cell': '#3949AB',
    'Cytotoxic T cells': '#3949AB',
    'Thymic epithelial cells': '#006D2C',
    'Thymic epithelial cell': '#006D2C',
    'epithelial cell of thymus': '#006D2C',
    'Epithelial cell': '#006D2C',
    'Medullary thymic epithelial cells': '#238B45',
    'Medullary thymic epithelial cell': '#238B45',
    'medullary thymic epithelial cell': '#238B45',
    'Cortical thymic epithelial cells': '#41AE76',
    'Cortical thymic epithelial cell': '#41AE76',
    'cortical thymic epithelial cell': '#41AE76',
    'Epithelial progenitor cell': '#66C2A4',
    'Epithelial progenitor cells': '#66C2A4',
    'myo-medullary thymic epithelial cell': '#006D2C',
    'neuro-medullary thymic epithelial cell': '#006D2C',
    'B cells': '#2E7D32',
    'B cell': '#2E7D32',
    'Naive B cells': '#1B5E20',
    'naive B cell': '#1B5E20',
    'Memory B cells': '#388E3C',
    'memory B cell': '#388E3C',
    'Precursor B cells': '#43A047',
    'precursor B cell': '#43A047',
    'B cells (Immature/Pre-B cells)': '#1B5E20',
    'Plasma cells': '#004D40',
    'plasma cell': '#004D40',
    'Plasma cell': '#004D40',
    'NK cells': '#C62828',
    'NK/NKT cells': '#E53935',
    'natural killer cell': '#C62828',
    'Natural Killer cell': '#C62828',
    'Natural Killer (NK) cell': '#C62828',
    'Natural Killer (NK) cells': '#C62828',
    'NK/NKT cell': '#E53935',
    'NKT cells': '#D32F2F',
    'dendritic cell': '#6A3805',
    'Dendritic cell': '#6A3805',
    'Dendritic cells': '#6A3805',
    'Conventional Dendritic Cells 1 (cDC1)': '#9C6B38',
    'plasmacytoid dendritic cell': '#BF812D',
    'Plasmacytoid dendritic cell': '#BF812D',
    'Plasmacytoid dendritic cells': '#BF812D',
    'monocyte': '#DFC27D',
    'Monocyte': '#DFC27D',
    'Monocytes': '#DFC27D',
    'Classical Monocytes': '#DFC27D',
    'Monocytes/Macrophage': '#DFC27D',
    'Monocytes/Macrophages': '#DFC27D',
    'macrophage': '#A67C33',
    'Macrophage': '#A67C33',
    'Macrophages': '#A67C33',
    'Myeloid cell': '#966842',
    'Myeloid cells': '#966842',
    'mast cell': '#DE77AE',
    'Mast cell': '#DE77AE',
    'Mast cells': '#DE77AE',
    'progenitor cell': '#A50F15',
    'Progenitor cells': '#A50F15',
    'hematopoietic precursor cell': '#A50F15',
    'endothelial cell': '#00695C',
    'Endothelial cell': '#00695C',
    'Endothelial cells': '#00695C',
    'fibroblast': '#8C510A',
    'Fibroblast': '#8C510A',
    'Fibroblasts': '#8C510A',
    'megakaryocyte': '#EF3B2C',
    'Megakaryocyte': '#EF3B2C',
    'Megakaryocytes': '#EF3B2C',
    'Myeloid dendritic cells': '#9C6B38',
    'Classical monocytes': '#E6C377',
    'Non-classical monocytes': '#D6B656',
    'Intermediate monocytes': '#CDAA54',
    'Alveolar macrophages': '#A67C33',
    'ILC3 cells': '#EF5350',
    'Innate lymphoid cells': '#E53935',
    'lymphocyte': '#0097A7',
    'neutrophil': '#C2185B',
    'endothelial cell of lymphatic vessel': '#00796B',
    'endothelial cell of artery': '#00695C',
    'vein endothelial cell': '#00695C',
    'capillary endothelial cell': '#00695C',
    'vascular associated smooth muscle cell': '#00897B',
    'Smooth muscle cell': '#00897B',
    'bronchial smooth muscle cell': '#00897B',
    'Smooth muscle cells': '#00897B',
    'smooth muscle cell': '#00897B',
    'fast muscle cell': '#00897B',
    'thymic fibroblast type 1': '#8C510A',
    'thymic fibroblast type 2': '#8C510A',
    'mesothelial cell': '#8C510A',
    'Erythroid cells': '#880E4F',
    'Erythroid cell': '#880E4F',
    'Erythroid precursor': '#AD1457',
    'Erythroid precursors': '#AD1457',
    'Erythroid progenitors': '#AD1457',
    'Erythroid cells': '#C2185B',
    'erythrocyte': '#880E4F',
    'Naive CD4+ T cells': '#6BAED6',
    'Naive CD8+ T cells': '#4292C6',
    'Activated CD4+ T cells': '#42A5F5',
    'Basophils': '#E91E63',
    'Neutrophils': '#C2185B',
    'Granulocytes': '#D81B60',
    'Platelets': '#EF3B2C',
    'Leukocytes': '#A50F15',
    'Erythrocytes': '#880E4F',
    'erythroid lineage cell': '#880E4F',
    'erythroid progenitor cell': '#AD1457',
    'Natural Killer Cells': '#C62828',
    'Proliferating NK cells': '#C62828',
    'Hematopoietic cells': '#A50F15',
    'Proliferating cells': '#5D4037',
    'Unknown': '#757575',
    'Other': '#9E9E9E',
    'Lymphocytes': '#0097A7',
    'Lymphocytes (likely cycling/proliferating cells)': '#00838F',
    'pulmonary alveolar type 1 cell': '#00441B',
    'Alveolar epithelial cells type I': '#00441B',
    'alveolar epithelial cells type I': '#00441B',
    'Alveolar Epithelial Type I Cells': '#00441B',
    'pulmonary alveolar type 2 cell': '#74C476',
    'Alveolar epithelial cells type II': '#74C476',
    'alveolar epithelial cells type II': '#74C476',
    'alveolar type II pneumocytes': '#A1D99B',
    'Alveolar type II pneumocytes': '#A1D99B',
    'Alveolar epithelial cells': '#4DB6AC',
    'pulmonary ionocyte': '#7BCCC4',
    'Pulmonary ionocytes': '#7BCCC4',
    'respiratory goblet cell': '#0868AC',
    'Goblet cells': '#0868AC',
    'lung goblet cell': '#0868AC',
    'mucus secreting cell': '#0868AC',
    'club cell': '#084081',
    'Club cells': '#084081',
    'serous cell of epithelium of bronchus': '#A8DDB5',
    'Serous cells': '#A8DDB5',
    'tracheobronchial serous cell': '#A8DDB5',
    'Secretory cells': '#A8DDB5',
    'basal cell': '#E0F3DB',
    'Basal epithelial cells': '#E0F3DB',
    'respiratory basal cell': '#E0F3DB',
    'adventitial cell': '#D67D1A',
    'alveolar adventitial fibroblast': '#D67D1A',
    'Adventitial fibroblasts': '#D67D1A',
    'pulmonary interstitial fibroblast': '#D67D1A',
    'myofibroblast cell': '#D8B365',
    'Extracellular Matrix': '#D8B365',
    'pericyte': '#A1887F',
    'Pericytes': '#A1887F',
    'Ciliated Epithelial Cells': '#CCEBC5',
    'Ciliated epithelial cells': '#CCEBC5',
    'Multiciliated Cells': '#E0F3DB',
    'lung ciliated cell': '#CCEBC5',
    'ciliated cell': '#CCEBC5',
    'lung neuroendocrine cell': '#43A2CA',
    'Neuroendocrine cells': '#43A2CA',
    'Smooth Muscle Cells': '#00897B',
    'Arterial endothelial cell': '#00695C',
    'Arterial endothelial cells': '#00695C',
    'Capillary endothelial cell': '#00695C',
    'Capillary endothelial cells': '#00695C',
    'Capillary Endothelial Cell': '#00695C',
    'Venous endothelial cells': '#00695C',
    'Lymphatic endothelial cells': '#00796B',
    'epithelial cell': '#66C2A4',
    'Epithelial cells': '#66C2A4',
    'mesothelial cell of pleura': '#8C510A',
    'Proliferating T cells': '#0277BD',
    'effector memory CD4-positive, alpha-beta T cell': '#1E88E5',
    'effector memory CD8-positive, alpha-beta T cell': '#3F51B5',
    'effector memory CD4-positive': '#1E88E5',
    'effector memory CD8-positive': '#3F51B5',
    'naive thymus-derived CD4-positive': '#6BAED6',
    'naive thymus-derived CD8-positive': '#4292C6',
    'Naive T cells': '#2196F3',
    'Natural killer cells': '#C62828',
    'alveolar macrophage': '#A67C33',
    'CD1c-positive myeloid dendritic cell': '#9C6B38',
    'myeloid dendritic cell, human': '#9C6B38',
    'plasmacytoid dendritic cell, human': '#BF812D',
    'Plasmacytoid Dendritic Cells': '#BF812D',
    'Mesothelial cells': '#8C510A',
    'Myofibroblasts': '#D8B365',
    'Pulmonary fibroblasts': '#D67D1A',
    # 需要添加的颜色定义（标准化后的名称）
'Atypical memory B cells': '#4CAF50',  # 绿色系，与其他B细胞相似
'CD8+ effector T cells': '#283593',  # 已有相同的颜色定义，但为了完整性添加
'Exhausted T cells': '#3949AB',  # 蓝色，与CD8+ T cells相似
'Memory T cells': '#5C6BC0',  # 蓝色，与记忆T细胞相似
}

# Define standardized labels for common cell types
name_mapping = {
    # 处理未知或空值
    'unknown': 'Unknown',
    'Unknown': 'Unknown',
    'NA': 'Unknown',
    'N/A': 'Unknown',
    'unclassified': 'Unknown',
    'Unclassified': 'Unknown',
    'undetermined': 'Unknown',
    'Undetermined': 'Unknown',

    # 处理复数形式
    'Cells': 'Cells',
    'Macrophages': 'Macrophages',
    'Monocytes': 'Monocytes',
    'Lymphocytes': 'Lymphocytes',
    'Neutrophils': 'Neutrophils',
    'Fibroblasts': 'Fibroblasts',
    'Pneumocytes': 'Pneumocytes',
    'Dendrites': 'Dendrites',
    'Epithelial cells': 'Epithelial cells',
    'Endothelial cells': 'Endothelial cells',
    'Natural killer cells': 'NK cells',
    'T cells': 'T cells',
    'T Cells': 'T cells',
    'B cells': 'B cells',
    'Dendritic cells': 'Dendritic cells',
    'Smooth muscle cell': 'Smooth muscle cells',
    'Regulatory T cell': 'Regulatory T cells',

    # 处理常见前缀
    'human ': '',
    'mouse ': '',
    'mature ': '',
    'immature ': '',
    'activated ': '',
    'resting ': '',

    # 处理同义词和变体
    'NK cell': 'NK cells',
    'NK cells': 'NK cells',
    'Type I alveolar cell': 'Alveolar epithelial cells type I',
    'Type 1 alveolar cell': 'Alveolar epithelial cells type I',
    'Type II alveolar cell': 'Alveolar epithelial cells type II',
    'Type 2 alveolar cell': 'Alveolar epithelial cells type II',
    'Alveolar type I cell': 'Alveolar epithelial cells type I',
    'Alveolar type II cell': 'Alveolar epithelial cells type II',
    'Alveolar type II cells': 'Alveolar epithelial cells type II',
    'Alveolar Epithelial Cells': 'Alveolar epithelial cells',
    'AT1': 'Alveolar epithelial cells type I',
    'AT2': 'Alveolar epithelial cells type II',
    'AM': 'Alveolar macrophage',
    'DC': 'Dendritic cell',
    'mDC': 'Myeloid dendritic cell',
    'pDC': 'Plasmacytoid dendritic cell',
    'Treg': 'Regulatory T cell',
    'Regulatory T': 'Regulatory T cell',
    'Gamma-delta': 'Gamma delta T cell',
    'Gamma delta': 'Gamma delta T cell',
    'CD4+ T': 'CD4+ T cell',
    'CD8+ T': 'CD8+ T cell',
    'CD4-positive': 'CD4+ T cell',
    'CD8-positive': 'CD8+ T cell',
    'CD4+': 'CD4+ T cell',
    'CD8+': 'CD8+ T cell',

    # Specific monocyte populations
    'Monocytes_CD14': 'Classical monocytes',
    'Monocytes_CD16': 'Non-classical monocytes',

    # General T cells and thymocytes
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'CD8+ T cells': 'CD8+ T cells',
    'CD8+ T cell': 'CD8+ T cells',
    'CD8-positive, alpha-beta memory T cell': 'CD8+ Memory T cells',
    'CD8+ Memory T cells': 'CD8+ Memory T cells',
    'Cytotoxic T cells': 'CD8+ T cells',
    'Cytotoxic T cell': 'CD8+ T cells',

    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'CD4+ T cells': 'CD4+ T cells',
    'CD4+ T cell': 'CD4+ T cells',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ Memory T cells',
    'CD4+ Memory T cells': 'CD4+ Memory T cells',

    'double-positive, alpha-beta thymocyte': 'Double-positive thymocytes',
    'Double positive thymocyte': 'Double-positive thymocytes',
    'Double-positive thymocytes': 'Double-positive thymocytes',
    'Cortical Thymocytes': 'Cortical Thymocytes',

    'double negative thymocyte': 'Double-negative thymocytes',
    'Double negative thymocytes': 'Double-negative thymocytes',
    'Early T cells/Thymocytes': 'early T lineage precursor',

    'T cell': 'T cells',
    'T cells': 'T cells',
    'T cells (developing)': 'Developing T cells',
    'Activated T cell': 'Activated T cells',
    'Activated T cells': 'Activated T cells',

    'gamma-delta T cell': 'γδ T cells',
    'Gamma delta T cell': 'γδ T cells',
    'Gamma delta T cells': 'γδ T cells',

    # NK cells
    'NK_CD56_Dim': 'NK cells',
    'NK_CD56_Bright': 'NK cells',
    'NK_Proliferating': 'Proliferating NK cells',
    'natural killer cell': 'NK cells',  # Added mapping
    'innate lymphoid cell': 'Innate lymphoid cells',

    # Monocytes
    'Monocytes_CD14': 'Classical monocytes',
    'Monocytes_CD16': 'Non-classical monocytes',
    'classical monocyte': 'Classical monocytes',
    'non-classical monocyte': 'Non-classical monocytes',
    'intermediate monocyte': 'Intermediate monocytes',
    'monocyte': 'Monocytes',

    # T cells
    'CD4_Naive_CCR7': 'Naive CD4+ T cells',
    'CD4_TCM_AQP3': 'CD4+ central memory T cells',
    'CD4_TEM_ANXA1': 'CD4+ effector memory T cells',
    'CD4_TEM_GNLY': 'CD4+ effector T cells',
    'CD4_Treg_FOXP3': 'Regulatory T cells',
    'CD8_Naive_LEF1': 'Naive CD8+ T cells',
    'CD8_TCM_HAVCR2': 'CD8+ central memory T cells',
    'CD8_TEM_CMC1': 'CD8+ effector memory T cells',
    'CD8_TEM_GNLY': 'CD8+ effector T cells',
    'CD8_TEM_ZNF683': 'CD8+ tissue-resident memory T cells',
    'CD8_MAIT_SLC4A10': 'MAIT cells',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'CD4-positive, alpha-beta thymocyte': 'Double-positive thymocytes',
    'CD8-positive, alpha-beta thymocyte': 'Double-positive thymocytes',
    'CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell': 'CD8+ T cells',
    'alpha-beta T cell': 'T cells',
    'group 3 innate lymphoid cell': 'ILC3 cells',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'Naive CD4+ T cells',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'Naive CD8+ T cells',
    'activated CD4-positive, alpha-beta T cell': 'Activated CD4+ T cells',
    'activated CD8-positive, alpha-beta T cell': 'Activated CD8+ T cells',
    'regulatory T cell': 'Regulatory T cells',
    'T cell': 'T cells',
    'gdT': 'γδ T cells',
    'gamma-delta T cell': 'γδ T cells',
    'Gamma Delta T cells': 'γδ T cells',
    'thymocyte': 'Thymocytes',
    'mature NK T cell': 'NKT cells',

    # B cells
    'B_Memory': 'Memory B cells',
    'B_Naive': 'Naive B cells',
    'B_BCR_GNLY': 'Plasma cells',
    'B_Atypical_Memory': 'Atypical memory B cells',
    'B cell': 'B cells',
    'plasma cell': 'Plasma cells',
    'Plasma cell': 'Plasma cells',

    # Dendritic cells
    'myeloid dendritic cell': 'Myeloid dendritic cells',
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',
    'Langerhans cell': 'Dendritic cells',
    'Conventional Dendritic Cells 1 (cDC1)': 'Myeloid dendritic cells',

    # Other cells
    'Mega': 'Megakaryocytes',
    'granulocyte': 'Granulocytes',
    'neutrophil': 'Neutrophils',
    'basophil': 'Basophils',
    'mast cell': 'Mast cells',
    'macrophage': 'Macrophages',
    'colon macrophage': 'Macrophages',
    'tissue-resident macrophage': 'Macrophages',
    'mononuclear phagocyte': 'Macrophages',
    'erythrocyte': 'Erythrocytes',
    'erythroid lineage cell': 'Erythrocytes',
    'erythroid progenitor cell': 'Erythroid progenitors',
    'platelet': 'Platelets',
    'leukocyte': 'Leukocytes',
    'hematopoietic cell': 'Hematopoietic cells',
    'hematopoietic precursor cell': 'Hematopoietic cells',
    'hematopoietic stem cell': 'Hematopoietic cells',
    'common myeloid progenitor': 'Hematopoietic cells',
    'myeloid cell': 'Myeloid cells',
    'myeloid leukocyte': 'Myeloid cells',
    'Regulatory T cells (Tregs)': 'Regulatory T cells',
    'Transitional B cells': 'B cells',
    'Secretory epithelial cells': 'Secretory cells',

    # Specific dendritic cell populations
    'mDC': 'Myeloid dendritic cells',
    'pDC': 'Plasmacytoid dendritic cells',

    # Additional immune cell types
    'mononuclear phagocyte': 'Macrophages',
    'erythroid lineage cell': 'Erythroid cells',
    'hematopoietic stem cell': 'Hematopoietic cells',
    'common myeloid progenitor': 'Hematopoietic cells',
    'platelet': 'Platelets',

    # LCA dataset specific mappings
    'pulmonary alveolar type 1 cell': 'Alveolar epithelial cells type I',
    'pulmonary alveolar type 2 cell': 'Alveolar epithelial cells type II',
    'alveolar type II pneumocytes': 'Alveolar epithelial cells type II',
'Alveolar type II pneumocytes': 'Alveolar epithelial cells type II',
    'respiratory goblet cell': 'Goblet cells',
    'club cell': 'Club cells',
    'serous cell of epithelium of bronchus': 'Serous cells',
    'pulmonary ionocyte': 'Pulmonary ionocytes',
    'adventitial cell': 'Adventitial fibroblasts',
    'alveolar adventitial fibroblast': 'Adventitial fibroblasts',
    'bronchial smooth muscle cell': 'Smooth muscle cell',
    'basal cell': 'Basal epithelial cells',
    'pericyte': 'Pericytes',

    # 添加LCA数据集中缺少标准化的细胞类型
    'CD1c-positive myeloid dendritic cell': 'Myeloid dendritic cells',
    'alveolar macrophage': 'Alveolar macrophages',
    'capillary endothelial cell': 'Capillary endothelial cells',
    'ciliated cell': 'Ciliated epithelial cells',
    'Ciliated cells': 'Ciliated epithelial cells',
    'classical monocyte': 'Classical monocytes',
    'dendritic cell': 'Dendritic cells',
    'effector memory CD4-positive, alpha-beta T cell': 'CD4+ effector memory T cells',
    'effector memory CD8-positive, alpha-beta T cell': 'CD8+ effector memory T cells',
    'endothelial cell': 'Endothelial cells',
    'endothelial cell of artery': 'Arterial endothelial cells',
    'endothelial cell of lymphatic vessel': 'Lymphatic endothelial cells',
    'epithelial cell': 'Epithelial cells',  # Added mapping
    'fibroblast': 'Fibroblasts',
    'intermediate monocyte': 'Intermediate monocytes',  # 添加
    'lung ciliated cell': 'Ciliated epithelial cells',
    'lung goblet cell': 'Goblet cells',
    'lung neuroendocrine cell': 'Neuroendocrine cells',
    'lymphocyte': 'Lymphocytes',  # Added mapping
    'macrophage': 'Macrophages',  # 添加
    'mature NK T cell': 'NKT cells',  # 添加
    'megakaryocyte': 'Megakaryocytes',
    'mesothelial cell': 'Mesothelial cells',  # 添加
    'mesothelial cell of pleura': 'Mesothelial cells',
    'monocyte': 'Monocytes',  # 添加
    'mucus secreting cell': 'Goblet cells',
    'myeloid dendritic cell': 'Myeloid dendritic cells',  # 添加
    'myeloid dendritic cell, human': 'Myeloid dendritic cells',
    'myofibroblast cell': 'Myofibroblasts',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'Naive CD4+ T cells',  # 添加
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'Naive CD8+ T cells',  # 添加
    'neutrophil': 'Neutrophils',  # 添加
    'non-classical monocyte': 'Non-classical monocytes',  # 添加
    'plasma cell': 'Plasma cells',  # 添加
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',  # 添加
    'plasmacytoid dendritic cell, human': 'Plasmacytoid dendritic cells',
    'pulmonary interstitial fibroblast': 'Pulmonary fibroblasts',
    'regulatory T cell': 'Regulatory T cells',  # 添加
    'respiratory basal cell': 'Basal epithelial cells',
    'smooth muscle cell': 'Smooth muscle cells',  # 添加
    'tracheobronchial serous cell': 'Serous cells',
    'vascular associated smooth muscle cell': 'Smooth muscle cells',
    'vein endothelial cell': 'Venous endothelial cells',
    'natural killer cell': 'NK cells',  # Added mapping

    # LLM预测结果标准化
    'Activated T cell': 'Activated T cells',  # 添加单数到复数的映射
    'Arterial endothelial cell': 'Arterial endothelial cells',  # 添加单数到复数的映射
    'B cell': 'B cells',  # 添加单数到复数的映射
    'Capillary endothelial cell': 'Capillary endothelial cells',  # 添加单数到复数的映射
    'CD8+ Naive T cells': 'CD8+ Naive T cells',  # 保持一致
    'Cortical Thymocytes': 'Cortical Thymocytes',  # 保持一致
    'Cytotoxic T cell': 'Cytotoxic T cells',  # 添加单数到复数的映射
    'Cytotoxic T cells': 'Cytotoxic T cells',  # 保持一致
    'Dendritic cell': 'Dendritic cells',  # 添加单数到复数的映射
    'Dendritic Cells': 'Dendritic cells',  # 添加单数到复数的映射
    'Double positive thymocyte': 'Double-positive thymocytes',  # 添加单数到复数的映射
    'Endothelial cell': 'Endothelial cells',  # 添加单数到复数的映射
    'Fibroblast': 'Fibroblasts',  # 添加单数到复数的映射
    'Monocyte': 'Monocytes',  # 添加单数到复数的映射
    'Natural Killer (NK) cells': 'NK cells',  # 标准化NK细胞名称
    'Smooth muscle cell': 'Smooth muscle cells',  # 添加单数到复数的映射
    'T cells (developing)': 'Developing T cells',  # 标准化名称格式
        # 添加其他缺失的标准化映射
    'Atypical memory B cells': 'Atypical memory B cells',
    'Effector CD8+ T cells': 'CD8+ effector T cells',
    'Effector T cells': 'Effector T cells',
    'Exhausted T cells': 'Exhausted T cells',
    'Gamma Delta T Cells': 'γδ T cells',
    'Gamma delta (γδ) T cells': 'γδ T cells',
    'Memory T cells': 'Memory T cells',
    'Mucosal-associated invariant T (MAIT) cells': 'MAIT cells',
    'Natural Killer T Cells': 'NKT cells',
    'Proliferating Cells': 'Proliferating cells',
    # 大小写和格式标准化
'Classical Monocytes': 'Classical monocytes',  # 大小写标准化
'Effector Memory CD8+ T cell (TEM)': 'CD8+ effector memory T cells',  # 标准化格式

# 单数复数标准化
'CD8+ T cell': 'CD8+ T cells',  # 单数到复数
'CD4+ T cell': 'CD4+ T cells',  # 单数到复数

# 保持一致性的映射
'Natural Killer Cells': 'NK cells',  # 标准化NK细胞名称
'Natural Killer T Cells': 'NKT cells',  # 标准化格式
}
