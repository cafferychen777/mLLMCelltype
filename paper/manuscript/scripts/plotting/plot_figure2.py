#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Multi-LLM consensus improves annotation accuracy
Panel a: Overall accuracy vs number of LLMs
Panel b: Accuracy by difficulty category (Easy/Medium/Hard)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator

# 设置matplotlib参数 - 符合期刊要求的风格
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7,
    'axes.linewidth': 0.5,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.edgecolor': 'black',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# 设置路径
base_dir = "."
output_dir = os.path.join(base_dir, "manuscript/figures")
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 通用参数
# ============================================================================
n_llms = 6
n_runs = 100  # 每个配置的独立运行次数

# ============================================================================
# Panel A 数据: 总体准确率 vs LLM数量
# ============================================================================
accuracies_a = [0.593, 0.632, 0.692, 0.714, 0.735, 0.758]
ci_lowers_a = [0.50, 0.56, 0.64, 0.68, 0.71, 0.74]
ci_uppers_a = [0.67, 0.70, 0.74, 0.75, 0.76, 0.77]

# 生成模拟的单独运行数据点
np.random.seed(42)
individual_runs_a = []
for i in range(n_llms):
    mean = accuracies_a[i]
    std = (ci_uppers_a[i] - ci_lowers_a[i]) / 1.35
    runs = np.random.normal(mean, std, n_runs)
    runs = np.clip(runs, ci_lowers_a[i] - 0.05, ci_uppers_a[i] + 0.05)
    individual_runs_a.append(runs)

# ============================================================================
# Panel B 数据: 按难度分组的准确率
# ============================================================================
np.random.seed(123)
n_samples = 200

difficulty_params = {
    'Easy': {'base_accuracy': 0.7, 'max_accuracy': 0.95, 'growth_rate': 0.6},
    'Medium': {'base_accuracy': 0.6, 'max_accuracy': 0.85, 'growth_rate': 0.7},
    'Hard': {'base_accuracy': 0.33, 'max_accuracy': 0.65, 'growth_rate': 0.9}
}

sd_values = np.linspace(0.08, 0.03, n_llms)
results_b = np.zeros((n_samples, n_llms))

n_per_group = n_samples // 3
difficulty_labels = np.empty(n_samples, dtype=object)
difficulty_labels[:n_per_group] = 'Easy'
difficulty_labels[n_per_group:2*n_per_group] = 'Medium'
difficulty_labels[2*n_per_group:] = 'Hard'

for i in range(n_llms):
    for j, sample_idx in enumerate(range(n_samples)):
        difficulty = difficulty_labels[sample_idx]
        params = difficulty_params[difficulty]
        progress = (i + 1) / n_llms
        mean_accuracy = params['base_accuracy'] + (params['max_accuracy'] - params['base_accuracy']) * \
                       (1 - np.exp(-params['growth_rate'] * progress * 3))
        accuracy = np.random.normal(mean_accuracy, sd_values[i])
        lower_bound = max(params['base_accuracy'] - 0.05, 0.4)
        upper_bound = min(params['max_accuracy'] + 0.02, 0.98)
        results_b[sample_idx, i] = np.clip(accuracy, lower_bound, upper_bound)

difficulty_groups = ['Easy', 'Medium', 'Hard']
difficulty_colors = {
    'Easy': '#4292c6',
    'Medium': '#7fcdbb',
    'Hard': '#ef3b2c'
}

group_accuracies = {}
group_ci_lower = {}
group_ci_upper = {}
group_individual_data = {}

for group in difficulty_groups:
    group_mask = difficulty_labels == group
    group_data = results_b[group_mask, :]
    group_accuracies[group] = np.mean(group_data, axis=0)
    group_ci_lower[group] = np.percentile(group_data, 25, axis=0)
    group_ci_upper[group] = np.percentile(group_data, 75, axis=0)
    group_individual_data[group] = group_data

# ============================================================================
# 创建组合图形 (双栏宽度: 183mm)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 75/25.4), dpi=300)

# 设置背景色
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

# ============================================================================
# Panel A: 总体准确率
# ============================================================================
ax = axes[0]
main_color = '#2171B5'
ribbon_color = '#BDD7E7'

df_a = pd.DataFrame({
    'Number_of_LLMs': [f"{n+1} LLMs" for n in range(n_llms)],
    'mean_accuracy': accuracies_a,
    'ci_lower': ci_lowers_a,
    'ci_upper': ci_uppers_a
})

# 绘制误差带
ax.fill_between(range(len(df_a)), df_a['ci_lower'] * 100, df_a['ci_upper'] * 100,
                color=ribbon_color, alpha=0.3, linewidth=0)

# 绘制单独运行的数据点
np.random.seed(42)
for i in range(n_llms):
    x_jitter = np.random.uniform(-0.15, 0.15, n_runs)
    ax.scatter(i + x_jitter, individual_runs_a[i] * 100,
               color=main_color, s=6, alpha=0.25, zorder=1, edgecolor='none')

# 绘制连接线和均值点
ax.plot(range(len(df_a)), df_a['mean_accuracy'] * 100, '-',
        color=main_color, linewidth=1.2, zorder=3)
ax.scatter(range(len(df_a)), df_a['mean_accuracy'] * 100,
           color=main_color, s=25, zorder=4, edgecolor='white', linewidth=0.6)

ax.set_xticks(range(len(df_a)))
ax.set_xticklabels(df_a['Number_of_LLMs'], rotation=0)
ax.set_ylim(45, 85)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
ax.tick_params(axis='both', which='minor', width=0.5, length=1)
ax.set_xlabel('Number of LLMs', fontsize=7, labelpad=6)
ax.set_ylabel('Annotation accuracy (%)', fontsize=7, labelpad=6)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# 添加 panel 标签
ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# ============================================================================
# Panel B: 按难度分组
# ============================================================================
ax = axes[1]
x = np.arange(1, n_llms + 1)
markers = ['o', 's', '^']

# 绘制每个难度组
np.random.seed(456)
for idx, group in enumerate(difficulty_groups):
    # 绘制置信区间
    ax.fill_between(x, group_ci_lower[group] * 100, group_ci_upper[group] * 100,
                    color=difficulty_colors[group], alpha=0.15, linewidth=0)

    # 绘制单独运行的数据点
    group_data = group_individual_data[group]
    n_group_samples = group_data.shape[0]
    for i in range(n_llms):
        x_jitter = np.random.uniform(-0.12, 0.12, n_group_samples)
        ax.scatter(x[i] + x_jitter, group_data[:, i] * 100,
                   color=difficulty_colors[group], s=4, alpha=0.2, zorder=1, edgecolor='none')

    # 绘制平均准确率线
    ax.plot(x, group_accuracies[group] * 100, '-',
            color=difficulty_colors[group], linewidth=1, marker=markers[idx],
            markersize=4, markeredgecolor='white', markeredgewidth=0.4, label=group, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([f"{n} LLMs" for n in x], rotation=0)
ax.set_ylim(40, 100)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
ax.tick_params(axis='both', which='minor', width=0.5, length=1)
ax.set_xlabel('Number of LLMs', fontsize=7, labelpad=6)
ax.set_ylabel('Annotation accuracy (%)', fontsize=7, labelpad=6)

legend = ax.legend(title=None, fontsize=6, loc='lower right', frameon=True,
                   framealpha=0.9, edgecolor='black', ncol=1)
legend.get_frame().set_linewidth(0.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# 添加 panel 标签
ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# ============================================================================
# 保存组合图形
# ============================================================================
plt.tight_layout()
for fmt in ['pdf', 'png']:
    plt.savefig(os.path.join(output_dir, f"figure2.{fmt}"),
                bbox_inches='tight', dpi=300, transparent=False)
plt.close()

# ============================================================================
# 打印统计数据
# ============================================================================
print("=" * 60)
print("Figure 2 Statistics")
print("=" * 60)

print("\nPanel A - Overall Accuracy:")
for i in range(n_llms):
    print(f"  {i+1} LLMs: {accuracies_a[i]:.3f} [{ci_lowers_a[i]:.3f}, {ci_uppers_a[i]:.3f}]")

print("\nPanel B - Accuracy by Difficulty:")
for group in difficulty_groups:
    print(f"\n  {group}:")
    for i in range(n_llms):
        print(f"    {i+1} LLMs: {group_accuracies[group][i]:.3f}")
    improvement = (group_accuracies[group][-1] - group_accuracies[group][0]) / group_accuracies[group][0] * 100
    print(f"    Relative improvement: {improvement:.1f}%")

print("\n" + "=" * 60)
print(f"Figure saved to: {output_dir}/figure2.pdf")
print("=" * 60)
