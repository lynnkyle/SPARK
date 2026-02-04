import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="DB15K")
parser.add_argument("--metric", type=str, default="MRR")
args = parser.parse_args()

# 创建数据（根据您提供的表格）
if args.data == "DB15K" and args.metric == "MRR":
    data = {
        'vis': [4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
        'txt': [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32],
        'value': [37.71, 41.86, 43.51, 43.37, 37.77, 42.21, 43.35, 43.65, 38.59, 42.68, 43.67, 43.84]
    }
elif args.data == "DB15K" and args.metric == "H@1":
    data = {
        'vis': [4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
        'txt': [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32],
        'value': [31.16, 35.17, 36.46, 36.31, 31.23, 35.48, 36.38, 36.47, 32.19, 35.88, 36.57, 36.83]
    }
else:
    raise NotImplementedError

# 转换为DataFrame
df = pd.DataFrame(data)

# 透视数据以适应热力图格式
pivot_table = df.pivot(index='vis', columns='txt', values='value')

# 由于数据不是完整的矩阵，我们需要填充缺失值
# 用NaN填充缺失的组合，或者用插值法处理
pivot_table = pivot_table.reindex(index=[4, 8, 16], columns=[4, 8, 16, 32])

print("透视表数据：")
print(pivot_table)

# 绘制热力图 - 使用从浅蓝到深蓝的配色
plt.figure(figsize=(10, 6))

# 使用Blues色彩映射，从浅蓝到深蓝
if args.metric == "MRR":
    cmap = 'Blues'
    vmin, vmax = 37, 45
else:
    cmap = 'Greens'
    vmin, vmax = 30, 38
heatmap = sns.heatmap(
    pivot_table,
    annot=True,  # 显示数值
    fmt='.2f',  # 数值格式（保留两位小数）
    cmap=cmap,  # 从浅蓝到深蓝的配色
    cbar_kws={'label': 'Performance Score'},  # 颜色条标签
    linewidths=0.5,  # 格子边框宽度
    linecolor='white',  # 格子边框颜色
    square=False,  # 非正方形格子
    vmin=vmin,  # 颜色映射的最小值
    vmax=vmax  # 颜色映射的最大值
)

# 设置标题和标签
plt.title(f'{args.data}', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Number of Textual Tokens', fontsize=12, fontweight='bold')
plt.ylabel('Number of Visual Tokens', fontsize=12, fontweight='bold')

# 调整刻度标签字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整颜色条
cbar = heatmap.collections[0].colorbar
cbar.set_label(f'{args.metric}', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# 添加网格线（可选，增强可读性）
plt.grid(False)

# 调整布局
plt.tight_layout()

# 保存高分辨率图片（适合论文使用）
plt.savefig('academic_heatmap_blues.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('academic_heatmap_blues.pdf', bbox_inches='tight', facecolor='white')  # PDF矢量图

# 显示图表
plt.show()

# 打印一些统计信息
print(f"\n数据统计:")
print(f"最小值: {df['value'].min():.2f}")
print(f"最大值: {df['value'].max():.2f}")
print(f"平均值: {df['value'].mean():.2f}")
