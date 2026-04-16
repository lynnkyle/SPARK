import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 创建数据（根据您提供的表格）
# DB15K
# data_label = 'DB15K'
# data = {
#     'vis': [4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
#     'txt': [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32],
#     'value': [
#         37.71, 40.86, 42.51, 42.37,  # vis=4
#         37.77, 40.21, 42.35, 42.65,  # vis=8
#         38.59, 40.68, 42.67, 42.84   # vis=16 (注意: 42.84 是图片中 vis=16, txt=32 的值)
#     ]
# }

# MKGW
data_label = 'MKG-W'
data = {
    'vis': [4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
    'txt': [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32],
    'value': [
        34.36, 36.54, 40.32, 41.42,  # vis=4
        36.04, 38.45, 41.70, 41.87,  # vis=8
        37.68, 39.70, 42.11, 41.65   # vis=16
    ]
}

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
plt.figure(figsize=(8, 5))

# 使用Blues色彩映射，从浅蓝到深蓝
heatmap = sns.heatmap(
    pivot_table,
    annot=True,           # 显示数值
    fmt='.2f',            # 数值格式（保留两位小数）
    cmap='Blues',         # 从浅蓝到深蓝的配色
    annot_kws={'size': 12}, # value字体大小
    cbar_kws={'label': 'Performance Score'},  # 颜色条标签
    linewidths=0.5,       # 格子边框宽度
    linecolor='white',    # 格子边框颜色
    square=False,         # 非正方形格子
    vmin=34,              # 颜色映射的最小值
    vmax=44               # 颜色映射的最大值
)

# 设置标题和标签
plt.title(data_label, fontsize=16, pad=20)
# plt.title('MKG-W', fontsize=16, pad=20)
plt.xlabel('Number of Textual Tokens', fontsize=16)
plt.ylabel('Number of Visual Tokens', fontsize=16)

# 调整刻度标签字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 调整颜色条
cbar = heatmap.collections[0].colorbar
cbar.set_label('MRR', fontsize=16)
cbar.ax.tick_params(labelsize=12)

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