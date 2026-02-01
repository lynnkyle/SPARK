import numpy as np
import matplotlib.pyplot as plt

# x轴的参数取值
margin = np.array([0.4, 0.5, 0.6, 0.7])

# 各指标
mrr = [0.754, 0.758, 0.751, 0.754]
hit1 = [0.697, 0.704, 0.693, 0.698]
hit3 = [0.795, 0.797, 0.791, 0.792]
hit10 = [0.853, 0.858, 0.852, 0.850]

# 柱子宽度
bar_width = 0.2
x = np.arange(len(margin))

plt.figure(figsize=(8, 5))

plt.bar(x - 1.5 * bar_width, mrr, width=bar_width, label='MRR', color="#ECD7C1")
plt.bar(x - 0.5 * bar_width, hit1, width=bar_width, label='Hit@1', color="#B3CDCB")
plt.bar(x + 0.5 * bar_width, hit3, width=bar_width, label='Hit@3', color="#E6C3BD")
plt.bar(x + 1.5 * bar_width, hit10, width=bar_width, label='Hit@10', color="#A9B2CA")

# 坐标轴
plt.xlabel('paramter margin')
plt.ylabel('metric value')
plt.xticks(x, margin)
plt.ylim(0.68, 0.87)

# 图例
plt.legend()

# 网格（可选，论文里一般开）
# plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()