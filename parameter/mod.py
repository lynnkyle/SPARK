import matplotlib.pyplot as plt
import numpy as np

# 超参数取值
lambda_mod = np.array([0.1, 0.3, 0.5, 0.7, 1.0])

# 对应性能指标（示例数据，替换为你的真实结果）
mrr   = [0.742, 0.758, 0.754, 0.746, 0.739]
hit1  = [0.685, 0.704, 0.698, 0.691, 0.683]
hit3  = [0.782, 0.797, 0.792, 0.785, 0.778]
hit10 = [0.850, 0.858, 0.853, 0.848, 0.842]

plt.figure(figsize=(6, 4))

plt.plot(lambda_mod, mrr,   marker='o', linewidth=2, label='MRR', color='#F9B7A1')
plt.plot(lambda_mod, hit1,  marker='s', linewidth=2, label='Hit@1', color='#2164AD')
plt.plot(lambda_mod, hit3,  marker='^', linewidth=2, label='Hit@3', color="#9DBB61")
plt.plot(lambda_mod, hit10, marker='d', linewidth=2, label='Hit@10',color="#EE822F")

plt.xlabel(r'parameter $\lambda_{mod}$', fontsize=12)
plt.ylabel('metric value', fontsize=12)

plt.xticks(lambda_mod)
plt.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()
