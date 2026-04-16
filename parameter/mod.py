import matplotlib.pyplot as plt
import numpy as np

# 超参数取值
# lambda_mod
lambda_mod = np.array([0.05, 0.5, 5, 50])
# lambda_ent
# lambda_mod = np.array([0.01, 0.1, 1, 10, 100])

# 对应性能指标（示例数据，替换为你的真实结果）
# DB15K $\lambda_{mod}$
mrr = [40.57, 41.76, 41.34, 40.26]
hit1 = [32.43, 33.39, 33.16, 32.29]
# DB15K $\lambda_{ent}$
# mrr = [41.27, 41.53, 41.76, 42.67, 41.84]
# hit1 = [32.73, 33.57, 33.39, 34.81, 33.68]

plt.figure(figsize=(8, 5))

plt.xscale('log')
plt.plot(lambda_mod, mrr, marker='o', linewidth=2, label='MRR', color='#F9B7A1')
plt.plot(lambda_mod, hit1, marker='s', linewidth=2, label='Hit@1', color='#2164AD')
# plt.plot(lambda_mod, hit3,  marker='^', linewidth=2, label='Hit@3', color="#9DBB61")
# plt.plot(lambda_mod, hit10, marker='d', linewidth=2, label='Hit@10',color="#EE822F")

plt.title('DB15K', fontsize=16)
plt.xlabel(r'$\lambda_{mod}$', fontsize=16)
plt.ylabel('performance (%)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xticks(lambda_mod, [r'0.05', r'0.5', r'5', r'50'])
# plt.xticks(lambda_mod, [r'0.01', r'0.1', r'1', r'10', r'100'])
plt.legend(fontsize=12, loc='right')

plt.tight_layout()
plt.show()
