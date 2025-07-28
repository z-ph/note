import numpy as np
from scipy.optimize import minimize

# 1. 定义目标函数（投资组合方差）
def portfolio_variance(weights):
    x, y, z = weights
    # variances = np.array([0.04, 0.01, 0.0225])
    covariances = np.array([
        [0.04, 0.006, 0.009],
        [0.006, 0.01, 0.003],
        [0.009, 0.003, 0.0225]
    ])
    return np.dot(weights, np.dot(covariances, weights))

# 2. 定义约束条件
# 约束1：预期收益率 >= 15%
def return_constraint(weights):
    returns = np.array([0.2, 0.08, 0.12])
    return np.dot(weights, returns) - 0.15  # >=0

# 约束2：权重和为1
def weight_constraint(weights):
    return np.sum(weights) - 1  # =0

# 3. 设置约束条件字典
# type: 'eq' 表示等于0,'ineq'表示大于等于0
constraints = [
    {'type': 'ineq', 'fun': return_constraint},  # 收益率约束
    {'type': 'eq', 'fun': weight_constraint}     # 权重和约束
]

# 4. 设置变量边界（非负约束）
bounds = [(0, None), (0, None), (0, None)]  # 每个权重 >=0

# 5. 设置初始猜测值（等权重）
initial_guess = [1/3, 1/3, 1/3]

# 6. 求解非线性规划
result = minimize(
    portfolio_variance,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# 7. 输出结果
if result.success:
    weights = result.x
    risk = result.fun
    expected_return = np.dot(weights, np.array([0.2, 0.08, 0.12]))
    print("最优投资组合：")
    print(f"股票比例：{weights[0]:.4f} ({weights[0]*100:.2f}%)")
    print(f"债券比例：{weights[1]:.4f} ({weights[1]*100:.2f}%)")
    print(f"黄金比例：{weights[2]:.4f} ({weights[2]*100:.2f}%)")
    print(f"预期年化收益率：{expected_return*100:.2f}%")
    print(f"投资组合风险（方差）：{risk:.6f}")
    print(f"投资组合标准差（风险）：{np.sqrt(risk)*100:.2f}%")
else:
    print("求解失败，原因：", result.message)