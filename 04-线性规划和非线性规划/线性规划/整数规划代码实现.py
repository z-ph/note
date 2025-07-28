# 为便于本地运行，移除了numpy依赖
from pulp import LpMaximize, LpProblem, LpVariable

# 创建最大化问题
prob = LpProblem("手机生产计划", LpMaximize)

# 定义决策变量（整数型）
x = LpVariable("A型号产量", lowBound=0, cat='Integer')  # 整数约束
y = LpVariable("B型号产量", lowBound=0, cat='Integer')  # 整数约束

# 定义目标函数
prob += 500*x + 400*y, "总利润"

# 添加约束条件
prob += 2*x + 3*y <= 120, "芯片约束"
prob += 3*x + 2*y <= 100, "电池约束"
prob += 4*x + 3*y <= 140, "人工约束"
prob += x >= 10, "最低A产量约束"
prob += y <= 1.5*x, "B产量上限约束"

# 求解问题
prob.solve()

# 输出结果
print("求解状态：", "成功" if prob.status == 1 else "失败")
print(f"最优产量：A型号手机 {x.value()} 部，B型号手机 {y.value()} 部")
print(f"最大周利润：{prob.objective.value()} 元")    