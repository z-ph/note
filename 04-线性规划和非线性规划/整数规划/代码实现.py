from pulp import LpMaximize, LpProblem, LpVariable

# 创建最大化问题
prob = LpProblem("设备采购优化", LpMaximize)

# 定义决策变量（二进制变量）
x = [LpVariable(f"x{i+1}", cat='Binary') for i in range(5)]  # x1~x5 分别表示设备A~E

# 设备参数
costs = [12, 8, 10, 5, 15]        # 采购成本
capacities = [8, 6, 7, 3, 10]     # 运输能力
maintenance = [0.5, 0.4, 0.6, 0.2, 0.7]  # 维护费用

# 定义目标函数：总运输能力
prob += sum(capacities[i] * x[i] for i in range(5)), "总运输能力"

# 添加约束条件
prob += sum(costs[i] * x[i] for i in range(5)) <= 50, "预算约束"
prob += sum(maintenance[i] * x[i] for i in range(5)) <= 3, "维护费用约束"
prob += sum(x) >= 2, "至少选2种设备"
prob += x[1] <= x[3], "选B则必须选D"
prob += x[0] + x[4] <= 1, "A和E最多选一个"

# 求解问题
prob.solve()

# 输出结果
if prob.status == 1:  # 求解成功
    print("最优采购方案：")
    total_cost = sum(costs[i] * x[i].value() for i in range(5))
    total_maintenance = sum(maintenance[i] * x[i].value() for i in range(5))
    total_capacity = sum(capacities[i] * x[i].value() for i in range(5))
    
    for i in range(5):
        if x[i].value() == 1:
            print(f"选择设备{chr(65+i)}（采购成本：{costs[i]}万元，运输能力：{capacities[i]}吨/日）")
    
    print(f"\n总采购成本：{total_cost}万元")
    print(f"每日维护费用：{total_maintenance}万元")
    print(f"每日总运输能力：{total_capacity}吨")
else:
    print("求解失败，原因：", prob.status)