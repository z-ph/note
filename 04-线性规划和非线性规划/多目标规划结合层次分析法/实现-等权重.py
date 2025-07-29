from pulp import LpMaximize, LpProblem, LpVariable

# 创建最大化问题
prob = LpProblem("交通系统优化", LpMaximize)

# 定义决策变量（二进制变量）
x = [LpVariable(f"x{i+1}", cat='Binary') for i in range(6)]  # x1~x6 分别表示L1~L6

# 线路参数
costs = [8, 12, 10, 15, 9, 11]          # 建设成本（亿元）
passengers = [12, 18, 15, 22, 14, 16]   # 日均客流量（万人次）
emissions = [5000, 7500, 6000, 9000, 5500, 7000]  # 减排量（吨/年）
periods = [18, 24, 20, 30, 16, 22]      # 施工周期（月）

# 计算各目标函数的理想最优值和最差值（简化计算，假设所有线路可选时为最优）
f1_max = sum(passengers)  # 最大客流量
f1_min = 0                # 最小客流量
f2_max = sum(emissions)   # 最大减排量
f2_min = 0                # 最小减排量
f3_min = min(costs)       # 最小成本
f3_max = sum(costs)       # 最大成本

# 定义目标函数：加权组合
w1 = w2 = w3 = 1/3  # 假设三个目标权重相等
normalized_f1 = sum(passengers[i] * x[i] for i in range(6)) / f1_max
normalized_f2 = sum(emissions[i] * x[i] for i in range(6)) / f2_max
normalized_f3 = sum(costs[i] * x[i] for i in range(6)) / f3_max

prob += w1 * normalized_f1 + w2 * normalized_f2 - w3 * normalized_f3, "综合目标"

# 添加约束条件
prob += sum(costs[i] * x[i] for i in range(6)) <= 30, "预算约束"
prob += sum(periods[i] * x[i] for i in range(6)) <= 48, "施工周期约束"
prob += x[1] <= x[4], "选L2则必须选L5"
prob += x[2] + x[3] <= 1, "L3和L4最多选一个"
prob += sum(x) >= 2, "至少选两条线路"

# 求解问题
from pulp import PULP_CBC_CMD  # 导入求解器
prob.solve(PULP_CBC_CMD(msg=False))  # 设置 msg=False 关闭详细日志


# 输出结果
if prob.status == 1:  # 求解成功
    print("最优线路组合：")
    selected_lines = [i+1 for i in range(6) if x[i].value() == 1]
    total_cost = sum(costs[i] for i in selected_lines if x[i].value() == 1)
    total_passengers = sum(passengers[i] for i in selected_lines if x[i].value() == 1)
    total_emissions = sum(emissions[i] for i in selected_lines if x[i].value() == 1)
    total_period = sum(periods[i] for i in selected_lines if x[i].value() == 1)
    
    for line in selected_lines:
        print(f"选择线路L{line}（成本：{costs[line-1]}亿元，客流量：{passengers[line-1]}万人次/日，减排：{emissions[line-1]}吨/年）")
    
    print(f"\n总建设成本：{total_cost}亿元")
    print(f"日均总客流量：{total_passengers}万人次")
    print(f"年总减排量：{total_emissions}吨")
    print(f"总施工周期：{total_period}个月")
else:
    print("求解失败，原因：", prob.status)