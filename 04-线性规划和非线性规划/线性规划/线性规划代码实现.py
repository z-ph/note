
import numpy as np
from scipy.optimize import linprog

# 1. 定义目标函数系数（因求max z=500x+400y，等价于min -z = -500x -400y）
c = [-500, -400]  # 对应x和y的系数

# 2. 定义不等式约束（Ax <= b）
# 约束1：2x + 3y <= 120
# 约束2：3x + 2y <= 100
# 约束3：4x + 3y <= 140
# 约束5：y - 1.5x <= 0 （即y <= 1.5x）
A = [
    [2, 3],    # 约束1
    [3, 2],    # 约束2
    [4, 3],    # 约束3
    [-1, 0],   # 约束4：-x <= -10（等价于x >=10）
    [-1.5, 1]  # 约束5：-1.5x + y <=0（等价于y <=1.5x）
]
b = [120, 100, 140, -10, 0]  # 对应约束的右侧值

# 3. 定义变量边界（x >=0, y >=0）
x_bounds = (0, None)
y_bounds = (0, None)
bounds = [x_bounds, y_bounds]

# 4. 求解线性规划
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# 5. 输出结果
print("求解状态：", "成功" if result.success else "失败")
print(f"最优产量：A型号手机 {round(result.x[0], 2)} 部，B型号手机 {round(result.x[1], 2)} 部")
print(f"最大周利润：{round(-result.fun, 2)} 元")  # 还原为正利润