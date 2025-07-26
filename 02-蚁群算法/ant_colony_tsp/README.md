# 蚁群算法（Ant Colony Optimization, ACO）文档

## 功能概述
蚁群算法是一种模拟蚂蚁觅食行为的优化算法，主要用于解决旅行商问题（TSP）。本实现提供了以下功能：
- 初始化蚁群优化器，支持自定义参数（如蚂蚁数量、信息素挥发系数等）。
- 计算城市间的距离矩阵。
- 蚂蚁路径构建与信息素更新。
- 最优路径求解与可视化。
- 算法收敛过程的可视化。
- 迭代过程的动画展示。

## 类与方法说明
### `AntColonyOptimizer` 类
#### 初始化参数
- `cities`: 城市坐标数组，形状为 `(n, 2)`，`n` 为城市数量。
- `num_ants`: 蚂蚁数量。
- `alpha`: 信息素重要程度因子。
- `beta`: 启发式因子重要程度因子。
- `rho`: 信息素挥发系数。
- `q`: 信息素增加强度系数。
- `iterations`: 迭代次数。

#### 主要方法
1. `solve()`: 执行蚁群优化算法，返回最优路径和最优路径距离。
2. `plot_result(city_names=None)`: 可视化最优路径。
3. `plot_convergence()`: 可视化算法收敛过程。
4. `animate_iterations(city_names=None, interval=200)`: 动画展示迭代过程中最优路径的变化。

## 调用示例
```python
import numpy as np
from ant_colony import AntColonyOptimizer

# 定义城市坐标
cities = np.array([
    [0, 0],
    [1, 2],
    [3, 1],
    [4, 3],
    [2, 4]
])

# 初始化蚁群优化器
aco = AntColonyOptimizer(
    cities=cities,
    num_ants=10,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    q=100,
    iterations=100
)

# 求解最优路径
best_path, best_distance = aco.solve()
print(f"最优路径: {best_path}")
print(f"最优路径距离: {best_distance}")

# 可视化结果
aco.plot_result()
aco.plot_convergence()
```

## 可视化功能
1. **最优路径可视化**：调用 `plot_result()` 方法，绘制包含城市点和最优路径的图表。
2. **收敛过程可视化**：调用 `plot_convergence()` 方法，绘制每轮迭代的最优路径距离曲线。
3. **动画展示**：调用 `animate_iterations()` 方法，动态展示迭代过程中最优路径的变化。