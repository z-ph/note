# 蚁群算法（Ant Colony Optimization, ACO）文档 - VRP版本

## 功能概述
蚁群算法是一种模拟蚂蚁觅食行为的优化算法，本实现针对车辆路径问题（VRP）进行了优化，主要功能包括：
1. **路径优化**：通过模拟蚂蚁的路径选择行为，找到满足载重约束的最优路径。
2. **参数自定义**：支持调整蚂蚁数量、信息素挥发系数等关键参数。
3. **可视化**：支持最优路径和算法收敛过程的可视化。
4. **动画展示**：动态展示迭代过程中最优路径的变化。

## 调用方法
### 1. 初始化蚁群优化器
```python
from ant_colony_vrp import VehicleRoutingProblem

# 定义城市坐标和需求量
cities = np.array([
    [0, 0],        # 配送中心
    [1067, 320],   # 城市1
    [1888, 690],   # 城市2
    [1978, 750],   # 城市3
])
demands = np.array([0, 10, 20, 15])  # 配送中心需求为0

# 初始化蚁群优化器
vrp = VehicleRoutingProblem(
    cities=cities,
    demands=demands,
    num_vehicles=2,  # 车辆数量
    vehicle_capacity=30,  # 车辆最大载重量
    num_ants=50,     # 蚂蚁数量
    alpha=1.0,       # 信息素重要程度因子
    beta=2.0,        # 启发式因子重要程度因子
    rho=0.5,         # 信息素挥发系数
    q=100.0,         # 信息素增加强度系数
    iterations=100   # 迭代次数
)
```

### 2. 求解最优路径
```python
best_routes, best_distance = vrp.solve()
print("最优路径:", best_routes)
print("最优距离:", best_distance)
```

### 3. 可视化结果
```python
vrp.plot_result()       # 可视化最优路径
vrp.plot_convergence()  # 可视化收敛过程
vrp.animate_iterations(interval=300)  # 动画展示迭代过程
```

## 参数说明
| 参数名 | 类型 | 描述 |
|--------|------|------|
| `cities` | `np.ndarray` | 城市坐标数组，形状为 `(n, 2)`，`n` 为城市数量 |
| `demands` | `np.ndarray` | 各城市需求量数组，配送中心需求为 `0` |
| `num_vehicles` | `int` | 车辆数量 |
| `vehicle_capacity` | `int` | 车辆最大载重量 |
| `num_ants` | `int` | 蚂蚁数量 |
| `alpha` | `float` | 信息素重要程度因子 |
| `beta` | `float` | 启发式因子重要程度因子 |
| `rho` | `float` | 信息素挥发系数（0到1之间） |
| `q` | `float` | 信息素增加强度系数 |
| `iterations` | `int` | 迭代次数 |

## 示例代码
```python
import numpy as np
from ant_colony_vrp import VehicleRoutingProblem

# 定义城市坐标和需求量
cities = np.array([
    [0, 0],        # 配送中心
    [1067, 320],   # 城市1
    [1888, 690],   # 城市2
    [1978, 750],   # 城市3
])
demands = np.array([0, 10, 20, 15])

# 初始化并求解
vrp = VehicleRoutingProblem(
    cities=cities,
    demands=demands,
    num_vehicles=2,
    vehicle_capacity=30,
    num_ants=50,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    q=100.0,
    iterations=100
)
best_routes, best_distance = vrp.solve()

# 输出结果
print("最优路径:", best_routes)
print("最优距离:", best_distance)

# 可视化
vrp.plot_result()
vrp.plot_convergence()
```

## 注意事项
1. **载重约束**：确保各路径的总需求量不超过车辆的最大载重量。
2. **参数调整**：根据问题规模调整 `num_ants` 和 `iterations` 以提高求解效率。
3. **可视化依赖**：需安装 `matplotlib` 库以支持可视化功能。