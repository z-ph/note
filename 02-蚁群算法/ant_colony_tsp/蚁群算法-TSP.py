import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import sys
import os

# 添加包路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ant_colony_tsp import AntColonyOptimizer

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 使用黑体，若没有该字体，可根据系统情况替换，如 'Microsoft YaHei'
# 解决负号显示问题
rcParams['axes.unicode_minus'] = False

# 中国主要城市TSP问题实例
if __name__ == "__main__":
    # 城市坐标数据（单位：千米）
    cities = np.array([
        [0, 0],        # 北京
        [1067, 320],   # 上海
        [1888, 690],   # 广州
        [1978, 750],   # 深圳
        [1490, 1010],  # 重庆
        [1510, 1140],  # 成都
        [960, 700],    # 武汉
        [1120, 400],   # 杭州
        [1000, 440],   # 南京
        [920, 1000]    # 西安
    ])
    
    # 城市名称
    city_names = ["北京", "上海", "广州", "深圳", "重庆", "成都", "武汉", "杭州", "南京", "西安"]
    
    # 设置参数
    num_ants = 50
    alpha = 1.0      # 信息素重要程度
    beta = 2.0       # 启发式因子重要程度
    rho = 0.5        # 信息素挥发系数
    q = 100.0        # 信息素增加强度系数
    iterations = 100
    
    # 创建蚁群优化器并求解
    aco = AntColonyOptimizer(cities, num_ants, alpha, beta, rho, q, iterations)
    best_path, best_distance = aco.solve()
    
    # 输出最优路径
    print("\n最优路径:")
    path_str = " → ".join([city_names[i] for i in best_path]) + " → " + city_names[best_path[0]]
    print(path_str)
    print(f"最优路径距离: {best_distance:.2f} 千米")
    
    # 可视化结果
    aco.plot_result(city_names)
    aco.plot_convergence()
    
    # 动画展示迭代过程
    aco.animate_iterations(city_names, interval=300)    