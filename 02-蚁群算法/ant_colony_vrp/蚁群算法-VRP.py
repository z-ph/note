import numpy as np
from ant_colony_vrp import VehicleRoutingProblem
# 生鲜配送VRP问题实例
if __name__ == "__main__":
    # 城市坐标数据（单位：千米）
    cities = np.array([
        [0, 0],        # 北京（配送中心）
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

    # 需求量数据（单位：吨）
    demands = np.array([0, 25, 30, 20, 35, 25, 40, 20, 30, 25])

    # 设置参数
    num_vehicles = 3
    vehicle_capacity = 100
    num_ants = 50
    alpha = 1.0      # 信息素重要程度
    beta = 2.0       # 启发式因子重要程度
    rho = 0.5        # 信息素挥发系数
    q = 100.0        # 信息素增加强度系数
    iterations = 100

    # 创建VRP求解器并求解
    vrp = VehicleRoutingProblem(cities, demands, num_vehicles, vehicle_capacity,
                                 num_ants, alpha, beta, rho, q, iterations)
    best_routes, best_distance = vrp.solve()

    # 输出最优路径
    print("\n最优配送方案:")
    for i, route in enumerate(best_routes):
        city_names_route = [vrp.city_names[j] for j in route]
        route_str = " → ".join(city_names_route)
        total_demand = sum(demands[j] for j in route)
        route_distance = vrp._calculate_route_distance(route)
        print(f"车辆 {i + 1}: {route_str}")
        print(f"  载重: {total_demand:.1f}吨 (容量: {vehicle_capacity}吨)")
        print(f"  距离: {route_distance:.2f}千米")

    print(f"\n总行驶距离: {best_distance:.2f}千米")

    # 可视化结果
    vrp.plot_result()
    vrp.plot_convergence()

    # 动画展示迭代过程
    vrp.animate_iterations(interval=300)    