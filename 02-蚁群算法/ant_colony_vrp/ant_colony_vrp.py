import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict
from matplotlib import rcParams


from matplotlib.animation import FuncAnimation
# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 使用黑体，若没有该字体，可根据系统情况替换，如 'Microsoft YaHei'
# 解决负号显示问题
rcParams['axes.unicode_minus'] = False
class VehicleRoutingProblem:
    def __init__(self, cities: np.ndarray, demands: np.ndarray, num_vehicles: int, vehicle_capacity: int,
                 num_ants: int, alpha: float, beta: float, rho: float, q: float, iterations: int):
        """
        初始化车辆路径规划问题
        :param cities: 城市坐标数组
        :param demands: 各城市需求量数组
        :param num_vehicles: 车辆数量
        :param vehicle_capacity: 车辆最大载重量
        :param num_ants: 蚂蚁数量
        :param alpha: 信息素重要程度因子
        :param beta: 启发式因子重要程度因子
        :param rho: 信息素挥发系数
        :param q: 信息素增加强度系数
        :param iterations: 迭代次数
        """
        self.num_cities = len(cities)
        self.cities = cities
        self.demands = demands
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.iterations = iterations
        self.city_names = ["北京", "上海", "广州", "深圳", "重庆", "成都", "武汉", "杭州", "南京", "西安"]

        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        np.fill_diagonal(self.pheromone_matrix, 0)  # 对角线元素为0

        # 记录最优路径和最优距离
        self.best_routes = None
        self.best_distance = float('inf')

        # 记录每轮迭代的最优路径和距离
        self.iteration_best_routes = []
        self.iteration_best_distances = []

    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算城市间的距离矩阵"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt(((self.cities[i] - self.cities[j]) ** 2).sum())
                matrix[i, j] = dist
                matrix[j, i] = dist  # 对称矩阵
        return matrix

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算单条路径的总距离"""
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        distance += self.distance_matrix[route[-1], route[0]]  # 返回起点
        return distance

    def _calculate_total_distance(self, routes: List[List[int]]) -> float:
        """计算所有路径的总距离"""
        total_distance = 0
        for route in routes:
            total_distance += self._calculate_route_distance(route)
        return total_distance

    def _is_valid_route(self, route: List[int]) -> bool:
        """检查路径是否满足载重约束"""
        total_demand = sum(self.demands[i] for i in route)
        return total_demand <= self.vehicle_capacity

    def _select_next_city(self, current_city: int, unvisited_cities: List[int], remaining_capacity: float) -> int:
        """蚂蚁选择下一个城市（考虑载重约束）"""
        valid_cities = [city for city in unvisited_cities if self.demands[city] <= remaining_capacity]
        if not valid_cities:
            return 0  # 返回配送中心

        probabilities = []
        for city in valid_cities:
            # 计算转移概率
            pheromone = self.pheromone_matrix[current_city, city] ** self.alpha
            heuristic = (1.0 / self.distance_matrix[current_city, city]) ** self.beta
            probabilities.append(pheromone * heuristic)

        # 归一化概率
        if sum(probabilities) == 0:
            return valid_cities[0]  # 避免概率和为0的情况
        probabilities = [p / sum(probabilities) for p in probabilities]

        # 根据概率选择下一个城市
        return np.random.choice(valid_cities, p=probabilities)

    def solve(self) -> Tuple[List[List[int]], float]:
        """执行蚁群优化算法求解VRP"""
        self.distance_matrix = self._calculate_distance_matrix()
        for iteration in range(self.iterations):
            all_routes = []
            all_distances = []
            iteration_best_distance = float('inf')
            iteration_best_routes = None

            # 每只蚂蚁构建路径
            for ant in range(self.num_ants):
                unvisited_cities = list(range(1, self.num_cities))
                routes = []
                for _ in range(self.num_vehicles):
                    route = [0]  # 从配送中心出发
                    remaining_capacity = self.vehicle_capacity
                    while unvisited_cities:
                        next_city = self._select_next_city(route[-1], unvisited_cities, remaining_capacity)
                        if next_city == 0:
                            break
                        route.append(next_city)
                        unvisited_cities.remove(next_city)
                        remaining_capacity -= self.demands[next_city]
                    route.append(0)  # 回到配送中心
                    routes.append(route)

                # 计算路径距离
                total_distance = self._calculate_total_distance(routes)
                all_routes.append(routes)
                all_distances.append(total_distance)

                # 更新迭代最优解
                if total_distance < iteration_best_distance:
                    iteration_best_distance = total_distance
                    iteration_best_routes = routes.copy()

                # 更新全局最优解
                if total_distance < self.best_distance:
                    self.best_distance = total_distance
                    self.best_routes = routes.copy()

            # 记录本轮迭代的最优路径和距离
            self.iteration_best_routes.append(iteration_best_routes)
            self.iteration_best_distances.append(iteration_best_distance)

            # 更新信息素矩阵
            self._update_pheromone_matrix(all_routes, all_distances)

        return self.best_routes, self.best_distance

    def _update_pheromone_matrix(self, all_routes: List[List[List[int]]], all_distances: List[float]):
        """更新信息素矩阵"""
        # 信息素挥发
        self.pheromone_matrix *= (1 - self.rho)

        # 信息素增加
        for routes, distance in zip(all_routes, all_distances):
            pheromone_to_add = self.q / distance
            for route in routes:
                for i in range(len(route) - 1):
                    self.pheromone_matrix[route[i], route[i + 1]] += pheromone_to_add
                    self.pheromone_matrix[route[i + 1], route[i]] += pheromone_to_add  # 对称更新
                # 从最后一个城市回到起点
                self.pheromone_matrix[route[-1], route[0]] += pheromone_to_add
                self.pheromone_matrix[route[0], route[-1]] += pheromone_to_add

    def plot_result(self):
        """可视化最优路径"""
        if self.best_routes is None:
            print("请先运行solve()方法求解VRP")
            return

        plt.figure(figsize=(12, 10))
        # 绘制城市点
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=70)

        # 添加城市名称标签
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(self.city_names[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)

        # 绘制最优路径
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.best_routes)))
        for i, route in enumerate(self.best_routes):
            path = route
            plt.plot(self.cities[path, 0], self.cities[path, 1], color=colors[i], linewidth=2, label=f'车辆 {i + 1}')

        # 添加距离信息
        plt.title(f'总行驶距离: {self.best_distance:.2f}')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """可视化算法收敛过程"""
        if not self.iteration_best_distances:
            print("请先运行solve()方法求解VRP")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.iterations + 1), self.iteration_best_distances, 'b-', linewidth=2)
        plt.axhline(y=self.best_distance, color='r', linestyle='--', label=f'全局最优: {self.best_distance:.2f}')

        plt.title('蚁群算法收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('最优路径距离')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def animate_iterations(self, interval=200):
        """
        动画展示迭代过程中最优路径的变化
        :param interval: 帧间隔时间（毫秒）
        """
        if not self.iteration_best_routes:
            print("请先运行solve()方法求解VRP")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制城市点（只绘制一次）
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=70)

        # 添加城市名称标签
        for i, (x, y) in enumerate(self.cities):
            ax.annotate(self.city_names[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)

        # 初始化路径线和标题
        lines = []
        for _ in range(self.num_vehicles):
            line, = ax.plot([], [], linewidth=2)
            lines.append(line)
        title = ax.set_title('')

        def init():
            for line in lines:
                line.set_data([], [])
            title.set_text('')
            return lines + [title]

        def update(frame):
            routes = self.iteration_best_routes[frame]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
            for i, route in enumerate(routes):
                path = route
                lines[i].set_data(self.cities[path, 0], self.cities[path, 1])
                lines[i].set_color(colors[i])
            title.set_text(f'迭代 {frame + 1}/{self.iterations}, 总距离: {self.iteration_best_distances[frame]:.2f}')
            return lines + [title]

        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(self.iteration_best_routes),
                            init_func=init, blit=True, interval=interval, repeat=False)

        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return ani

