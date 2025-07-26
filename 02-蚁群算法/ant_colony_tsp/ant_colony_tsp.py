import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 使用黑体，若没有该字体，可根据系统情况替换，如 'Microsoft YaHei'
# 解决负号显示问题
rcParams['axes.unicode_minus'] = False

class AntColonyOptimizer:
    def __init__(self, cities: np.ndarray, num_ants: int, alpha: float, beta: float, rho: float, q: float, 
                 iterations: int):
        """
        初始化蚁群优化器

        :param cities: 城市坐标数组，形状为 (n, 2)，n 为城市数量，每个元素表示一个城市的二维坐标
        :param num_ants: 蚂蚁数量，即每轮迭代中参与路径构建的蚂蚁数量
        :param alpha: 信息素重要程度因子，控制信息素在蚂蚁选择下一个城市时的影响程度
        :param beta: 启发式因子重要程度因子，控制城市间距离在蚂蚁选择下一个城市时的影响程度
        :param rho: 信息素挥发系数，范围在 0 到 1 之间，表示每轮迭代后信息素的挥发比例
        :param q: 信息素增加强度系数，控制蚂蚁在路径上留下的信息素量
        :param iterations: 迭代次数，算法执行的总轮数
        """
        self.num_cities = len(cities)
        self.cities = cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.iterations = iterations
        
        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        np.fill_diagonal(self.pheromone_matrix, 0)  # 对角线元素为0
        
        # 记录最优路径和最优距离
        self.best_path = None
        self.best_distance = float('inf')
        
        # 记录每轮迭代的最优路径和距离
        self.iteration_best_paths = []
        self.iteration_best_distances = []

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        计算城市间的距离矩阵

        :return: 距离矩阵，形状为 (num_cities, num_cities)，matrix[i][j] 表示城市 i 到城市 j 的距离
        """
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                dist = np.sqrt(((self.cities[i] - self.cities[j])**2).sum())
                matrix[i, j] = dist
                matrix[j, i] = dist  # 对称矩阵
        return matrix

    def _calculate_path_distance(self, path: List[int]) -> float:
        """
        计算路径总距离

        :param path: 城市访问顺序列表，每个元素为城市的索引
        :return: 路径的总距离，即依次访问路径中城市再回到起点的距离之和
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i], path[i+1]]
        distance += self.distance_matrix[path[-1], path[0]]  # 返回起点
        return distance

    def _select_next_city(self, current_city: int, unvisited_cities: List[int]) -> int:
        """
        蚂蚁选择下一个城市

        :param current_city: 当前蚂蚁所在的城市索引
        :param unvisited_cities: 未访问的城市索引列表
        :return: 蚂蚁选择的下一个城市的索引
        """
        probabilities = []
        for city in unvisited_cities:
            # 计算转移概率
            pheromone = self.pheromone_matrix[current_city, city] ** self.alpha
            heuristic = (1.0 / self.distance_matrix[current_city, city]) ** self.beta
            probabilities.append(pheromone * heuristic)
        
        # 归一化概率
        if sum(probabilities) == 0:
            return unvisited_cities[0]  # 避免概率和为0的情况
        probabilities = [p / sum(probabilities) for p in probabilities]
        
        # 根据概率选择下一个城市
        return np.random.choice(unvisited_cities, p=probabilities)

    def solve(self) -> Tuple[List[int], float]:
        """
        执行蚁群优化算法

        :return: 一个元组，包含最优路径（城市索引列表）和最优路径的总距离
        """
        self.distance_matrix = self._calculate_distance_matrix()
        for iteration in range(self.iterations):
            all_paths = []
            all_distances = []
            iteration_best_distance = float('inf')
            iteration_best_path = None
            
            # 每只蚂蚁构建路径
            for ant in range(self.num_ants):
                # 随机选择起点
                start_city = np.random.randint(0, self.num_cities)
                path = [start_city]
                unvisited_cities = list(range(self.num_cities))
                unvisited_cities.remove(start_city)
                
                # 构建完整路径
                while unvisited_cities:
                    next_city = self._select_next_city(path[-1], unvisited_cities)
                    path.append(next_city)
                    unvisited_cities.remove(next_city)
                
                # 计算路径距离
                distance = self._calculate_path_distance(path)
                all_paths.append(path)
                all_distances.append(distance)
                
                # 更新迭代最优解
                if distance < iteration_best_distance:
                    iteration_best_distance = distance
                    iteration_best_path = path.copy()
                
                # 更新全局最优解
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # 记录本轮迭代的最优路径和距离
            self.iteration_best_paths.append(iteration_best_path)
            self.iteration_best_distances.append(iteration_best_distance)
            
            # 更新信息素矩阵
            self._update_pheromone_matrix(all_paths, all_distances)
        
        return self.best_path, self.best_distance

    def _update_pheromone_matrix(self, all_paths: List[List[int]], all_distances: List[float]):
        """
        更新信息素矩阵

        :param all_paths: 本轮迭代中所有蚂蚁构建的路径列表，每个元素为一个城市索引列表
        :param all_distances: 本轮迭代中所有蚂蚁构建路径的距离列表，与 all_paths 对应
        """
        # 信息素挥发
        self.pheromone_matrix *= (1 - self.rho)
        
        # 信息素增加
        for path, distance in zip(all_paths, all_distances):
            pheromone_to_add = self.q / distance
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i], path[i+1]] += pheromone_to_add
                self.pheromone_matrix[path[i+1], path[i]] += pheromone_to_add  # 对称更新
            # 从最后一个城市回到起点
            self.pheromone_matrix[path[-1], path[0]] += pheromone_to_add
            self.pheromone_matrix[path[0], path[-1]] += pheromone_to_add

    def plot_result(self, city_names: List[str] = None):
        """
        可视化最优路径

        绘制包含城市点、城市名称标签和最优路径的图表，并显示最优路径的总距离

        :param city_names: 城市名称列表，与城市坐标对应
        """
        if self.best_path is None:
            print("请先运行solve()方法求解TSP")
            return
        
        plt.figure(figsize=(12, 10))
        # 绘制城市点
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=70)
        
        # 添加城市名称标签
        if city_names:
            for i, (x, y) in enumerate(self.cities):
                plt.annotate(city_names[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        # 绘制最优路径
        path = self.best_path + [self.best_path[0]]  # 闭合路径
        plt.plot(self.cities[path, 0], self.cities[path, 1], 'r-', linewidth=2)
        
        # 添加距离信息
        plt.title(f'最优路径距离: {self.best_distance:.2f}')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """
        可视化算法收敛过程

        绘制每轮迭代的最优路径距离曲线，并显示全局最优路径距离的水平线
        """
        if not self.iteration_best_distances:
            print("请先运行solve()方法求解TSP")
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

    def animate_iterations(self, city_names: List[str] = None, interval=200):
        """
        动画展示迭代过程中最优路径的变化

        :param city_names: 城市名称列表，与城市坐标对应
        :param interval: 帧间隔时间（毫秒），控制动画播放速度
        :return: 动画对象
        """
        if not self.iteration_best_paths:
            print("请先运行solve()方法求解TSP")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制城市点（只绘制一次）
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=70)
        
        # 添加城市名称标签
        if city_names:
            for i, (x, y) in enumerate(self.cities):
                ax.annotate(city_names[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        # 初始化路径线和标题
        line, = ax.plot([], [], 'r-', linewidth=2)
        title = ax.set_title('')
        
        def init():
            line.set_data([], [])
            title.set_text('')
            return line, title
        
        def update(frame):
            path = self.iteration_best_paths[frame] + [self.iteration_best_paths[frame][0]]
            line.set_data(self.cities[path, 0], self.cities[path, 1])
            title.set_text(f'迭代 {frame+1}/{self.iterations}, 路径距离: {self.iteration_best_distances[frame]:.2f}')
            return line, title
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(self.iteration_best_paths),
                            init_func=init, blit=True, interval=interval, repeat=False)
        
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return ani
