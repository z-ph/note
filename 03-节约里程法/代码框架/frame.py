import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SavingsAlgorithm:
    def __init__(self, depot: Tuple[float, float], customers: Dict[int, Tuple[float, float]], 
                 demands: Dict[int, float], vehicle_capacity: float):
        """
        初始化节约里程法求解器
        
        参数:
            depot: 配送中心坐标 (x, y)
            customers: 客户点字典 {客户点ID: (x坐标, y坐标)}
            demands: 客户点需求字典 {客户点ID: 需求量}
            vehicle_capacity: 车辆容量限制
        """
        self.depot = depot
        self.customers = customers
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distances = self._calculate_distances()
        self.savings = self._calculate_savings()
        self.speed = 25  # 假设车辆速度为25km/h
        self.stop_time = 10  # 停留时间为10分钟
    def calculate_distance(self, dx: float, dy: float) -> float:
        """计算两点之间的欧几里得距离"""
        return np.sqrt(dx ** 2 + dy ** 2)
    def _calculate_distances(self) -> Dict[Tuple[int, int], float]:
        """计算所有点对之间的欧几里得距离"""
        distances = {}
        all_points = {0: self.depot}  # 0表示配送中心
        all_points.update(self.customers)
        
        for i in all_points:
            for j in all_points:
                # 包含 i == j 的情况，距离设为 0
                dx = all_points[i][0] - all_points[j][0]
                dy = all_points[i][1] - all_points[j][1]
                distances[(i, j)] = self.calculate_distance(dx, dy)
        return distances
    def _calculate_savings(self) -> List[Tuple[int, int, float]]:
        """计算所有客户点对之间的节约里程，并按降序排序"""
        savings_list = []
        for i in self.customers:
            for j in self.customers:
                if i < j:  # 避免重复计算
                    saving = self.distances[(i, 0)] + self.distances[(0, j)] - self.distances[(i, j)]
                    savings_list.append((i, j, saving))
        
        # 按节约里程降序排序
        savings_list.sort(key=lambda x: x[2], reverse=True)
        return savings_list
    
    def solve(self, max_distance: Optional[float] = None, max_time: int = 6) -> List[List[int]]:
        """
        求解车辆路径问题
        
        参数:
            max_distance: 可选，单车最大行驶距离限制
            max_time: 最大工作时间（小时）
            
        返回:
            路径列表，每个子列表表示一辆车的路径（包含配送中心0）
        """
        # 初始化每个点一条路径
        routes = [[0, i, 0] for i in self.customers.keys()]

        # 路径合并
        for i, j, s in self.savings:
            # 找到i和j分别所在的路径
            route_i = None
            route_j = None
            for r in routes:
                if r[1] == i or r[-2] == i:
                    route_i = r
                if r[1] == j or r[-2] == j:
                    route_j = r
            if route_i is None or route_j is None or route_i is route_j:
                continue

            # 调用 _merge_routes 方法进行路径合并
            new_route = self._merge_routes(route_i, route_j, i, j, max_distance, max_time)
            if new_route:
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)

        return routes
    
    def _merge_routes(self, route1: List[int], route2: List[int], node1: int, node2: int, max_distance: Optional[float], max_time: int) -> Optional[List[int]]:
        """
        合并两条路径，并检查合并后的路径是否满足约束条件
        
        参数:
            route1, route2: 待合并的路径
            node1, node2: 分别在route1和route2中的合并点
            max_distance: 可选，单车最大行驶距离限制
            max_time: 最大工作时间（小时）
            
        返回:
            合并后的路径，或None（如果合并不可行）
        """
        # 只允许首尾合并，且不能成环
        new_route = None
        if route1[-2] == node1 and route2[1] == node2:
            new_route = route1[:-1] + route2[1:]
        elif route1[1] == node1 and route2[-2] == node2:
            new_route = route2[:-1] + route1[1:]

        if new_route is None:
            return None

        # 检查合并后重量
        total_weight = sum(self.demands.get(node, 0) for node in new_route if node != 0)
        if total_weight > self.vehicle_capacity:
            return None

        # 检查合并后时间
        total_distance = self._calculate_route_distance(new_route)
        total_time = (total_distance / self.speed) * 60 + (len(new_route) - 2) * self.stop_time
        if max_time is not None and total_time > max_time * 60:
            return None

        # 检查合并后路径长度是否超过限制
        if max_distance is not None and total_distance > max_distance:
            return None

        return new_route

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route) - 1):
            pair = (route[i], route[i + 1])
            if pair in self.distances:
                total_distance += self.distances[pair]
            else:
                print(f"警告：未找到距离 {pair}")
        return total_distance
    
    def visualize_routes(self, routes: List[List[int]], title: str = "节约里程法路径规划结果"):
        """可视化路径规划结果"""
        plt.figure(figsize=(10, 8))
        
        # 绘制配送中心
        plt.scatter(self.depot[0], self.depot[1], c='red', s=200, marker='s', label='配送中心')
        
        # 绘制客户点
        for customer_id, (x, y) in self.customers.items():
            plt.scatter(x, y, c='blue', s=100, marker='o')
            plt.annotate(str(customer_id), (x+0.1, y+0.1))
        
        # 绘制路径
        colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            x_coords = [self.depot[0] if node == 0 else self.customers[node][0] for node in route]
            y_coords = [self.depot[1] if node == 0 else self.customers[node][1] for node in route]
            plt.plot(x_coords, y_coords, 'o-', color=color, linewidth=1.5, 
                     label=f'车辆 {i+1}: 距离 = {self._calculate_route_distance(route):.2f}')
        
        plt.title(title)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def print_solution(self, routes: List[List[int]]):
        """打印解决方案摘要"""
        total_distance = sum(self._calculate_route_distance(route) for route in routes)
        total_vehicles = len(routes)
        
        print("\n===== 节约里程法路径规划结果 =====")
        print(f"总车辆数: {total_vehicles}")
        print(f"总行驶距离: {total_distance:.2f}")
        print("\n各车辆路径详情:")
        for i, route in enumerate(routes):
            route_distance = self._calculate_route_distance(route)
            route_demand = sum(self.demands.get(node, 0) for node in route if node != 0)
            print(f"\n车辆 {i+1}:")
            print(f"  路径: {' -> '.join(map(str, route))}")
            print(f"  行驶距离: {route_distance:.2f}")
            print(f"  装载量: {route_demand:.2f} / {self.vehicle_capacity}")

# 使用示例
if __name__ == "__main__":
    data = {
        '送货点': list(range(1, 31)),
        '快件量T(kg)': [8, 8.2, 6, 5.5, 4.5, 3, 7.2, 2.3, 1.4, 6.5, 4.1, 12.7, 5.8, 3.8, 3.4,
                        3.5, 5.8, 7.5, 7.8, 4.6, 6.2, 6.8, 2.4, 7.6, 9.6, 10, 12, 6, 8.1, 4.2],
        '坐标x(km)': [3, 1, 5, 4, 3, 0, 7, 9, 10, 14, 17, 14, 12, 10, 19, 2, 6, 11, 15, 7,
                      22, 21, 27, 15, 15, 20, 21, 24, 25, 28],
        '坐标y(km)': [2, 5, 4, 7, 11, 8, 9, 6, 2, 0, 3, 6, 9, 12, 9, 16, 18, 17, 12, 14,
                      5, 0, 9, 19, 14, 17, 13, 20, 16, 18]
    }

    # 示例1: 简单测试案例
    depot = (0, 0)
    customers = {i: (data['坐标x(km)'][i-1], data['坐标y(km)'][i-1]) for i in data['送货点']}
    demands = {i: data['快件量T(kg)'][i-1] for i in data['送货点']}
    vehicle_capacity = 25
    
    # 创建节约里程法求解器并求解
    solver = SavingsAlgorithm(depot, customers, demands, vehicle_capacity)
    routes = solver.solve(max_time=6)
    
    # 打印结果并可视化
    solver.print_solution(routes)
    solver.visualize_routes(routes)    