import pandas as pd

# 1. 输入基础数据
# 小区需求(kg)：A、B、C、D、E
demands = {'A': 20, 'B': 35, 'C': 25, 'D': 40, 'E': 15}
# 距离矩阵(米)：行/列依次为服务中心(0)、A(1)、B(2)、C(3)、D(4)、E(5)
distance_matrix = [
    [0, 800, 1200, 1500, 900, 600],    # 服务中心到各点
    [800, 0, 500, 900, 1100, 400],     # A到各点
    [1200, 500, 0, 600, 800, 700],     # B到各点
    [1500, 900, 600, 0, 500, 1000],    # C到各点
    [900, 1100, 800, 500, 0, 800],     # D到各点
    [600, 400, 700, 1000, 800, 0]      # E到各点
]
# 映射：小区名称→距离矩阵索引（服务中心为0）
name_to_idx = {'服务中心': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
max_load = 100  # 最大载重(kg)


# 2. 计算节约里程矩阵
def calculate_savings():
    savings = []
    nodes = list(demands.keys())  # ['A','B','C','D','E']
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            # 节约里程公式：S(a,b) = d(0,a) + d(0,b) - d(a,b)
            s = distance_matrix[0][name_to_idx[a]] + distance_matrix[0][name_to_idx[b]] - distance_matrix[name_to_idx[a]][name_to_idx[b]]
            savings.append((a, b, s))
    # 按节约里程降序排序
    savings.sort(key=lambda x: x[2], reverse=True)
    return savings


# 3. 构建初始路线（每个小区单独配送）
def init_routes():
    routes = []
    for node in demands.keys():
        # 路线格式：(起点, [途经点], 终点, 总载重, 总距离)
        distance = distance_matrix[0][name_to_idx[node]] * 2  # 往返距离
        routes.append(('服务中心', [node], '服务中心', demands[node], distance))
    return routes


# 4. 合并路线（核心步骤）
def merge_routes(routes, savings):
    # 记录每个小区所属的路线索引
    node_route = {node: i for i, (_, nodes, _, _, _) in enumerate(routes) for node in nodes}
    
    for a, b, s in savings:
        # 跳过已在同一条路线的小区
        if node_route[a] == node_route[b]:
            continue
        
        # 获取a和b所在的路线
        idx_a = node_route[a]
        idx_b = node_route[b]
        start_a, path_a, end_a, load_a, dist_a = routes[idx_a]
        start_b, path_b, end_b, load_b, dist_b = routes[idx_b]
        
        # 计算合并后的载重
        total_load = load_a + load_b
        if total_load > max_load:
            continue  # 超重，跳过合并
        
        # 提取实际端点（途经点的首尾元素）
        actual_start_a, actual_end_a = path_a[0], path_a[-1]
        actual_start_b, actual_end_b = path_b[0], path_b[-1]
        
        new_path = None
        new_start = start_a  # 始终从服务中心出发
        new_end = end_a      # 始终回到服务中心
        
        # 合并规则修正：基于实际端点判断
        # 情况1：a是路线A的末端，b是路线B的起点
        if actual_end_a == a and actual_start_b == b:
            new_path = path_a + path_b
        # 情况2：b是路线B的末端，a是路线A的起点
        elif actual_end_b == b and actual_start_a == a:
            new_path = path_b + path_a
        # 情况3：a是路线A的末端，b是路线B的末端
        elif actual_end_a == a and actual_end_b == b:
            new_path = path_a + path_b[::-1]  # 反转路线B
        # 情况4：a是路线A的起点，b是路线B的起点
        elif actual_start_a == a and actual_start_b == b:
            new_path = path_b + path_a[::-1]  # 反转路线A
        
        if new_path is None:
            continue  # 不符合合并条件
        
        # 计算合并后的距离（原距离和 - 节约里程）
        new_distance = dist_a + dist_b - s
        
        # 更新路线列表（删除旧路线，添加新路线）
        new_routes = []
        for i in range(len(routes)):
            if i != idx_a and i != idx_b:
                new_routes.append(routes[i])
        new_routes.append((new_start, new_path, new_end, total_load, new_distance))
        
        # 重新构建小区-路线映射（完全重建避免遗漏）
        node_route = {}
        for i, route in enumerate(new_routes):
            for node in route[1]:
                node_route[node] = i
        
        routes = new_routes
    
    return routes

# 5. 执行算法并输出结果
def main():
    savings = calculate_savings()
    print("=== 节约里程排序（前5名） ===")
    for i in range(min(5, len(savings))):
        print(f"{savings[i][0]}-{savings[i][1]}: {savings[i][2]}米")
    
    routes = init_routes()
    print("\n=== 初始路线 ===")
    for i, route in enumerate(routes):
        print(f"路线{i+1}：{route[0]}→{'→'.join(route[1])}→{route[2]}，载重{route[3]}kg，距离{route[4]}米")
    
    optimal_routes = merge_routes(routes, savings)
    print("\n=== 最优配送方案 ===")
    total_distance = 0
    for i, route in enumerate(optimal_routes):
        print(f"车辆{i+1}：{route[0]}→{'→'.join(route[1])}→{route[2]}，载重{route[3]}kg，距离{route[4]}米")
        total_distance += route[4]
    print(f"\n总配送距离：{total_distance}米")


if __name__ == "__main__":
    main()