import numpy as np
import matplotlib.pyplot as plt

def calculate_circle_area(num_points=10000, radius=1.0, plot=True):
    """
    使用蒙特卡洛方法计算圆的面积
    
    参数:
    - num_points: 随机点的数量
    - radius: 圆的半径
    - plot: 是否可视化结果
    
    返回:
    - 圆面积的估计值
    """
    # 生成均匀分布的随机点 (x, y) ∈ [-radius, radius] × [-radius, radius]
    x = np.random.uniform(-radius, radius, num_points)
    y = np.random.uniform(-radius, radius, num_points)
    
    # 判断点是否在圆内 (x² + y² ≤ radius²)
    inside_circle = (x**2 + y**2) <= radius**2
    
    # 计算圆面积的估计值
    square_area = (2 * radius) ** 2  # 正方形面积
    ratio = np.sum(inside_circle) / num_points  # 圆内点的比例
    circle_area = square_area * ratio  # 圆面积 = 正方形面积 × 比例
    
    # 可视化结果
    if plot:
        plt.figure(figsize=(8, 8))
        # 绘制圆内的点 (绿色)
        plt.scatter(x[inside_circle], y[inside_circle], color='green', s=1, alpha=0.5)
        # 绘制圆外的点 (红色)
        plt.scatter(x[~inside_circle], y[~inside_circle], color='red', s=1, alpha=0.5)
        # 绘制理论圆边界
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(radius * np.cos(theta), radius * np.sin(theta), 'b-', linewidth=2)
        plt.title(f'蒙特卡洛方法计算圆面积 (n={num_points})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    # 理论圆面积
    true_area = np.pi * radius**2
    error = abs(circle_area - true_area) / true_area * 100
    
    print(f"随机点数: {num_points}")
    print(f"估计圆面积: {circle_area:.6f}")
    print(f"实际圆面积: {true_area:.6f}")
    print(f"相对误差: {error:.2f}%")
    
    return circle_area

# 示例：生成10000个随机点计算单位圆面积
if __name__ == "__main__":
    calculate_circle_area(num_points=10000, radius=1.0)    