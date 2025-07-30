import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import datetime as dt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------
# 1. 数据加载与清洗
# -------------------
# 模拟生成示例数据
def generate_sample_data(n_customers=1000):
    # 随机结果可复现
    np.random.seed(42)
    customer_ids = range(1, n_customers + 1)
    
    data = []
    for customer_id in customer_ids:
        # 每个客户的订单数
        n_orders = np.random.poisson(lam=3) + 1
        
        # 最近一次购买时间 (30天内到365天前)
        last_purchase_days = np.random.randint(1, 365)
        last_purchase_date = dt.datetime.now() - dt.timedelta(days=last_purchase_days)
        
        for _ in range(n_orders):
            # 订单日期 (最后一次购买日期之前的随机时间)
            days_ago = np.random.randint(0, last_purchase_days)
            order_date = last_purchase_date - dt.timedelta(days=days_ago)
            
            # 订单金额 (正态分布，均值500，标准差200)
            amount = max(10, np.random.normal(500, 200))
            
            data.append({
                'CustomerID': customer_id,
                'OrderDate': order_date,
                'OrderID': f"ORD{np.random.randint(1000, 9999)}{customer_id}",
                'Amount': round(amount, 2)
            })
    
    return pd.DataFrame(data)

# 生成示例数据
df = generate_sample_data()

# 检查数据
print(f"数据基本信息：")
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows >= 1000:
    print(f"数据全部内容共有{rows}行，数据全部内容展示：")
    print(df.to_string())
else:
    print(f"数据全部内容共有{rows}行，数据前几行内容展示：")
    print(df.head().to_string())

# -------------------
# 2. 特征工程 - 构建RFM指标
# -------------------
# 计算RFM值
snapshot_date = df['OrderDate'].max() + dt.timedelta(days=1)  # 截止日期设为最后订单日期的后一天

rfm = df.groupby('CustomerID').agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days,  # 最近一次购买距今天数
    'OrderID': lambda x: len(x),  # 购买频率
    'Amount': lambda x: x.sum()  # 总消费金额
})

# 重命名列名
rfm.rename(columns={
    'OrderDate': 'Recency',  # 最近一次购买
    'OrderID': 'Frequency',  # 购买频率
    'Amount': 'Monetary'  # 消费金额
}, inplace=True)

print("\nRFM数据基本信息：")
rfm.info()

# 查看RFM数据集行数和列数
rows, columns = rfm.shape

if rows >= 1000:
    print(f"RFM数据全部内容共有{rows}行，RFM数据全部内容展示：")
    print(rfm.to_string())
else:
    print(f"RFM数据全部内容共有{rows}行，RFM数据前几行内容展示：")
    print(rfm.head().to_string())

# -------------------
# 3. 数据预处理
# -------------------
# 检查异常值
print("\n特征描述性统计：")
print(rfm.describe())

# 处理异常值（可选）
def remove_outliers(df, column, multiplier=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 对每个特征处理异常值
for col in rfm.columns:
    rfm = remove_outliers(rfm, col)

# 对数转换（处理数据偏态）
rfm_log = np.log(rfm + 1)  # +1避免对数处理0值

# 标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# -------------------
# 4. 确定最优聚类数k
# -------------------
# 手肘法
wss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wss.append(kmeans.inertia_)  # 簇内平方和
    if k > 1:  # 轮廓系数要求至少2个簇
        labels = kmeans.labels_
        score = silhouette_score(rfm_scaled, labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# 绘制手肘图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wss, 'bx-')
plt.xlabel('簇数量 (k)')
plt.ylabel('簇内平方和 (WSS)')
plt.title('手肘法确定最优k值')
plt.grid(True)

# 绘制轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'bx-')
plt.xlabel('簇数量 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数确定最优k值')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimal_k.png')
plt.close()

# 确定最优k（手肘法和轮廓系数综合考虑）
optimal_k = 4  # 根据图形手动选择，实际中可自动化实现

# -------------------
# 5. 应用K-Means聚类
# -------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# -------------------
# 6. 聚类结果分析
# -------------------
# 计算每个簇的RFM均值
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'})

print("\n聚类结果分析：")
print(cluster_analysis)

# 绘制雷达图比较各簇特征
plt.figure(figsize=(10, 8))

# 标准化各簇特征用于雷达图
stats = cluster_analysis[['Recency', 'Frequency', 'Monetary']].copy()
stats_max = stats.max()
stats = stats / stats_max

# 生成雷达图
angles = np.linspace(0, 2*np.pi, len(stats.columns), endpoint=False).tolist()
stats = pd.concat([stats, stats[stats.columns[0]]], axis=1)  # 闭合雷达图
angles = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 为每个簇绘制雷达图
colors = ['blue', 'red', 'green', 'purple']
for i, cluster in enumerate(stats.index):
    values = stats.loc[cluster].tolist()
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=f'簇 {cluster}')
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# 设置雷达图属性
ax.set_thetagrids(np.degrees(angles[:-1]), stats.columns[:-1])
ax.set_ylim(0, 1.2)
plt.title('各簇RFM特征比较')
plt.legend(loc='upper right')
plt.savefig('cluster_radar_chart.png')
plt.close()

# 绘制聚类散点图
plt.figure(figsize=(15, 5))

# 散点图1: Recency vs Frequency
plt.subplot(1, 3, 1)
scatter = plt.scatter(rfm['Recency'], rfm['Frequency'], c=rfm['Cluster'], 
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='簇')
plt.xlabel('最近购买时间 (天)')
plt.ylabel('购买频率')
plt.title('客户聚类分布（R vs F）')

# 散点图2: Frequency vs Monetary
plt.subplot(1, 3, 2)
scatter = plt.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster'], 
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='簇')
plt.xlabel('购买频率')
plt.ylabel('总消费金额')
plt.title('客户聚类分布（F vs M）')

# 散点图3: Recency vs Monetary
plt.subplot(1, 3, 3)
scatter = plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], 
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='簇')
plt.xlabel('最近购买时间 (天)')
plt.ylabel('总消费金额')
plt.title('客户聚类分布（R vs M）')

plt.tight_layout()
plt.savefig('cluster_scatter_plots.png')
plt.close()

# -------------------
# 7. 客户群体命名与营销策略建议
# -------------------
# 基于聚类中心为每个簇命名
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                              columns=['Recency', 'Frequency', 'Monetary'])
cluster_centers = np.expm1(cluster_centers)  # 逆对数转换
print("\n聚类中心（原始值）：")
print(cluster_centers)

# 根据聚类中心定义客户群体名称
segment_names = {
    0: '高价值客户',
    1: '潜力客户',
    2: '普通客户',
    3: '流失客户'
}

# 映射客户群体名称
rfm['Segment'] = rfm['Cluster'].map(segment_names)

# 计算各群体占比
segment_distribution = rfm['Segment'].value_counts(normalize=True) * 100
print("\n客户群体分布（百分比）：")
print(segment_distribution.round(1))

# 绘制客户群体分布图
plt.figure(figsize=(8, 6))
sns.barplot(x=segment_distribution.index, y=segment_distribution.values)
plt.xlabel('客户群体')
plt.ylabel('占比 (%)')
plt.title('客户群体分布')
plt.xticks(rotation=45)

# 在柱状图上添加数值标签
for i, v in enumerate(segment_distribution.values):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('customer_segments.png')
plt.close()

# -------------------
# 8. 营销策略建议
# -------------------
print("\n客户细分营销策略建议：")
print("1. 高价值客户：")
print("   - 提供VIP专属服务和个性化推荐")
print("   - 设计高端会员计划，给予独家优惠")
print("   - 定期发送定制化内容，增强品牌忠诚度")

print("\n2. 潜力客户：")
print("   - 发送满减券和限时优惠，刺激消费")
print("   - 推荐相关产品，提高购买频率")
print("   - 通过邮件营销增加品牌曝光")

print("\n3. 普通客户：")
print("   - 提供入门级会员服务，培养消费习惯")
print("   - 发送产品组合优惠，提高客单价")
print("   - 定期推送热门商品和促销活动")

print("\n4. 流失客户：")
print("   - 发送召回优惠券，设置较低使用门槛")
print("   - 调研流失原因，改进产品或服务")
print("   - 推送限时重新激活福利，如首次复购额外折扣")

# 保存结果
rfm.to_csv('customer_segmentation_results.csv', index=True)    