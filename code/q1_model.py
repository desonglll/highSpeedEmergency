import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设定参数
K_jam = 180  # 堵塞密度（车辆/公里）
V_c = 35     # 临界速度（公里/小时）

# 读取CSV文件
df = pd.read_csv('/home/ns/homework/res/32.31.250.108/20240501_20240501113543_20240501135236_113542.csv')

# 定义Greenberg模型函数
def estimate_speed_greenberg(K, V_c=V_c, K_jam=K_jam):
    """
    根据Greenberg模型估算速度
    """
    K = np.maximum(K, 1e-5)  # 防止密度为零，避免log(0)
    V = V_c * np.log(K_jam / K)
    return np.maximum(V, 0)  # 确保速度不为负

# 计算速度
df['Estimated_Speed'] = estimate_speed_greenberg(df['Density'])

# 将结果保存到新的CSV文件
df.to_csv('traffic_estimated_speed.csv', index=False)
print('结果已保存到文件：traffic_estimated_speed.csv')

# 可视化Estimated_Speed并保存图片
# 创建图形和坐标轴
plt.figure(figsize=(8, 6))

# 绘制估计速度曲线
plt.plot(df['Frame'], df['Estimated_Speed'], marker='o', linestyle='-', color='blue', label='Estimate the velocity')

# 添加水平线 y = 70
plt.axhline(y=70, color='red', linestyle='--', linewidth=1, label='Threshold 70 km/h')

# 添加水平线 y = 80
plt.axhline(y=80, color='green', linestyle='--', linewidth=1, label='Threshold 80 km/h')

# 设置坐标轴标签
plt.xlabel('Frame')
plt.ylabel('Estimated speed (km/h)')

# 设置图形标题
plt.title('Estimated Speed over Frames')

# 显示图例
plt.legend()

# 添加网格
plt.grid(True)

# 保存图形为PNG文件
plt.savefig('estimated_speed_plot.png')

# 打印保存确认信息
print('Estimated_Speed的可视化已保存为：estimated_speed_plot.png')

# 关闭图形
plt.close()
