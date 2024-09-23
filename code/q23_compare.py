import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file1 = '/home/ns/homework/traffic_estimated_speed.csv'
file2 = '/home/ns/homework/processed_file.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 创建图形和轴
plt.figure(figsize=(10, 6))

# 绘制第一个文件的数据
plt.plot(df1['Frame'], df1['Density'],  label='Before')

# 绘制第二个文件的数据
plt.plot(df2['Frame'], df2['Density'],  label='After')

# 添加标题和标签
plt.title('Density-Frame')
plt.xlabel('Frame')
plt.ylabel('Density')

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend()

# 显示图形

plt.savefig("compare_plot.png")
