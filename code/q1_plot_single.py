# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置Seaborn的主题
sns.set(style="whitegrid")

# 定义CSV文件路径
csv_file = 'XX/res/32.31.250.103/20240501_20240501125647_20240501140806_125649.csv'

# 检查CSV文件是否存在
if not os.path.exists(csv_file):
    # 如果CSV文件不存在，创建一个示例文件
    sample_data = """Frame,Flow,Density
25,4,19
50,2,20
75,2,18
100,2,20
125,3,20
150,1,18
175,1,12
200,2,14
225,2,12
250,1,11
275,3,14
"""
    with open(csv_file, 'w') as f:
        f.write(sample_data)
    print(f"已创建示例CSV文件: {csv_file}")

# 从CSV文件中读取数据
df = pd.read_csv(csv_file)

# 查看数据基本信息
print("数据预览：")
print(df.head())

# 检查缺失值
print("\n缺失值检查：")
print(df.isnull().sum())

# 描述性统计
print("\n描述性统计：")
print(df.describe())

# 创建一个保存图像的目录
output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 时间序列图
plt.figure(figsize=(10, 6))
plt.plot(df['Frame'], df['Flow'], marker='o', label='Flow', color='blue')
plt.plot(df['Frame'], df['Density'], marker='s', label='Density', color='orange')
plt.title('Traffic flow and density change with frame rate')
plt.xlabel('Frame')
plt.ylabel('value')
plt.legend()
plt.tight_layout()
# 保存图像
time_series_path = os.path.join(output_dir, 'time_series.png')
plt.savefig(time_series_path)
plt.close()
print(f"时间序列图已保存为 {time_series_path}")

# 2. 散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Flow', y='Density', data=df, s=100, color='b')
plt.title('Scatter plot of the relationship between traffic flow and vehicle density')
plt.xlabel('Flow')
plt.ylabel('Density')
plt.tight_layout()
# 保存图像
scatter_path = os.path.join(output_dir, 'scatter_plot.png')
plt.savefig(scatter_path)
plt.close()
print(f"散点图已保存为 {scatter_path}")

# 3. 相关性热图
corr = df[['Flow', 'Density']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heat map of the correlation between traffic flow and vehicle density')
plt.tight_layout()
# 保存图像
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.close()
print(f"相关性热图已保存为 {heatmap_path}")

# 4. 直方图
plt.figure(figsize=(12, 5))

# 车流量直方图
plt.subplot(1, 2, 1)
sns.histplot(df['Flow'], bins=5, kde=True, color='g')
plt.title('Flow')
plt.xlabel('Flow')
plt.ylabel('Frequency')

# 车密度直方图
plt.subplot(1, 2, 2)
sns.histplot(df['Density'], bins=5, kde=True, color='r')
plt.title('Density')
plt.xlabel('Density')
plt.ylabel('frequency')

plt.tight_layout()
# 保存图像
histogram_path = os.path.join(output_dir, 'histograms.png')
plt.savefig(histogram_path)
plt.close()
print(f"直方图已保存为 {histogram_path}")
