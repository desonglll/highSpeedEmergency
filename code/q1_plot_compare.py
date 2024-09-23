# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置Seaborn的主题
sns.set(style="whitegrid")

# 定义CSV文件路径
csv_file1 = 'res/32.31.250.103/20240501_20240501125647_20240501140806_125649.csv'
csv_file2 = 'XX/res/32.31.250.105/20240501_20240501130415_20240501141554_130415.csv'

# 检查并创建示例CSV文件（如果不存在）
def create_sample_csv(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(data)
        print(f"已创建示例CSV文件: {file_path}")

sample_data1 = """Frame,Flow,Density
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

sample_data2 = """Frame,Flow,Density
25,5,18
50,3,19
75,2,17
100,3,19
125,4,21
150,2,17
175,1,13
200,3,15
225,1,13
250,2,12
275,4,15
"""

create_sample_csv(csv_file1, sample_data1)
create_sample_csv(csv_file2, sample_data2)

# 从CSV文件中读取数据
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# 查看数据基本信息
print("数据1预览：")
print(df1.head())

print("\n数据2预览：")
print(df2.head())

# 合并两个数据集，基于'Frame'列
merged_df = pd.merge(df1, df2, on='Frame', suffixes=('1', '2'))

# 查看合并后的数据
print("\n合并后的数据预览：")
print(merged_df.head())

# 检查缺失值
print("\n缺失值检查：")
print(merged_df.isnull().sum())

# 描述性统计
print("\n描述性统计：")
print(merged_df.describe())

# 创建一个保存图像的目录
output_dir = 'plots_comparison'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 时间序列对比图
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Frame'], merged_df['Flow1'], marker='o', label='车流量 Data1', color='blue')
plt.plot(merged_df['Frame'], merged_df['Flow2'], marker='o', label='车流量 Data2', color='red')
plt.plot(merged_df['Frame'], merged_df['Density1'], marker='s', label='车密度 Data1', color='green')
plt.plot(merged_df['Frame'], merged_df['Density2'], marker='s', label='车密度 Data2', color='orange')
plt.title('两个数据集的车流量与车密度随帧数变化趋势')
plt.xlabel('帧数 (Frame)')
plt.ylabel('值')
plt.legend()
plt.tight_layout()
# 保存图像
time_series_cmp_path = os.path.join(output_dir, 'time_series_comparison.png')
plt.savefig(time_series_cmp_path)
plt.close()
print(f"时间序列对比图已保存为 {time_series_cmp_path}")

# 2. 散点图 - Flow1 vs Flow2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Flow1', y='Flow2', data=merged_df, s=100, color='purple')
plt.title('Data1 车流量 vs Data2 车流量 散点图')
plt.xlabel('车流量 Data1 (Flow1)')
plt.ylabel('车流量 Data2 (Flow2)')
plt.tight_layout()
# 保存图像
scatter_flow_path = os.path.join(output_dir, 'scatter_flow_comparison.png')
plt.savefig(scatter_flow_path)
plt.close()
print(f"车流量散点图已保存为 {scatter_flow_path}")

# 3. 散点图 - Density1 vs Density2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Density1', y='Density2', data=merged_df, s=100, color='brown')
plt.title('Data1 车密度 vs Data2 车密度 散点图')
plt.xlabel('车密度 Data1 (Density1)')
plt.ylabel('车密度 Data2 (Density2)')
plt.tight_layout()
# 保存图像
scatter_density_path = os.path.join(output_dir, 'scatter_density_comparison.png')
plt.savefig(scatter_density_path)
plt.close()
print(f"车密度散点图已保存为 {scatter_density_path}")

# 4. 相关性热图
corr = merged_df[['Flow1', 'Flow2', 'Density1', 'Density2']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation heat map between two datasets')
plt.tight_layout()
# 保存图像
heatmap_cmp_path = os.path.join(output_dir, 'correlation_heatmap_comparison.png')
plt.savefig(heatmap_cmp_path)
plt.close()
print(f"相关性热图已保存为 {heatmap_cmp_path}")

# 5. 直方图比较
plt.figure(figsize=(14, 6))

# 车流量直方图比较
plt.subplot(1, 2, 1)
sns.histplot(merged_df['Flow1'], bins=5, kde=True, color='blue', label='Data1 Flow', alpha=0.6)
sns.histplot(merged_df['Flow2'], bins=5, kde=True, color='red', label='Data2 Flow', alpha=0.6)
plt.title('Comparison of distributions (Flow) ')
plt.xlabel('(Flow)')
plt.ylabel('F')
plt.legend()

# 车密度直方图比较
plt.subplot(1, 2, 2)
sns.histplot(merged_df['Density1'], bins=5, kde=True, color='green', label='Data1 Density', alpha=0.6)
sns.histplot(merged_df['Density2'], bins=5, kde=True, color='orange', label='Data2 Density', alpha=0.6)
plt.title('Comparison of distributions (Density) ')
plt.xlabel(' (Density)')
plt.ylabel('F')
plt.legend()

plt.tight_layout()
# 保存图像
histogram_cmp_path = os.path.join(output_dir, 'histograms_comparison.png')
plt.savefig(histogram_cmp_path)
plt.close()
print(f"直方图比较已保存为 {histogram_cmp_path}")

# 6. 相关系数计算与输出
print("\n相关系数矩阵：")
print(corr)

# 7. 进一步的散点图和回归线（可选）
# Flow1 vs Flow2 with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='Flow1', y='Flow2', data=merged_df, scatter_kws={'s':100}, line_kws={'color':'red'})
plt.title('Data1 Traffic Flow vs Data2 Traffic Flow Regression scatter plot')
plt.xlabel('Data1 (Flow1)')
plt.ylabel(' Data2 (Flow2)')
plt.tight_layout()
# 保存图像
scatter_flow_reg_path = os.path.join(output_dir, 'scatter_flow_regression.png')
plt.savefig(scatter_flow_reg_path)
plt.close()
print(f"车流量回归散点图已保存为 {scatter_flow_reg_path}")

# Density1 vs Density2 with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='Density1', y='Density2', data=merged_df, scatter_kws={'s':100}, line_kws={'color':'red'})
plt.title('Data1 Traffic Density vs Data2 Traffic Density Regression scatter plot')
plt.xlabel('Data1 (Density1)')
plt.ylabel('Data2 (Density2)')
plt.tight_layout()
# 保存图像
scatter_density_reg_path = os.path.join(output_dir, 'scatter_density_regression.png')
plt.savefig(scatter_density_reg_path)
plt.close()
print(f"车密度回归散点图已保存为 {scatter_density_reg_path}")

print("\n所有图像已成功保存到 'plots_comparison' 目录中。")
