import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations

# 假设您的四个文件名为 data1.csv, data2.csv, data3.csv, data4.csv
file_list = ['XX/res/32.31.250.108/20240501_20240501113543_20240501135236_113542.csv', 
             'XX/res/32.31.250.107/20240501_20240501114103_20240501135755_114103.csv', 
             'XX/res/32.31.250.105/20240501_20240501115227_20240501130415_115227.csv', 
             'XX/res/32.31.250.103/20240501_20240501125647_20240501140806_125649.csv']


# 创建一个空的DataFrame来存储所有文件的Density数据
density_df = pd.DataFrame()

# 读取所有文件的Density列并合并到density_df中
for i, file in enumerate(file_list):
    df = pd.read_csv(file)
    density_df[f'Density_File_{i+1}'] = df['Density']

# 计算Density的相关系数矩阵
correlation_matrix = density_df.corr()

print("不同文件的Density相关系数矩阵：")
print(correlation_matrix)

# 可视化相关系数矩阵
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Density Across Files')
plt.savefig('density_correlation_matrix.png')
plt.close()

# 绘制不同文件Density之间的散点图并保存图片
file_indices = range(len(file_list))
file_pairs = list(combinations(file_indices, 2))

for (i, j) in file_pairs:
    plt.figure(figsize=(8,6))
    plt.scatter(density_df[f'Density_File_{i+1}'], density_df[f'Density_File_{j+1}'], color='purple')
    plt.xlabel(f'Density_File_{i+1}')
    plt.ylabel(f'Density_File_{j+1}')
    plt.title(f'Density Comparison between File {i+1} and File {j+1}')
    # 计算这两个文件Density的相关系数
    corr_ij = density_df[f'Density_File_{i+1}'].corr(density_df[f'Density_File_{j+1}'])
    plt.text(0.05, 0.95, f'Correlation: {corr_ij:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.grid(True)
    plt.savefig(f'density_scatter_file_{i+1}_vs_file_{j+1}.png')
    plt.close()
