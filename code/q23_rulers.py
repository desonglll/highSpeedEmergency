import pandas as pd

# 读取CSV文件
df = pd.read_csv('XX/traffic_estimated_speed.csv')

# 定义阈值
density_threshold = 12  # 你可以根据需要修改阈值
speed_threshold = 100

# 处理Density阈值大于指定阈值的连续三个时间帧
for i in range(len(df) - 2):
    if (df.loc[i, 'Density'] > density_threshold and 
        df.loc[i+1, 'Density'] > density_threshold and 
        df.loc[i+2, 'Density'] > density_threshold):
        for j in range(i+3, min(i+8, len(df))):  # 修改接下来的5个时间帧
            df.loc[j, 'Density'] *= 0.8

# 处理Estimated_Speed小于指定阈值的连续三个时间帧
for i in range(len(df) - 2):
    if (df.loc[i, 'Estimated_Speed'] < speed_threshold and 
        df.loc[i+1, 'Estimated_Speed'] < speed_threshold and 
        df.loc[i+2, 'Estimated_Speed'] < speed_threshold):
        for j in range(i+3, min(i+8, len(df))):  # 修改接下来的5个时间帧
            df.loc[j, 'Density'] *= 0.8

# 保存处理后的数据到新的CSV文件
df = df.drop(columns=['Estimated_Speed'])
df.to_csv('processed_file.csv', index=False)

print("处理完成，结果已保存为 'processed_file.csv'")
