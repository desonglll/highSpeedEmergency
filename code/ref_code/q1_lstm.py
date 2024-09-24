import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print(os.getcwd())
# 设置设备为GPU（如果可用），否则为CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备：{device}')

# 读取CSV数据
df = pd.read_csv('../res/32.31.250.103/20240501_20240501125647_20240501140806_125649.csv')

# 设置'Frame'为索引
df.set_index('Frame', inplace=True)

# 选择要预测的特征，这里以'密度'为例
series = df['Density'].values.astype(float)


# 准备数据集
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 10  # 序列长度设置为10
X, y = create_sequences(series, seq_length)

# 检查是否有足够的数据进行训练
if len(X) == 0:
    print("数据不足以创建指定长度的序列，请减少序列长度或增加数据量。")
else:
    # 转换为张量并移动到指定设备
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)


    # 定义LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=16, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.linear(out[:, -1, :])
            return out


    model = LSTMModel().to(device)  # 将模型移动到指定设备

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 训练模型
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'第{epoch + 1}次迭代，损失值: {loss.item():.4f}')

    # 进行预测
    model.eval()
    with torch.no_grad():
        # 使用训练集进行预测，并将结果移动到CPU
        predicted = model(X).squeeze().cpu().numpy()

        # 使用最近的seq_length个数据预测下一个数据点
        recent_sequence = torch.tensor(series[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        next_prediction = model(recent_sequence).cpu().item()
        print(f'下一个数据点的预测值：{next_prediction:.2f}')

    # 可视化结果并保存图片
    plt.figure()
    plt.plot(range(len(series)), series, label='raw data')
    plt.plot(range(seq_length, len(series)), predicted, label='The training set predicts the data')
    # 将下一个预测值添加到图中
    plt.xlabel('Timeframe')
    plt.ylabel('density')
    plt.legend()
    plt.savefig('prediction_plot.png')  # 保存图片
    plt.close()
