import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import font_manager
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def show(df,state):
    time_data = df['seconds']
    # 创建图表
    plt.figure(figsize=(10, 6))  # 设置图表大小

    # 绘制平滑后的车速曲线
    plt.plot(time_data, df[state], label='', color='red')

    # 设置图表标题
    plt.title("车速度-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

    # 设置横轴标签
    plt.xlabel("时间（十秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

    # 设置纵轴标签
    plt.ylabel("车速度（千米每小时）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

    # 显示图表
    plt.legend()
    plt.show()



# 1. 数据预处理
data_path = 'data/csv/car_flow_density_speed.csv'
df = pd.read_csv(data_path)

# span = 20
# df['107_flow'] = df['107_flow'].ewm(span=span, adjust=False).mean()
# df['107_density'] = df['107_density'].ewm(span=span, adjust=False).mean()
# df['107_speed'] = df['107_speed'].ewm(span=span, adjust=False).mean()
# df['108_flow'] = df['108_flow'].ewm(span=span, adjust=False).mean()
# df['108_density'] = df['108_density'].ewm(span=span, adjust=False).mean()
# df['108_speed'] = df['108_speed'].ewm(span=span, adjust=False).mean()
# df['105_flow'] = df['105_flow'].ewm(span=span, adjust=False).mean()
# df['105_density'] = df['105_density'].ewm(span=span, adjust=False).mean()
# df['105_speed'] = df['105_speed'].ewm(span=span, adjust=False).mean()

# df.to_csv(data_path,index=False)

# show(df,'107_flow')
# 选择要使用的特征：第107和105点的数据预测第108点的数据
input_features = df[['107_flow', '107_density', '107_speed', '105_flow', '105_density', '105_speed']].values
output_features = df[['108_flow', '108_density', '108_speed']].values

# 归一化数据
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

scaled_input = scaler_input.fit_transform(input_features).astype(np.float32)  # 转换为 float32 类型
scaled_output = scaler_output.fit_transform(output_features).astype(np.float32)

# 划分数据集，90%训练集，10%测试集
train_size = int(len(scaled_input)*0.90)
train_input = scaled_input[:train_size]
train_output = scaled_output[:train_size]
test_input = scaled_input[train_size:]
test_output = scaled_output[train_size:]
print(train_size)


# 定义用于生成时间序列数据的 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, input_data, output_data, sequence_length):
        self.input_data = input_data
        self.output_data = output_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.input_data) - self.sequence_length

    def __getitem__(self, index):
        return (self.input_data[index:index + self.sequence_length],
                self.output_data[index + self.sequence_length])


sequence_length = 5  # 假设我们使用前5个时间点的数据来预测下一个时间点
train_dataset = TimeSeriesDataset(train_input, train_output, sequence_length)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 2. 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        # src shape: [batch_size, seq_length, input_dim]
        src = self.embedding(src) + self.pos_encoder
        src = self.transformer_encoder(src)
        output = self.decoder(src[:, -1, :])  # 只取最后一个时间点的输出进行预测
        return output


# 模型参数
input_dim = 6  # 输入是6个特征：第107和105点的流量、密度、速度
embed_dim = 64
num_heads = 8
num_layers = 3
output_dim = 3  # 输出是3个特征：第108点的流量、密度、速度
seq_length = sequence_length

# 实例化模型
model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, output_dim, seq_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 训练模型
# epochs = 50
# losses = []
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in train_dataloader:
#         # 确保 inputs 和 targets 是 PyTorch 张量，并转换为 float32
#         inputs, targets = torch.tensor(inputs).float(), torch.tensor(targets).float()
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     avg_loss = running_loss / len(train_dataloader)
#     losses.append(avg_loss)
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

# 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(losses, label='Loss', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.grid()
# plt.show()

# 保存模型
# torch.save(model.state_dict(), 'transformer_model.pth')

# 加载模型
model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, output_dim, seq_length)
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()  # 设置模型为评估模式

# 4. 预测未来的数据
def predict_future(model, input_data, future_steps):
    model.eval()
    inputs = torch.FloatTensor(input_data[-sequence_length:]).unsqueeze(0)  # 加入 batch 维度
    predictions = []

    for i in range(future_steps):
        with torch.no_grad():
            output = model(inputs)  # 预测第三个监控点（第108点）的流量、密度、速度
            predictions.append(output.numpy().squeeze(0))  # 存储预测结果

            # 更新输入，只使用当前时间点的第一个和第二个监控点的数据作为下一个时间点的输入
            if i + sequence_length < len(input_data):
                next_input = torch.FloatTensor(input_data[i + 1:i + 1 + sequence_length, :6]).unsqueeze(
                    0)  # 只取前两个监控点（6个特征）
                inputs = next_input
            else:
                break  # 如果数据不足以形成新的输入，结束预测

    return np.array(predictions)


# 预测测试集的数据
future_steps = len(test_input)
predicted_future = predict_future(model, scaled_input[train_size - sequence_length:], future_steps)

# 将预测的结果反归一化
predicted_future_rescaled = scaler_output.inverse_transform(predicted_future)

# 输出预测结果
print(predicted_future_rescaled)

# predicted_future_rescaled.to_csv('data/csv/new_data.csv')

# 绘制流量预测图
time_points = np.arange(len(input_features))

plt.figure(figsize=(12, 8))

# 读取新的数据文件
new_data_path = 'data/csv/new_data.csv'
new_df = pd.read_csv(new_data_path)

# 提取新的流量、密度和速度数据
flow_new = new_df['flow_new'].values
density_new = new_df['density_new'].values
speed_new = new_df['speed_new'].values

# 绘制历史流量数据
plt.subplot(3, 1, 1)  # 3行1列的第1个子图
plt.plot(time_points[:train_size], scaler_output.inverse_transform(train_output)[:, 0], label='flow_Train', color='blue')
plt.plot(time_points[train_size:], scaler_output.inverse_transform(test_output)[:, 0], label='flow_Test', color='lightblue')
plt.plot(time_points[train_size:], flow_new[1:], label='flow_predict', color='orange')
plt.xlabel('时间（十秒）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.ylabel('流量（辆每十秒）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.title('车流量预测效果展示（未来10分钟）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.axvline(x=train_size, color='red', linestyle='--', label='Prediction Start')
plt.legend()
plt.grid()

# 绘制速度预测图
plt.subplot(3, 1, 2)  # 3行1列的第2个子图
plt.plot(time_points[:train_size], scaler_output.inverse_transform(train_output)[:, 2], label='speed_train', color='green')
plt.plot(time_points[train_size:], scaler_output.inverse_transform(test_output)[:, 2], label='speed_test', color='lightgreen')
plt.plot(time_points[train_size:], speed_new[1:], label='speed_predict', color='orange')
plt.xlabel('时间（十秒）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.ylabel('速度（千米每小时）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.title('车速度预测效果展示（未来10分钟）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.axvline(x=train_size, color='red', linestyle='--', label='Prediction Start')
plt.legend()
plt.grid()

# 绘制密度预测图
plt.subplot(3, 1, 3)  # 3行1列的第3个子图
plt.plot(time_points[:train_size], scaler_output.inverse_transform(train_output)[:, 1], label='density_train', color='purple')
plt.plot(time_points[train_size:], scaler_output.inverse_transform(test_output)[:, 1], label='density_test', color='purple')
plt.plot(time_points[train_size:], density_new[1:], label='density_predict', color='orange')
plt.xlabel('时间（十秒）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.ylabel('密度（辆每米）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.title('车密度预测效果展示（未来10分钟）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
plt.axvline(x=train_size, color='red', linestyle='--', label='Prediction Start')
plt.legend()
plt.grid()

plt.tight_layout()  # 自动调整子图间距
plt.show()