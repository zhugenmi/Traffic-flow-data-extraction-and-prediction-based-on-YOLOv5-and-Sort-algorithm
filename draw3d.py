import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import openpyxl

# 读取数据
file_path = 'data/csv/car_flow_density_speed.xlsx'
df = pd.read_excel(file_path)

span=20
# 提取数据
time_data = df['seconds']
flow_data_103 = df['103_speed'].ewm(span=span, adjust=False).mean()
flow_data_105 = df['105_speed'].ewm(span=span, adjust=False).mean()
flow_data_107 = df['107_speed'].ewm(span=span, adjust=False).mean()
flow_data_108 = df['108_speed'].ewm(span=span, adjust=False).mean()

# 创建三维图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制每个点位的流量数据
ax.plot(time_data, np.full_like(time_data, 103), flow_data_103, label='103', color='r')
ax.plot(time_data, np.full_like(time_data, 105), flow_data_105, label='105', color='g')
ax.plot(time_data, np.full_like(time_data, 107), flow_data_107, label='107', color='b')
ax.plot(time_data, np.full_like(time_data, 108), flow_data_108, label='108', color='y')

# 设置标签
ax.set_xlabel('时间（十秒）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
ax.set_ylabel('点位', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
ax.set_zlabel('速度（千米每小时）', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))
ax.set_title('速度-时间-点位三维图', fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置y轴刻度和标签
ax.set_yticks([103, 105, 107, 108])  # 仅显示这四个点位
ax.set_yticklabels(['103', '105', '107', '108'])  # 设置对应的标签

# 添加图例
ax.legend()

# 调整视角
ax.view_init(elev=20, azim=30)

# 显示图表
plt.show()