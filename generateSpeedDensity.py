from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import font_manager

which_file=8

# 读取 CSV 文件
df_frame = pd.read_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_sec_rect.csv')
df_car_flow_per_sec_line = pd.read_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_sec_line.csv')

df = df_frame

# 添加秒数列
df['seconds'] = df_frame.index

# 车流量
df['car_flow'] = df_car_flow_per_sec_line['car_flow']

df['car_density'] = df_frame['car_density']

df['car_speed'] = df_car_flow_per_sec_line['car_flow'] / df['car_density']

print('avg_speed: '+str(df['car_speed'].mean()))
# print(df['car_speed'])

# df['car_speed'] *= 3.6  # 换算成千米每小时

# # 限制速度
# for i in range(len(df)):
#     if df['car_speed'][i] > 150:
#         df['car_speed'][i] = 150

# 重新排序列
df = df[['seconds', 'small_cars', 'large_cars', 'car_flow', 'car_density', 'car_speed']]

print(df)
# 导出 CSV 文件
df.to_csv('data/csv/10'+which_file.__str__()+'/car_flow_density_speed.csv', index=False)

# 时间数据（按秒计）
time_data = df['seconds']*10

# 创建图表
plt.figure(figsize=(10, 6))  # 设置图表大小

# 图表 1: small_cars 和 large_cars 的车流量
plt.subplot(4, 1, 1)  # 创建子图，3 行 1 列，第一个子图
plt.plot(time_data, df['small_cars'], label='small_cars')
plt.plot(time_data, df['large_cars'], label='large_cars')

# 设置图表标题
plt.title("车流量-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("车流量（辆/秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 添加图例
plt.legend()

# 图表 2: small_cars 和 large_cars 的总和
plt.subplot(4, 1, 2)  # 创建子图，3 行 1 列，第二个子图
plt.plot(time_data, df['car_flow'], label='car_flow')

# 设置图表标题
plt.title("总车流量-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("车流量（辆/秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 添加图例
plt.legend()

# 图表 3: car_density
plt.subplot(4, 1, 3)  # 创建子图，3 行 1 列，第3个子图
plt.plot(time_data, df['car_density'], label='car_density')

# 设置图表标题
plt.title("车密度-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("密度（辆/米）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 添加图例
plt.legend()

# 图表 4: car_speed
plt.subplot(4, 1, 4)  # 创建子图，4 行 1 列，第4个子图
plt.plot(time_data, df['car_speed'], label='car_speed')

# 设置图表标题
plt.title("车速度-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("速度（千米/小时）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴速度上限为200
plt.ylim(top=200)  # 使用 plt.ylim(top=200) 设置纵轴上限

# 添加图例
plt.legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()
