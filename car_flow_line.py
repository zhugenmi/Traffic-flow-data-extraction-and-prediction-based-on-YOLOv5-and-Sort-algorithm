from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import font_manager

which_file=3

# 读取 CSV 文件
df_frame = pd.read_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_frame_line.csv')

# 每25帧为一秒，将源数据分隔开，一秒内车流量等于25帧中过线的车辆总数
df = df_frame.groupby(df_frame.index // 25).sum()

# 添加秒数列
df['seconds'] = df.index
span=20

# 计算总车流量 小型车+大型车*2
df['car_flow'] = df['small_cars'] + df['large_cars']*2

# df['car_flow']=df['car_flow'].ewm(span=span, ignore_na=True, adjust= True).mean()
# print(df['car_flow'])
# df['car_flow'] = df['car_flow']/10 # 视频10倍加速，恢复
# print(df['car_flow'])
# 重新排序列
df = df[['seconds', 'small_cars', 'large_cars', 'car_flow']]

# 导出 CSV 文件
# df.to_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_sec_line.csv', index=False)

# 时间数据（按秒计）
time_data = df['seconds']*10

# 创建图表
plt.figure(figsize=(10, 6))  # 设置图表大小

# 绘制车辆过线数量的折线图，使用 df 的第一列
plt.plot(time_data, df['car_flow'], label='Vehicles')

# 设置图表标题
plt.title("车流量-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("车流量（辆/秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 显示图表
plt.legend()
plt.show()
