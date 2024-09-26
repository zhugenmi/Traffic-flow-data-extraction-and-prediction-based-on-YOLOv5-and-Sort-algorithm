from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import font_manager

abTocd_len = 100
which_file=8

# 读取 CSV 文件
df_frame = pd.read_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_frame_rect.csv')

# 每25帧为一秒 将源数据分隔开 这一秒的车流量等于25帧中车流量的平均值
df = df_frame.groupby(df_frame.index // 25).mean()  # 使用 groupby 将数据按 25 帧一组进行分组，并计算每组的平均值

# 添加秒数列
df['seconds'] = df.index

# 计算密度
df['car_flow'] = df['small_cars'] + df['large_cars']*2

df['car_density'] = df['car_flow'] / abTocd_len

# 重新排序列
df = df[['seconds', 'small_cars', 'large_cars', 'car_flow','car_density']]

print(df)
# 导出 CSV 文件
df.to_csv('data/csv/10'+which_file.__str__()+'/car_flow_per_sec_rect.csv', index=False)

# 时间数据（按秒计）
time_data = df['seconds']*10

# 创建图表
plt.figure(figsize=(10, 6))  # 设置图表大小

# 图表 1: small_cars 和 large_cars 的车流量
plt.subplot(3, 1, 1)  # 创建子图，3 行 1 列，第一个子图
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
plt.subplot(3, 1, 2)  # 创建子图，3 行 1 列，第二个子图
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
plt.subplot(3, 1, 3)  # 创建子图，3 行 1 列，第3个子图
plt.plot(time_data, df['car_density'], label='car_density')

# 设置图表标题
plt.title("车密度-时间变化规律", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置横轴标签
plt.xlabel("时间（秒）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 设置纵轴标签
plt.ylabel("密度（辆/米）", fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

# 添加图例
plt.legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()