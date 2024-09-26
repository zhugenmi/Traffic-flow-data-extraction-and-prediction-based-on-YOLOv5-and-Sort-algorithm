import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
1、设置检测区域四个顶点ABCD ，这由points定义
2、运行car_flow_line.py生成车流量（辆/秒）
3、设置AB到CD的实际距离abTocd_len，运行car_flow_rect.py生成密度（辆/米）
4、运行generateSpeedDensity.py,运用公式计算速度=流量/密度，同时包含每秒小型车和大型车流量情况
'''

# 判断车辆中心点是否在界线上的精度
epsilon = 8
distance_threshold = 20  # 用于判断两帧中车辆是否为同一辆的距离阈值

def is_on_line(x1, y1, x2, y2, x, y, epsilon):
    """
    判断点(x, y) 是否在由点(x1, y1) 和 (x2, y2) 确定的直线上。

    Args:
        x1, y1: 第一个点的坐标。
        x2, y2: 第二个点的坐标。
        x, y: 要判断的点的坐标。

    Returns:
        True 如果点在直线上，否则 False。
    """
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # 计算点到直线的距离，并判断是否在误差范围内
    distance = abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

    return distance <= epsilon  # 判断是否在误差范围内


def calculate_distance(point1, point2):
    """计算两个点之间的欧几里得距离"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s为轻量化模型

# 打开视频
video_path = 'data/103_20s.mp4'

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
car_flow_per_frame_line = [{'small_cars': 0, 'large_cars': 0}]  # 每帧中界线上的车辆情况
car_flow_per_frame_rect = [{'small_cars': 0, 'large_cars': 0}]  # 每帧中监测区域内的车辆情况


# 根据监控视频设置检测区域的顶点列表 points
points= [(39, 234), (957, 626), (1104, 230), (438, 151)]   # 四个点，前两个点作为监测车流量的界限

# 将 points 转换为 NumPy 数组，并调整为 OpenCV 需要的形状
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))  # 转换为 OpenCV 格式

frame_count = 0

# 记录前一帧中检测到的辆车的中心位置
pre_centers = set()
# 读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 绘制多边形检测区域
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # 绿色多边形

    # 使用 YOLOv5 模型进行车辆检测
    results = model(frame)  # 检测结果
    detections = results.pandas().xyxy[0]  # 获取检测结果的坐标

    # 每一帧的车辆计数器
    vehicle_count_rect = {'car': 0, 'truck': 0, 'bus': 0}
    vehicle_count_line = {'car': 0, 'truck': 0, 'bus': 0}

    # 记录当前帧检测到的车辆
    detected_centers = set()
    # 遍历检测到的物体，过滤出在多边形内的车辆
    for index, row in detections.iterrows():
        if row['name'] in ['car', 'truck', 'bus']:  # 检测车辆类型
            # 获取检测框的坐标
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # 计算车辆中心点
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            center = (center_x, center_y)

            # 检查车辆中心是否在多边形内
            if cv2.pointPolygonTest(pts, center, False) >= 0:  # 判断中心点是否在区域内
                # 更新计数器
                vehicle_count_rect[row['name']] += 1

                # 在车辆上绘制矩形框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 蓝色矩形框标记车辆

                # 在中心点绘制红色圆点
                cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)  # 红色圆点标记中心点

                # 显示车辆类别
                cv2.putText(frame, row['name'], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # 判断车辆中心点是否在界线上
            if is_on_line(points[0][0], points[0][1], points[1][0], points[1][1], center_x, center_y, epsilon):
                # ncars_on_line += 1
                # print('center: ' + center.__str__() + ' ,points[0]: ' + points[0].__str__() + ' ,points[1] ' + points[
                #     1].__str__() + ' ' + ncars_on_line.__str__())
                # 在检测到车辆时
                if center in pre_centers:
                    continue

                detected_centers.add(center)
                # 更新计数器
                vehicle_count_line[row['name']] += 1

                # 在车辆上绘制矩形框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 蓝色矩形框标记车辆

                # 在中心点绘制绿色圆点
                cv2.circle(frame, center, radius=5, color=(0, 255, 0), thickness=-1)  # 红色圆点标记中心点

                # 显示车辆类别
                cv2.putText(frame, row['name'], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    pre_centers = detected_centers
    # 实时显示车辆统计信息
    info_text = f"Small_cars: {vehicle_count_rect['car']} | Large_cars: {vehicle_count_rect['truck'] + vehicle_count_rect['bus']}"
    cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 显示带有检测和标记的帧
    cv2.imshow('Vehicle Detection', frame)
    car_flow_per_frame_line.append(
        {'small_cars': vehicle_count_line['car'],
         'large_cars': vehicle_count_line['bus'] + vehicle_count_line['truck']})
    # print(vehicle_count)

    car_flow_per_frame_rect.append(
        {'small_cars': vehicle_count_rect['car'],
         'large_cars': vehicle_count_rect['bus'] + vehicle_count_rect['truck']})
    # print(vehicle_count)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break
    # 按 'q' 键退出
    # key = cv2.waitKey(0)  # 暂停，等待用户输入
    # if key == ord('q'):
    #     break
    # elif key == ord('e'):  # 按 'e' 键调整 epsilon
    #     epsilon = int(input("请输入新的 epsilon 值: "))
    # elif key == ord('n'):  # 按 'n' 键继续播放
    #     continue
    frame_count += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 将 car_flow_per_frame_line 转换为 DataFrame
df1 = pd.DataFrame(car_flow_per_frame_line)

df1.to_csv("data/csv/103/car_flow_per_frame_line.csv")  # 该文件包含了每一帧经过界线AB的车辆情况

# 将 car_flow_per_frame_rect 转换为 DataFrame
df2 = pd.DataFrame(car_flow_per_frame_rect)

df2.to_csv("data/csv/103/car_flow_per_frame_rect.csv")  # 该文件包含了每一帧中在监测矩形方框范围内的车辆情况
