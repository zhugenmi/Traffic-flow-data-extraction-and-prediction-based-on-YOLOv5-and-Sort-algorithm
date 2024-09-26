import torch
import cv2
import numpy as np
import pandas as pd
from sort.sort import Sort  # 导入 SORT 追踪算法

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 使用轻量化的 YOLOv5 模型

# 实际物理距离
actual_distance_meters = 40

# 跟踪区域（实际物理距离）
points= [(39, 234), (957, 626), (1104, 230), (438, 151)]  # 监测区域的四个顶点
pts = np.array(points, np.int32).reshape((-1, 1, 2))

# 打开视频
video_path = 'data/103_20s.mp4'

cap = cv2.VideoCapture(video_path)

# 获取经过采样后的视频的帧率
fps_sampled = cap.get(cv2.CAP_PROP_FPS)

# 将帧率恢复为原帧率（假设采样率为 10，因此帧率乘以 10）
fps = fps_sampled

# 获取视频宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义追踪器
tracker = Sort()

# 保存每辆车的速度信息
vehicle_speeds = {}

def calculate_speed(prev_position, current_position, fps, distance_per_pixel):
    """
    根据车辆在连续两帧中的位置变化计算速度。

    Args:
        prev_position: 上一帧车辆的位置 (x, y)。
        current_position: 当前帧车辆的位置 (x, y)。
        fps: 视频的帧率。
        distance_per_pixel: 每个像素对应的实际物理距离 (米/像素)。

    Returns:
        车辆的速度 (米/秒)。
    """
    # 计算帧间车辆的像素位移
    displacement_pixels = np.sqrt((current_position[0] - prev_position[0]) ** 2 +
                                  (current_position[1] - prev_position[1]) ** 2)

    # 转换为实际物理距离
    displacement_meters = displacement_pixels * distance_per_pixel

    # 根据帧率计算每秒的位移（速度）
    speed_meters_per_sec = displacement_meters * fps

    return speed_meters_per_sec


# 获取每个像素的实际物理距离
distance_per_pixel = actual_distance_meters / np.sqrt(
    (points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 检测
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # 用于存储 SORT 需要的格式 [xmin, ymin, xmax, ymax, confidence]
    dets = []
    for _, row in detections.iterrows():
        if row['name'] in ['car', 'truck', 'bus']:  # 只检测车辆类型
            dets.append([row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence']])

    # 将检测到的物体传递给 SORT 进行跟踪，返回追踪结果 [xmin, ymin, xmax, ymax, track_id]
    if len(dets) > 0:
        dets = np.array(dets)
        tracked_objects = tracker.update(dets)

        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

            # 获取车辆中心位置
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            center = (center_x, center_y)

            # 如果该车辆之前没有记录位置，则初始化
            if track_id not in vehicle_speeds:
                vehicle_speeds[track_id] = {'prev_position': center, 'speed': 0}
            else:
                # 计算速度
                prev_position = vehicle_speeds[track_id]['prev_position']
                speed = calculate_speed(prev_position, center, fps, distance_per_pixel)

                # 更新车辆速度和位置
                vehicle_speeds[track_id]['speed'] = speed
                vehicle_speeds[track_id]['prev_position'] = center

            # 绘制跟踪边界框和 ID
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} Speed: {vehicle_speeds[track_id]['speed']:.2f} m/s",
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示视频
    cv2.imshow('Vehicle Detection and Speed Calculation', frame)

    if cv2.waitKey(int(1000 / fps_sampled)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 将车辆速度数据保存为 CSV 文件
vehicle_speed_data = [{'track_id': track_id, 'speed': info['speed']} for track_id, info in vehicle_speeds.items()]
df = pd.DataFrame(vehicle_speed_data)
df.to_csv("data/csv/103/vehicle_speeds.csv", index=False)
