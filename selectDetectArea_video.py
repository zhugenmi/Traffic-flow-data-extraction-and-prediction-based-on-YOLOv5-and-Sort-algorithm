import cv2
import numpy as np

# 用于保存点击的坐标
points = []


# 回调函数，获取鼠标点击的四个点
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 当鼠标左键点击时，记录点击的坐标
        points.append((x, y))
        if len(points) == 4:  # 当点击了四个点时，关闭鼠标响应
            cv2.destroyAllWindows()


# 视频路径或图像路径
video_path = 'data/103_20s.mp4'


# 打开视频的第一帧，或者你可以直接加载图像
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# 检查是否成功读取视频帧
if not ret:
    print("无法读取视频帧")

# 设置鼠标回调函数
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_points)

# 显示帧并等待用户选择四个点
while True:
    frame_copy = frame.copy()

    # 画出已经选择的点
    for point in points:
        cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)  # 绿色小圆点标记点击点

    cv2.imshow('Frame', frame_copy)

    if len(points) == 4:  # 当选择了四个点后退出
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 绘制倾斜的矩形（四边形）
if len(points) == 4:
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # 绘制多边形

    # 显示最终选择的矩形
    cv2.imshow('Selected Rectangle', frame)
    cv2.waitKey(0)

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 按顺序打印出选择的四个点
print("Selected points:", points)
