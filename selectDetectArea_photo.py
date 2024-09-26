import cv2
import numpy as np

# 用于保存点击的坐标
points = []

# 缩放因子
scale_factor = 1.0


# 回调函数，获取鼠标点击的四个点
def select_points(event, x, y, flags, param):
    global points, scale_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将缩放后的坐标转换回原始图像坐标
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)

        points.append((original_x, original_y))
        if len(points) == 4:
            cv2.destroyAllWindows()

    elif event == cv2.EVENT_MOUSEWHEEL:
        global img_scaled
        if flags > 0:  # 滚轮向上滚动（放大）
            scale_factor *= 1.1
        else:  # 滚轮向下滚动（缩小）
            scale_factor /= 1.1

        # 更新缩放后的图像
        img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow('Image', img_scaled)


# 读取图片
def read_image_and_select_points(image_path):
    global points, scale_factor, img_scaled
    points = []
    scale_factor = 1.0

    img = cv2.imread(image_path)

    # 计算合适的最大窗口大小
    max_window_size = 800  # 可根据需要调整
    height, width = img.shape[:2]
    scale_factor = min(max_window_size / max(height, width), 1.0)

    # 缩放图像
    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # 设置窗口大小
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', int(width * scale_factor), int(height * scale_factor))

    # 设置鼠标回调函数
    cv2.setMouseCallback('Image', select_points)

    # 显示图片并等待用户选择四个点
    while True:
        img_copy = img_scaled.copy()

        # 画出已经选择的点（需要将原始坐标转换为缩放后的坐标）
        for point in points:
            scaled_x = int(point[0] * scale_factor)
            scaled_y = int(point[1] * scale_factor)
            cv2.circle(img_copy, (scaled_x, scaled_y), 5, (0, 255, 0), -1)

        cv2.imshow('Image', img_copy)

        if len(points) == 4:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return points


# 测试代码
image_path = 'data/img/103_40m.png'
selected_points = read_image_and_select_points(image_path)
print("Selected points:", selected_points)
