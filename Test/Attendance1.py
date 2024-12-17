import cv2
import numpy as np

# 读取图像
image = cv2.imread("attendance.png")

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用二值化处理图像，提升对比度
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

# 筛选符合条件的矩形区域
for contour in contours:
    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 设置一个阈值，假设日历区域的宽高比和面积符合一定条件
    if w > 300 and h > 550:  # 根据实际情况调整阈值
        # 提取日历区域
        calendar_roi = image[y:y + h, x:x + w]

        # 显示提取的日历区域
        cv2.imshow("Calendar ROI", calendar_roi)

        # 将日历区域从原图中删除，填充为白色（255, 255, 255）
        image[y:y + h, x:x + w] = (255, 255, 255)

# 显示修改后的图像
cv2.imshow("Image without Calendar", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
