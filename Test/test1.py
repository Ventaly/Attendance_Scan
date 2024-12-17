import cv2
import numpy as np

# 读取图像
image = cv2.imread("attendance.png")

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用二值化处理图像
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# 显示二值化处理后的图像（用于调试）
cv2.imshow("Binary Image", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制轮廓（方便调试）
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# 显示检测到的轮廓
cv2.imshow("Contours", image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 在每个元素块上绘制红框
image_with_red_boxes = image.copy()

for contour in contours:
    # 计算轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 绘制红框
    cv2.rectangle(image_with_red_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框

# 显示带红框的图像
cv2.imshow("Red Boxes on Elements", image_with_red_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
