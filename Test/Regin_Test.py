import cv2

# 加载图片与模板
image = cv2.imread("../Image/attendance.png")
template = cv2.imread("../Image/contours_image.png", 0)

# 转灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 模板匹配
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)  # 匹配位置

# 提取匹配坐标并裁剪
h, w = template.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
cv2.imwrite("cropped_calendar.jpg", cropped)
