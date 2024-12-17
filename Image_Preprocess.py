import cv2


def preprocess_image(image_path):
    # 加载图像
    img = cv2.imread(image_path)

    # 灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪处理（高斯模糊）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 提高对比度（直方图均衡化）
    resized = cv2.equalizeHist(blurred)

    return img, resized