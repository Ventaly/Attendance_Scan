import cv2
import numpy as np


def detect_and_display_regions(img,resized_img):
    """
    检测图像中的月份和日期区域，并显示结果。

    参数:
    - image: 待处理的图像。
    - min_w_month: 月份区域的最小宽度。
    - min_h_month: 月份区域的最小高度。
    - min_w_date: 日期区域的最小宽度。
    - min_h_date: 日期区域的最小高度。
    """
    # Canny 边缘检测
    edges = cv2.Canny(resized_img, 1, 10)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化区域列表
    month_region = None
    date_regions = []
    month_regions = []
    max_area=None
    largest_contour=None
    # 遍历轮廓，找到面积最大的轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    # 检查是否找到外层轮廓
    largest_inner_contour = None
    if largest_contour is not None:
        # 在原图上绘制外层最大轮廓
        cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)  # 用绿色绘制外层最大轮廓

        # 提取感兴趣区域（ROI）
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = img[y:y + h, x:x + w]
        # 灰度化处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 去噪处理（高斯模糊）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 提高对比度（直方图均衡化）
        resized = cv2.equalizeHist(blurred)
        # 在 ROI 内检测轮廓
        roi_edges = cv2.Canny(resized, 50, 150)
        roi_contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 查找 ROI 内的最大轮廓
        max_inner_area = 0
        largest_inner_contour = None

        for cnt in roi_contours:
            area = cv2.contourArea(cnt)
            if area > max_inner_area:
                max_inner_area = area
                largest_inner_contour = cnt

        # 检查是否找到内层轮廓
        if largest_inner_contour is not None:
            # 将内层轮廓偏移到原图坐标系中
            largest_inner_contour = largest_inner_contour + np.array([x, y])

            print(f"Inner contour area: {max_inner_area}")
        else:
            print("No inner contour found in ROI.")
    else:
        print("No outer contour found!")




    cv2.destroyAllWindows()

    return month_region, month_regions, date_region, date_regions


def crop_region_from_original(img, region):
    """
    根据裁剪区域从原图中获取未变形的区域。

    参数:
    - img: 原始彩色图像。
    - region: 裁剪区域的边界，格式为 (x, y, w, h)。

    返回:
    - cropped_region: 从原图中裁剪出来的真实区域。
    """
    # 从原始图像中裁剪出未变形的区域
    for region in region:
        x, y, w, h = region
    cropped_region = img[y:y + h, x:x + w]

    return cropped_region