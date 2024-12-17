import cv2


def detect_and_display_regions(image, min_w_month=200, min_h_month=120, min_w_date=200, min_h_date=300):
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
    edges = cv2.Canny(image, 1, 10)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化区域列表
    month_region = None
    date_regions = []
    month_regions = []
    # 遍历轮廓
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 根据面积过滤区域（假设月份区域较大）
        if min_w_month < w and min_h_month < h < 200:
            month_regions.append((x, y, w, h))
            # 月份区域的预估大小

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝框表示月份区域

        if w > min_w_date and h > min_h_date:  # 日期区域的预估大小
            date_regions.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示月份区域和部分日期区域
    for region in month_regions:
        x, y, w, h = region
        month_region = image[y:y + h, x:x + w]
        # cv2.imshow("Date Region", month_region)
        # cv2.waitKey(0)

    for region in date_regions:
        x, y, w, h = region
        date_region = image[y:y + h, x:x + w]

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