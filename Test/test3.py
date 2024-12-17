import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import re
# 加载图像
import paddle
from sklearn.cluster import KMeans

paddle.disable_signal_handler()


# 图像预处理部分
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


def clean_and_convert_to_json(raw_text):
    """
    清理输入文本并转化为 JSON 格式。
    """
    # 清理空白字符和无关内容
    cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())  # 将多余空格替换为单个空格
    print("清理后的文本：", cleaned_text)

    title_match = re.search(r'(\d{1,2}月汇总)', cleaned_text)
    data_match = re.search(r'(\d+\.\d+)\s+(\d+)\s+(\d+)', cleaned_text)
    metrics = ["平均工时(小时)", "出勤天数(天)", "缺卡次数(次)"]

    if not title_match or not data_match:
        print("无法匹配所需数据")
        return None

    # 提取标题和数据内容
    title = title_match.group(1)
    data_values = list(map(float, data_match.groups()))

    # 生成 JSON 格式
    result = {
        "标题": title,
        "数据": {metrics[i]: data_values[i] for i in range(len(metrics))}
    }

    return json.dumps(result, ensure_ascii=False, indent=4)


def extract_month_info(month_region):
    """
    使用OCR提取月份信息。
    """
    if month_region is None:
        print("未检测到月份区域")
        return None


    paddleocr = PaddleOCR(lang='ch', show_log=False)
    # 打开需要识别的图片
    text = paddleocr.ocr(month_region)


    result_json = extract_to_json(text)

    return result_json


# 提取数据并存储为 JSON 格式
def extract_to_json(ocr_results):
    # 定义存储的字段
    extracted_data = {
        "月份": "",
        "平均工时": "",
        "出勤天数": "",
        "缺卡次数": ""
    }

    for entry in ocr_results[0]:  # 解析最外层列表
        text, confidence = entry[1]  # 提取文字和置信度
        if "汇总" in text:
            extracted_data["月份"] = text
        elif "平均工时" in text:
            extracted_data["平均工时"] = ocr_results[0][1][1][0]  # 关联数值部分
        elif "出勤天数" in text:
            extracted_data["出勤天数"] = ocr_results[0][2][1][0]  # 关联数值部分
        elif "缺卡次数" in text:
            extracted_data["缺卡次数"] = ocr_results[0][3][1][0]  # 关联数值部分

    # 转为 JSON 格式
    json_result = json.dumps(extracted_data, ensure_ascii=False, indent=4)
    json_obj = json.loads(json_result)
    return json_obj


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


def should_merge(rect, current_rect, horiz_thresh=5, vert_thresh=2):
    """
    判断两个矩形是否需要合并。

    参数:
    - rect: 第一个矩形 (x1, y1, x2, y2)
    - current_rect: 第二个矩形 (x1, y1, x2, y2)
    - horiz_thresh: 水平接壤的阈值
    - vert_thresh: 垂直高度接近的阈值

    返回:
    - True 或 False，表示是否需要合并
    """
    # 提取坐标
    x1, y1, x2, y2 = rect
    cx1, cy1, cx2, cy2 = current_rect

    # 高度差
    height_diff = abs((y2 - y1) - (cy2 - cy1))

    # 水平接壤（同一排且水平接近）
    if abs(x2 - cx1) < horiz_thresh and height_diff < 3:
        return True

    # 反向水平接壤（同一排且反向接近）
    if abs(x1 - cx2) < horiz_thresh and height_diff < 3:
        return True

    # # 垂直接近（同一列且垂直方向上接近）
    # if abs(y2 - cy1) < vert_thresh and abs(x1 - cx1) < horiz_thresh:
    #     return True

    return False


def detect_calendar_contours(img):
    """
    检测日历区域的轮廓并合并相邻文字或数字。

    参数:
    - img: 输入的日历区域的图像。

    返回:
    - 合并后带有轮廓的图像。
    """
    split_res = img.copy()  # 显示每个轮廓结构
    merge_res = img.copy()  # 显示合并后轮廓结构
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化处理，提取文本区域
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # 形态学操作去噪及连接相邻字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 检测轮廓
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 合并相邻轮廓
    result_json = merge_adjacent_contours(img, contours)

    return result_json


def merge_adjacent_contours(img, contours, distance_threshold=10):
    """
    合并相邻的轮廓，如果它们彼此接近，就圈在一起。

    参数:
    - img: 原始图像。
    - contours: 检测到的轮廓列表。
    - distance_threshold: 判断轮廓是否接近的阈值。

    返回:
    - img_with_merged_contours: 绘制了合并轮廓的结果图像。
    """
    # 合并的轮廓边界框
    merged_rectangles = []

    for cnt in contours:

        # 过滤小面积轮廓
        if cv2.contourArea(cnt) < 2:
            continue  # 如果轮廓面积小于最小面积，跳过它
        # 获取当前轮廓的外接矩形边界框
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 8:
            continue
        if 5 < w < 10:  # 如果宽度小于5
            w += 15  # 增加宽度
        if 5 < h < 15:  # 如果高度小于5
            h += 2  # 增加宽度

        current_rect = (x, y, x + w, y + h)  # (x1, y1, x2, y2)

        # 检查是否与已检测的合并区域接近
        merged = False

        for rect in merged_rectangles:
            # 检查当前轮廓是否与已存在的矩形接近
            if (
                    (abs(rect[0] - current_rect[2]) < 20 and  # 水平接近
                     abs(rect[1] - current_rect[3]) < 10)

            ):
                # 如果接近，将当前轮廓和目标轮廓合并
                new_rect = (
                    min(rect[0], current_rect[0]),
                    min(rect[1], current_rect[1]),
                    max(rect[2], current_rect[2]),
                    max(rect[3], current_rect[3])
                )
                rect = new_rect
                merged = True
                break

        if not merged:
            merged_rectangles.append(current_rect)
    filtered_contours = []
    result = merge_overlapping_rectangles(merged_rectangles)
    for cnt in result:
        width = cnt[2] - cnt[0]  # 计算宽度


        # 检查宽度是否小于最小阈值
        if width > 9:
            filtered_contours.append(cnt)  # 保留宽度满足条件的轮廓


    # 遍历合并区域进行 OCR 文字识别
    all_results = []
    i = 0
    for rect in filtered_contours:

        i = i + 1
        x1, y1, x2, y2 = rect
        # 从原图裁剪区域

        # 设置扩展的像素值
        padding = 10  # 向四周扩展10个像素

        # 确保裁剪区域的坐标不会超出图像边界
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        # 从原图裁剪区域
        cropped_img = img[y1:y2, x1:x2]

        # 显示或保存裁剪后的图片
        cv2.imwrite(f"cropped_image{i}.png", cropped_img)

        # 从原图裁剪区域，y2向下扩展10px
        new_y1 = y2  # 新的y1为当前区域的y2
        new_y2 = min(y2 + 10, img.shape[0])  # 向下扩展10px，确保不超出图像下边界

        cropped_img2 = img[new_y1:new_y2, x1:x2]

        # 保存裁剪后的图像
        cv2.imwrite(f"cropped_image__{i}.png", cropped_img2)
        paddleocr = PaddleOCR(
            use_gpu=False,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.5,
            rec_algorithm='CRNN',

        )
        # paddleocr = PaddleOCR(lang='ch', show_log=False)
        # 打开需要识别的图片
        text = paddleocr.ocr(cropped_img)



        if text==[None]:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            # 使用二值化处理图像
            _, binary_image = cv2.threshold(gray, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            text = paddleocr.ocr(binary_image)


        # 提取出 日期文本
        extracted_text = text[0][0][1][0] if text and text[0] else None
        status = check_status(cropped_img2)

        # 构建JSON格式的数据
        result = {
            "每日日期": extracted_text,
            "打卡状态": status
        }

        all_results.append(result)


    # 正则表达式用于匹配1-31范围内的日期

    # 正则表达式用于匹配日期标题和1-31范围内的日期
    date_title_pattern = re.compile(r'^\d{4}年\d{1,2}月$')
    date_pattern = re.compile(r'^(?:[012]?[1-9]|3[01])$')



    # 构建最终的JSON结构
    # 构建最终的JSON结构
    result_json = {
        "月份": [],
        '每日记录': []
    }
    daily_date = [item for item in all_results if date_pattern.match(item['每日日期'])]
    filtered_data2 = [item for item in all_results if date_title_pattern.match(item['每日日期'])]
    result_json["月份"] = filtered_data2[0]['每日日期']
    # 过滤出每日记录数据，并构建每日记录的JSON结构
    filtered_data = [item for item in all_results if date_pattern.match(item['每日日期'])]
    for item in filtered_data:
        daily_record = {
            "日期": int(item['每日日期']),
            "打卡状态": item['打卡状态']
        }
        result_json['每日记录'].append(daily_record)

    # 打印结果
    # 打印最终的JSON结构




    return result_json


def check_status(cropped_img):
    # 定义颜色的BGR范围
    # 定义颜色的BGR范围


    # 将图像转换为RGB格式
    image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # 将图像转换为二维数组
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # 使用K-means聚类
    kmeans = KMeans(n_clusters=2)  # 将颜色分成5类
    kmeans.fit(image)

    # 获取聚类中心的颜色
    colors = kmeans.cluster_centers_

    # 将颜色转换为整数
    colors = np.round(colors).astype(int)

    statuses = [check_color_status(color) for color in colors]
    # 判断颜色组合
    if "blue" in statuses and "white" in statuses:
        return "已打卡"  # 一白一蓝为已打卡
    elif statuses.count("white") == 2:
        return "未打卡"  # 两白为未打卡
    elif "yellow" in statuses and "white" in statuses:
        return "异常"  # 一白一黄为异常
    else:
        return "未知"  # 其他组合为未知


    # elif white_pixels > 0:
    #     return "未打卡"


def check_color_status(color):
    # 定义颜色的范围（BGR格式）
    blue_lower = np.array([60, 100, 150])  # 蓝色的低范围
    blue_upper = np.array([150, 150, 255])  # 蓝色的高范围

    yellow_lower = np.array([200, 150, 0])  # 黄色的低范围
    yellow_upper = np.array([255, 255, 100])  # 黄色的高范围

    white_lower = np.array([230, 230, 230])  # 白色的低范围
    white_upper = np.array([255, 255, 255])  # 白色的高范围

    # 判断颜色是否在蓝色范围内
    if np.all(color >= blue_lower) and np.all(color <= blue_upper):
        return "blue"  # 蓝色范围内的颜色

    # 判断颜色是否在黄色范围内
    elif np.all(color >= yellow_lower) and np.all(color <= yellow_upper):
        return "yellow"  # 黄色范围内的颜色

    # 判断颜色是否在白色范围内
    elif np.all(color >= white_lower) and np.all(color <= white_upper):
        return "white"  # 白色范围内的颜色

    # 默认返回未知，如果不在已定义的颜色范围内
    else:
        return "未知"
def is_overlapping(rect1, rect2):
    """
    判断两个矩形是否有重叠。

    参数:
    - rect1: 第一个矩形 (x1, y1, x2, y2)。
    - rect2: 第二个矩形 (x1, y1, x2, y2)。

    返回:
    - True: 如果有重叠。
    - False: 如果没有重叠。
    """
    x1_a, y1_a, x2_a, y2_a = rect1
    x1_b, y1_b, x2_b, y2_b = rect2

    # 检查水平和垂直方向是否有交集
    if x1_a < x2_b and x1_b < x2_a and y1_a < y2_b and y1_b < y2_a:
        return True  # 存在重叠
    return False  # 没有重叠


def merge_overlapping_rectangles(rectangles):
    """
    合并轮廓列表中有部分重叠的矩形。

    参数:
    - rectangles: 矩形列表 [(x1, y1, x2, y2), ...]

    返回:
    - merged_rectangles: 合并后的矩形列表。
    """
    merged = True  # 用于标记是否有新的合并
    while merged:
        merged = False  # 每次循环开始时重置标记
        new_rectangles = []
        skip = set()  # 跳过已合并的索引

        for i in range(len(rectangles)):
            if i in skip:  # 如果已合并，跳过
                continue
            rect1 = rectangles[i]
            merged_rect = rect1  # 初始化为当前矩形

            for j in range(i + 1, len(rectangles)):
                if j in skip:  # 如果已合并，跳过
                    continue
                rect2 = rectangles[j]

                if is_overlapping(merged_rect, rect2):  # 检查是否重叠
                    # 合并两个矩形
                    merged_rect = (
                        min(merged_rect[0], rect2[0]),
                        min(merged_rect[1], rect2[1]),
                        max(merged_rect[2], rect2[2]),
                        max(merged_rect[3], rect2[3]),
                    )
                    skip.add(j)  # 标记 rect2 已合并
                    merged = True  # 标记本轮有新的合并

            new_rectangles.append(merged_rect)  # 添加合并后的矩形

        # 更新矩形列表
        rectangles = [rect for idx, rect in enumerate(rectangles) if idx not in skip]
        rectangles = new_rectangles

    return rectangles


def extract_date_and_status(date_regions):
    """
    从日期区域提取日期和对应的打卡状态。
    返回日期和打卡状态的字典。
    """
    if date_regions is None:
        print("未检测到月份区域")
        return None


# 图像预处理
if __name__ == '__main__':
    # 配置tesseract路径（根据你自己的安装路径进行配置）

    img, resized_img = preprocess_image("attendance.png")
    month_region, month_regions, date_region, date_regions = detect_and_display_regions(resized_img)
    # 提取月份信息
    cropped_result1 = crop_region_from_original(img, month_regions)
    month_json = extract_month_info(cropped_result1)






    cropped_result2 = crop_region_from_original(img, date_regions)
    # 提取日期和打卡状态
    daily_json = detect_calendar_contours(cropped_result2)


    # 合并JSON数据
    combined_json = {
        "月份": month_json["月份"],
        "平均工时": month_json["平均工时"],
        "出勤天数": month_json["出勤天数"],
        "缺卡次数": month_json["缺卡次数"],
        "月份": daily_json["月份"],  # 使用每日记录中的月份

        "每日记录": daily_json["每日记录"]
    }

    # 打印合并后的JSON数据
    print(json.dumps(combined_json, ensure_ascii=False, indent=4))
