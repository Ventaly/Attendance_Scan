import cv2
import numpy as np
from paddleocr import PaddleOCR
import  json
import re
# 加载图像
import paddle

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

    # # 插值放大（可选，按需求）
    # scale = 1  # 放大倍数
    # height, width = enhanced.shape
    # resized = cv2.resize(enhanced, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    #显示预处理后的结果
    # cv2.imshow("Preprocessed Image", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

    # # 绘制所有轮廓
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽为2
    #
    # # 显示图像
    # cv2.imshow('Contours', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 初始化区域列表
    month_region = None
    date_regions = []
    month_regions = []
    # 遍历轮廓
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 根据面积过滤区域（假设月份区域较大）
        if min_w_month< w  and min_h_month<h <200 :
            month_regions.append((x, y, w, h))
            # 月份区域的预估大小

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝框表示月份区域

        if w > min_w_date and h > min_h_date:  # 日期区域的预估大小
            date_regions.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # # 显示定位结果
    # cv2.imshow("Region Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 显示月份区域和部分日期区域
    for region in month_regions:
        x, y, w, h = region
        month_region = image[y:y + h, x:x + w]
        # cv2.imshow("Date Region", month_region)
        # cv2.waitKey(0)

    for region in date_regions:
        x, y, w, h = region
        date_region = image[y:y + h, x:x + w]
        # cv2.imshow("Date Region", date_region)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

    return month_region,month_regions, date_region,date_regions
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

    # OCR 提取文字text = pytesseract.image_to_string(image, lang='chi_sim')
    # reader=easyocr.Reader(['ch_sim','en'])
    # result=reader.readtext(month_region)
    # text = pytesseract.image_to_string(month_region, lang='chi_sim')  # psm 7 适合单行文本
    paddleocr = PaddleOCR(lang='ch', show_log=False)
     # 打开需要识别的图片
    text = paddleocr.ocr(month_region)
    print("OCR提取的月份信息：", text)
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
    return json_result


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

    # 可视化结果
    # cv2.imshow("Cropped Region", cropped_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    img_with_merged_contours = merge_adjacent_contours(img, contours)

    return img_with_merged_contours


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
        if 5<w < 10:  # 如果宽度小于5
            w += 15  # 增加宽度
        if 5 < h < 15:  # 如果高度小于5
            h += 2  # 增加宽度

        current_rect = (x, y, x + w, y + h)  # (x1, y1, x2, y2)

        # 检查是否与已检测的合并区域接近
        merged = False

        for rect in merged_rectangles:
            # 检查当前轮廓是否与已存在的矩形接近
            if (
                    (abs(rect[0] - current_rect[2]) < 20 and   # 水平接近
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
        print(f"矩形: {cnt}, 宽度: {width}")

        # 检查宽度是否小于最小阈值
        if  width> 9:
            filtered_contours.append(cnt)  # 保留宽度满足条件的轮廓
    # 绘制结果框
    img_with_merged_contours = img.copy()
    for rect in filtered_contours:
        cv2.rectangle(
            img_with_merged_contours,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            (0, 0, 255),
            1
        )

    # 显示检测结果
    cv2.imshow("Merged Contours", img_with_merged_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 遍历合并区域进行 OCR 文字识别
    recognized_texts = []
    for rect in filtered_contours:


        x1, y1, x2, y2 = rect
        # 从原图裁剪区域
        new_x1 = min(x1 - 25, img.shape[1])
        new_y2 = min(y2+25, img.shape[0])
        cropped_img = img[y1:new_y2, x1:x2]


        # 保存裁剪后的图像
        cropped_img.save('cropped_image.png')
        # # 显示裁剪的图像区域
        # cv2.imshow("Cropped Image", cropped_img)
        # cv2.waitKey(0)  # 等待键盘输入
        # cv2.destroyAllWindows()  # 关闭窗口



        # 使用 pytesseract 进行 OCR 识别
        paddleocr = PaddleOCR(lang='ch', show_log=False)
        # 打开需要识别的图片
        text = paddleocr.ocr(cropped_img)
        print("OCR提取的月份信息：", text)
        recognized_texts.append({ "text": text.strip()})
        print(recognized_texts)



    return img_with_merged_contours


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
    month_region,month_regions, date_region,date_regions = detect_and_display_regions(resized_img)
    # 提取月份信息
    cropped_result1 = crop_region_from_original(img, month_regions)
    month_info = extract_month_info(cropped_result1)
    print("月份信息：", month_info)
    cropped_result2 = crop_region_from_original(img, date_regions)
    # 提取日期和打卡状态

    cells = detect_calendar_contours(cropped_result2)
    date_status = extract_date_and_status(cropped_result2)
    print("每日打卡信息：")
    #
    # cv2.imshow("Detected Grid", grid_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()