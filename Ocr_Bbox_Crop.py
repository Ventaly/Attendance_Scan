from paddleocr.tools.infer.utility import draw_ocr
import cv2
import numpy as np
from sklearn.cluster import KMeans

def Kmeans(image):
    blue_lower = np.array([60, 100, 150])  # 蓝色的低范围
    blue_upper = np.array([150, 150, 255])  # 蓝色的高范围
    yellow_lower = np.array([200, 150, 0])  # 黄色的低范围
    yellow_upper = np.array([255, 255, 100])  # 黄色的高范围

    white_lower = np.array([230, 230, 230])  # 白色的低范围
    white_upper = np.array([255, 255, 255])  # 白色的高范围


    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像转换为二维数组
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # 使用K-means聚类
    kmeans = KMeans(n_clusters=2)  # 将颜色分成5类
    kmeans.fit(image)

    # 获取聚类中心的颜色
    colors = kmeans.cluster_centers_

    # 将颜色转换为整数
    colors = np.round(colors).astype(int)

    # 输出颜色
    print("Colors in the image:")
    for color in colors:
        print(color)
    # 判断颜色是否在白色范围内
    if np.all(color >= white_lower) and np.all(color <= white_upper):
        print("未打卡")  # 白色范围内的颜色

    # 判断颜色是否在蓝色范围内
    elif np.all(color >= blue_lower) and np.all(color <= blue_upper):
        print("已打卡")


    # 判断颜色是否在红色范围内
    elif np.all(color >= yellow_lower) and np.all(color <= yellow_upper):
        print("异常")


    # 默认返回未知，如果不在已定义的颜色范围内
    else:
        print("未打卡")

def find_date_distance(result, start_date, end_date):
    found_start_date = None
    found_end_date = None

    # 遍历result数组，寻找包含特定数字的文本
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            if str(start_date) == text:
                found_start_date =word_info[0][2][1]# (日期, y_min, y_max)
            elif str(end_date) == text:
                found_end_date =  word_info[0][0][1]

    # 检查是否找到了起始和结束日期，如果没找到，寻找下一对差值为7的日期对
    if found_start_date and found_end_date:

        distance =found_end_date-found_start_date  # 计算起始日期的下边界和结束日期的上边界之间的距离
        return distance
    else:
        next_start_date = start_date + 1
        next_end_date = end_date + 1

        # 递归调用，寻找下一个日期对
        return find_date_distance(result, next_start_date, next_end_date)
        # 如果没有找到任何日期对，返回None

    return None

def crop_region(result,image):
    boxes = [line[0] for line in result[0]]  # 获取边界框
    texts = [line[1][0] for line in result[0]]  # 获取文字内容
    scores = [line[1][1] for line in result[0]]  # 获取置信度
    # 调用函数，例如寻找2日和9日之间的距离
    # 绘制边界框和文字
    image_with_boxes = draw_ocr(
        image,
        boxes,
        txts=None,  # 不显示文字
        scores=None  # 不显示置信度
    )
    distance = find_date_distance(result, 3, 10)
    # 确定裁剪坐标
    x1,y1,x2,y2=boxes
    y1=y2
    y2 = y2 + distance
    # 裁剪出所在的位置
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    result_status=Kmeans(cropped_image)

    return  texts,result_status
