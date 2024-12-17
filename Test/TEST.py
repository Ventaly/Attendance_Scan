# from paddleocr import PaddleOCR
# import cv2
#
#
#
# # 使用 pytesseract 进行 OCR 识别
# paddleocr = PaddleOCR(
#     use_gpu=False,
#     det_db_box_thresh=0.3,
#     det_db_unclip_ratio=2.0,
#     rec_algorithm='CRNN',
#     rec_char_dict_path='digit_dict.txt',
#
# )
# # 读取图片
# img = cv2.imread('cropped_image30.png',)
#
# # # 图像预处理（灰度化）
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# # # 二值化
# # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#
# # # 放大图片，提高分辨率
# # scale = 2
# # resized_img = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#
# paddleocr = PaddleOCR(lang='en', rec=True)
# # 打开需要识别的图片
# text = paddleocr.ocr(img, cls=False)
# print(text)
# if text==[None]:
#     print("空")

# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
#
# blue_lower = np.array([60, 100, 150])  # 蓝色的低范围
# blue_upper = np.array([150, 150, 255])  # 蓝色的高范围
# yellow_lower = np.array([200, 150, 0])  # 黄色的低范围
# yellow_upper = np.array([255, 255, 100])  # 黄色的高范围
#
# white_lower = np.array([230, 230, 230])  # 白色的低范围
# white_upper = np.array([255, 255, 255])  # 白色的高范围
# # 读取图像
# image = cv2.imread('cropped_image__1.png')
#
# # 将图像转换为RGB格式
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 将图像转换为二维数组
# image = image.reshape((image.shape[0] * image.shape[1], 3))
#
# # 使用K-means聚类
# kmeans = KMeans(n_clusters=2)  # 将颜色分成5类
# kmeans.fit(image)
#
# # 获取聚类中心的颜色
# colors = kmeans.cluster_centers_
#
# # 将颜色转换为整数
# colors = np.round(colors).astype(int)
#
# # 输出颜色
# print("Colors in the image:")
# for color in colors:
#     print(color)
#  # 判断颜色是否在白色范围内
# if np.all(color >= white_lower) and np.all(color <= white_upper):
#       print("未打卡")   # 白色范围内的颜色
#
#     # 判断颜色是否在蓝色范围内
# elif np.all(color >= blue_lower) and np.all(color <= blue_upper):
#         print("已打卡")
#
#
#     # 判断颜色是否在红色范围内
# elif np.all(color >= yellow_lower) and np.all(color <= yellow_upper):
#         print("异常")
#
#
#     # 默认返回未知，如果不在已定义的颜色范围内
# else:
#         print("未打卡")
import os
import cv2
from paddleocr import PaddleOCR, draw_ocr

# 初始化 PaddleOCR 模型
paddleocr = PaddleOCR(
    use_gpu=False,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=2.0,
    rec_algorithm='CRNN',
    rec_char_dict_path='digit_dict.txt',  # 自定义字符集文件路径
    lang='ch',  # 设置语言为中文
    show_log=False  # 关闭 PaddleOCR 的日志输出
)

# 检查图片路径
img_path = '王璟璟.jpg'
if os.path.exists(img_path):
    print("路径存在。")
else:
    raise FileNotFoundError("路径不存在，请检查文件路径是否正确。")

# 读取图片
image = cv2.imread(img_path)
if image is None:
    raise ValueError("无法加载图片，请检查图片格式是否支持。")

# OCR 处理
paddleocr = PaddleOCR(lang='ch', show_log=False)
result = paddleocr.ocr(image)

# 提取结果中的边界框和文本
boxes = [line[0] for line in result[0]]  # 获取边界框
texts = [line[1][0] for line in result[0]]  # 获取文字内容
scores = [line[1][1] for line in result[0]]  # 获取置信度

# 绘制边界框和文字
image_with_boxes = draw_ocr(
    image,
    boxes,
    txts=None,  # 不显示文字
    scores=None  # 不显示置信度

)

# 转换为 BGR 格式以适配 OpenCV 显示和保存
image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

# 保存结果图片
output_path = 'output_with_boxes.jpg'
cv2.imwrite(output_path, image_with_boxes)
print(f"处理后的图片已保存到：{output_path}")
