from paddleocr import PaddleOCR
import cv2

#

# 使用 pytesseract 进行 OCR 识别
paddleocr = PaddleOCR(
    use_gpu=False,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=2.0,
    rec_algorithm='CRNN',
    rec_char_dict_path='digit_dict.txt',

)
# 读取图片
img = cv2.imread('cropped_image30.png',)



paddleocr = PaddleOCR(lang='en', rec=True)
# 打开需要识别的图片
text = paddleocr.ocr(img, cls=False)
print(text)
if text==[None]:
    print("空")

import cv2
import numpy as np
from sklearn.cluster import KMeans

blue_lower = np.array([60, 100, 150])  # 蓝色的低范围
blue_upper = np.array([150, 150, 255])  # 蓝色的高范围
yellow_lower = np.array([200, 150, 0])  # 黄色的低范围
yellow_upper = np.array([255, 255, 100])  # 黄色的高范围

white_lower = np.array([230, 230, 230])  # 白色的低范围
white_upper = np.array([255, 255, 255])  # 白色的高范围
# 读取图像
image = cv2.imread('cropped_image__1.png')

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
      print("未打卡")   # 白色范围内的颜色

    # 判断颜色是否在蓝色范围内
elif np.all(color >= blue_lower) and np.all(color <= blue_upper):
        print("已打卡")


    # 判断颜色是否在红色范围内
elif np.all(color >= yellow_lower) and np.all(color <= yellow_upper):
        print("异常")


    # 默认返回未知，如果不在已定义的颜色范围内
else:
        print("未打卡")