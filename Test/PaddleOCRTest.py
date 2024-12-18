
import os
import cv2
from paddleocr import PaddleOCR, draw_ocr





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
        print()
        print()
        distance =found_end_date-found_start_date  # 计算起始日期的下边界和结束日期的上边界之间的距离
        return distance
    else:
        find_date_distance(result,start_date+1,end_date+1)

    return None




if __name__ == '__main__':
    # 初始化 PaddleOCR 模型
    paddleocr = PaddleOCR(
        use_gpu=False,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=2.0,
        rec_algorithm='CRNN',

        lang='ch',  # 设置语言为中文
        show_log=False  # 关闭 PaddleOCR 的日志输出
    )

    # 检查图片路径
    img_path = 'Image/任思源.jpg'
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
    # 调用函数，例如寻找2日和9日之间的距离
    # 绘制边界框和文字
    image_with_boxes = draw_ocr(
        image,
        boxes,
        txts=None,  # 不显示文字
        scores=None  # 不显示置信度

    )

    # 确定裁剪坐标
    x1, y1 = 82, 149  # 左上角坐标
    x2, y2 = 94, 161  # 右下角坐标
    y2 = y2 + 15
    # 裁剪出3所在的位置
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

    # 保存结果图片的目录
    save_dir = 'Image/test'
    # 确保目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = os.path.join(save_dir, 'cropped_3.jpg')
    # 保存裁剪后的图片
    cv2.imwrite(output_path, cropped_image)
    # 转换为 BGR 格式以适配 OpenCV 显示和保存
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

    # 保存结果图片的完整路径
    output_path = os.path.join(save_dir, '任思源.jpg')
    # 保存结果图片


    cv2.imwrite(output_path, image_with_boxes)
    print(f"处理后的图片已保存到：{output_path}")
