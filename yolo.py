import os
import cv2
import torch
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR

from yolov5.utils.flask_rest_api.restapi import models
from ultralytics import YOLO
from yolov5 import  models# 根据您的YOLOv5实现导入模型架构

def load_model(weights='yolov5/runs/train/exp4/weights/best.pt'):
    """
    加载训练好的 YOLOv5 模型
    """
    model = torch.hub.load('/home/gqa/zx/Attendance_Scan/yolov5/', 'custom', ' /home/gqa/zx/Attendance_Scan/yolov5/runs/train/exp4/weights/best.pt', source='local')
  # 加载自定义训练模型
    return model


def detect_objects(model, image_path):
    """
    对上传的图片进行推理，获取类别坐标信息
    """
    # 加载图片
    img = cv2.imread(image_path)
    results = model(img)  # 推理

    # 获取结果（boxes：边界框，conf：置信度，names：类别名称）
    xyxy_tensor = results.xyxy[0].cpu().numpy()

    # 提取边界框坐标、识别概率和类别标签
    detections = []
    for detection in xyxy_tensor:

        x1, y1, x2, y2, confidence, class_id =  detection.tolist()
        # 确保边界框坐标不超出图像尺寸
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        # 根据 class_id 判断类别并裁剪
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

        if int(class_id) == 0:
            paddleocr = PaddleOCR(lang='ch', show_log=False)
            result = paddleocr.ocr(cropped_img)
            print(result)
            print(f"Detected Class 0: Saving cropped region daily.")

        elif int(class_id) == 1:
            paddleocr = PaddleOCR(lang='ch', show_log=False)
            result = paddleocr.ocr(cropped_img)
            print(result)
            print(f"Detected Class 1: Saving cropped region month.")

        elif int(class_id) == 2:
            paddleocr = PaddleOCR(lang='ch', show_log=False)
            result = paddleocr.ocr(cropped_img)
            print(result)
            print(f"Detected Class 2: Saving cropped region name.")

        else:
            print(f"Unknown Class {class_id} detected.")

    # 打印提取的结果

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        # 确保边界框坐标不超出图像尺寸
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        # 裁剪图像


        # 显示裁剪后的图像

        cv2.imwrite('path_to_save_cropped_image.jpg', cropped_img)
   # 获取边界框坐标
    conf = results.conf[0].gpu().numpy()  # 获取置信度
    names = results.names  # 类别名称

    return boxes, conf, names, img


def get_coordinates_and_crop_image(boxes, conf, names, image, conf_threshold=0.25):
    """
    根据坐标在原图中裁剪每个区域并返回坐标信息和裁剪后的图片
    """
    cropped_images = []
    coordinates = []

    image_height, image_width = image.shape[:2]

    for i, (x_center, y_center, width, height) in enumerate(boxes):
        # 过滤低置信度的预测
        if conf[i] < conf_threshold:
            continue

        # 计算像素坐标
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        # 获取裁剪区域
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)

        # 保存坐标和类别信息
        coordinates.append({
            'class': names[int(i)],
            'confidence': conf[i],
            'coordinates': [x_min, y_min, x_max, y_max]
        })

    return cropped_images, coordinates


def save_cropped_images(cropped_images, coordinates, output_dir='output'):
    """
    保存裁剪后的图片
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出文件夹

    for idx, cropped_image in enumerate(cropped_images):
        # 获取每个裁剪图像对应的类别名称
        class_name = coordinates[idx]['class']
        confidence = coordinates[idx]['confidence']
        coord = coordinates[idx]['coordinates']

        # 构建保存路径
        filename = f"{class_name}_{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.jpg"
        save_path = os.path.join(output_dir, filename)

        # 保存裁剪的图像
        cv2.imwrite(save_path, cropped_image)

        print(f"Saved cropped image: {save_path}")


def main(image_folder, model_weights='runs/train/exp4/weights/best.pt', conf_threshold=0.25):
    """
    主函数，执行上传、推理、裁剪和保存图片
    """
    # 加载模型
    model = load_model(weights=model_weights)

    # 遍历图片文件夹，进行推理和裁剪
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片
            print(f"Processing {image_filename}...")

            # 获取推理结果
            boxes, conf, names, img = detect_objects(model, image_path)

            # 获取每个区域的坐标并进行裁剪
            cropped_images, coordinates = get_coordinates_and_crop_image(boxes, conf, names, img, conf_threshold)

            # 保存裁剪后的图像
            save_cropped_images(cropped_images, coordinates, output_dir='output')

    print("Process complete.")


if __name__ == '__main__':
    # 输入图片文件夹路径
    image_folder = 'Image/test/'  # 更改为你的图片文件夹路径
    main(image_folder)
