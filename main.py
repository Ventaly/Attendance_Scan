import json
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from Image_Preprocess import preprocess_image
from Paddle_OCR import extract_month_info, detect_calendar_contours
from Region_Detection import detect_and_display_regions, crop_region_from_original

if __name__ == '__main__':


    img, resized_img = preprocess_image("Image/2024年11月/11月份E点通打卡-谢超/20241101.png")
    month_region, month_regions, date_region, date_regions = detect_and_display_regions(img,resized_img)
    # 提取月份信息
    cropped_result1 = crop_region_from_original(img, month_regions)
    month_json = extract_month_info(cropped_result1)






    cropped_result2 = crop_region_from_original(img, date_regions)
    # 提取日期和打卡状态
    daily_json = detect_calendar_contours(cropped_result2)


    # 合并JSON数据
    combined_json = {
        "Month": month_json["Month"],
        "Average_Working_Hours": month_json["Average_Working_Hours"],
        "Attendance_Days": month_json["Attendance_Days"],
        "Missed_Card_Times": month_json["Missed_Card_Times"],
        "Daily_Records": daily_json["Month"],  # 使用每日记录中的月份

        "Daily_Records": daily_json["Daily_Records"]
    }

    # 打印合并后的JSON数据
    json_str =json.dumps(combined_json, ensure_ascii=False, indent=4)
    print(json)


    print("JSON数据集已成功写入文件。")

