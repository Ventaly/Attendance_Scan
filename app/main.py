from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from Image_Preprocess import preprocess_image
from Paddle_OCR import                              extract_month_info
from Region_Detection import detect_and_display_regions,detect_calendar_contours
app = FastAPI()

# 假设你的模块名为your_module，并且包含了所需的函数

# 定义请求模型
class ImageUpload(BaseModel):
    image_path: str

@app.post("/attendance/")
async def attendance(image_upload: ImageUpload):
    try:
        # 调用你的预处理函数
        img, resized_img = preprocess_image(image_upload.image_path)

        # 调用区域检测函数
        month_region, date_region = detect_and_display_regions(img, resized_img)

        # 提取月份信息
        month_json = extract_month_info(month_region)

        # 提取日期和打卡状态
        daily_json = detect_calendar_contours(date_region)

        # 合并JSON数据
        combined_json = {
            "月份": daily_json["月份"],  # 使用每日记录中的月份
            "平均工时": month_json["平均工时"],
            "出勤天数": month_json["出勤天数"],
            "缺卡次数": month_json["缺卡次数"],
            "每日记录": daily_json["每日记录"]
        }

        # 返回合并后的JSON数据
        return json.dumps(combined_json, ensure_ascii=False, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
