from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from Image_Preprocess import preprocess_image
from Paddle_OCR import extract_month_info, detect_calendar_contours
from Region_Detection import detect_and_display_regions, crop_region_from_original

app = FastAPI()


@app.get("/status")
def read_status():
    """
    检查服务状态
    """
    return {"status": "Running"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """curl -X POST http://http://172.16.19.208:8080/upload
    上传图片并处理，返回结果
    """
    # 保存上传的图片到本地
    file_path = f"resources/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 图像预处理
    img, resized_img = preprocess_image(file_path)

    # 区域检测
    month_region, month_regions, date_region, date_regions = detect_and_display_regions(resized_img)

    # 提取月份信息
    cropped_month = crop_region_from_original(img, month_regions)
    month_json = extract_month_info(cropped_month)

    # 提取每日信息
    cropped_date = crop_region_from_original(img, date_regions)
    daily_json = detect_calendar_contours(cropped_date)

    # 合并JSON数据
    combined_json = {

        "Month": month_json["Month"],
        "Average_Working_Hours": month_json["Average_Working_Hours"],
        "Attendance_Days": month_json["Attendance_Days"],
        "Missed_Card_Times": month_json["Missed_Card_Times"],


        "Daily_Records": daily_json["Daily_Records"]
    }

    # 返回结果
    return JSONResponse(content=combined_json)
