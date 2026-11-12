from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi import Body
import cv2
import base64
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi import Query
from src.recognition_engine import recognize_face
from src.database import (init_db, save_stamp, get_logs, get_logs_by_month)

app = FastAPI()
attendance_logs = []
init_db()


templates = Jinja2Templates(
    directory="templates"
)

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="camera_stamp.html"
    )

# @app.post("/recognize")
# async def recognize(
#     data: dict = Body(...)
# ):

#     return {
#         "success": True,
#         "name": "sample",
#         "score": 0.95
#     }


@app.post("/recognize")
async def recognize(
    data: dict = Body(...)
):

    image = decode_image(
        data["image"]
    )

    result = recognize_face(
        image
    )

    return result

def decode_image(base64_string):

    image_data = base64_string.split(",")[1]

    image_bytes = base64.b64decode(
        image_data
    )

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    return image

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

@app.post("/stamp")
async def stamp(
    data: dict = Body(...)
):

    save_stamp(

        data["name"],

        data["type"],

        data["score"],

        data["time"]

    )

    return {
        "success": True
    }

@app.get("/logs")
async def logs_page(
    request: Request
):

    logs = get_logs()

    return templates.TemplateResponse(
        request=request,
        name="emp_history.html",
        context={
            "request": request,
            "logs": logs
        }
    )

@app.get("/api/logs")
async def api_logs(
    year: int,
    month: int
):

    # print(f"year={year}")
    # print(f"month={month}")

    logs = get_logs_by_month(
        year,
        month
    )

    # print(logs)

    result = []

    for row in logs:

        result.append({

            "name": row[0],
            "type": row[1],
            "score": row[2],
            "time": row[3]

        })

    # print(result)

    return result