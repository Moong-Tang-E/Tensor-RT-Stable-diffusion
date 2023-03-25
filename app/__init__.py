from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
from threading import Lock, Thread
import json
import os
from module.main_process_txt2image import image_generator
from module.main_process_image2image import image_generator as image_generator2
from module.huggingface import import_index_from_hf
from io import BytesIO
from PIL import Image

app = FastAPI()
app.mount("/demo", StaticFiles(directory="demo"), name="demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "basic.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    modelsrc = config['model_path'].split(":")
    if modelsrc[0] == 'local':
        config['index_path'] = os.path.join(modelsrc[1], 'model_index.json')
    elif modelsrc[0] == 'hf':
        config['index_path'] = import_index_from_hf(modelsrc[1])
    else:
        raise Exception("Unknown model origin")
    
# Queue 생성
request_queue = Queue()
image_queue = Queue()
queue_lock = Lock()

@app.get("/q")
async def queues():
    request_length = request_queue.qsize()
    return str(request_length)

@app.post("/base")
async def txt2img(response: Response, json: dict = None):
    print("front req", json)
    with queue_lock:
        r = {
            "prompt": str(json['prompt']),
            "negprompt": str(json['negprompt']),
            "width": int(json['width']),
            "height": int(json['height']),
            "steps": int(json['steps']),
            "cfg": float(json['cfg']),
            "seed": int(json['seed']),
            "scheduler": str(json['scheduler']),
            "lpw": bool(json['lpw']),
            "img": str(json['img']) if 'img' in json else None,
            "strength": float(json['strength']) if 'strength' in json else None,
            "mode": str(json['mode']) # <- "I expect a response that is X"
        }
        request_queue.put(r)
    response_data = image_queue.get()
    if response_data['status'] == 'fail':
        response.status_code = 400
        return response_data

    if json['mode'] == 'file':
        file = response_data['content']
        pil_img = Image.open(BytesIO(file))
        return FileResponse(BytesIO(file), media_type='image/jpeg')
    elif json['mode'] == 'json':
        return JSONResponse(content=response_data)

# 이미지 생성 쓰레드 생성
image_generator_thread = Thread(
    target=image_generator, 
    args=(
        config, 
        request_queue,
        image_queue,
        )
    )
image_generator_thread.start()
