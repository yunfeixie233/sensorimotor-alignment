# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import argparse
import base64
import io
import os
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from llava import conversation as clib
from llava.media import Image, Video
from PIL import Image as PILImage
from pydantic import BaseModel
from termcolor import colored

from robo_orchard_lab.models.aux_think import AuxThink

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./aux_think_8B")
parser.add_argument("--port", type=int, required=True, help="Port to serve on")

args = parser.parse_args()
model_path = args.model

model = None
clib.default_conversation = clib.conv_templates["auto"].copy()

app = FastAPI(title="VLN Aux-think Inference API")


@app.on_event("startup")
async def load_model_on_startup():
    global model
    print(
        "--- Loading model and processor on startup ---"
    )  # Optional: temporary print

    model = AuxThink.load_model(model_path)
    print("--- Model and processor loaded ---")  # Optional: temporary print


class InferenceRequest(BaseModel):
    prompt: str
    images: List[str]


class InferenceResponse(BaseModel):
    generated_text: str


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded or failed to load"
        )

    prompt = []
    os.makedirs("tmp", exist_ok=True)
    tmp_dir = os.path.join("tmp", str(args.port))
    os.makedirs(tmp_dir, exist_ok=True)
    if request.images is not None:
        image_paths = []

        for idx, image_base64 in enumerate(request.images):
            try:
                image_data = base64.b64decode(image_base64)
                image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
                image_path = f"tmp/{args.port}/image_{idx}.jpg"
                image.save(image_path)
                image_paths.append(image_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {idx}: {str(e)}",
                )

        for image_path in image_paths:
            if any(
                image_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
            ):
                image = Image(image_path)
            elif any(
                image_path.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]
            ):
                image = Video(image_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported media type: {image_path}",
                )
            prompt.append(image)
    if request.prompt is not None:
        prompt.append(request.prompt)
    # print(prompt)
    response = model.generate_content(prompt)
    print(colored(response, "cyan", attrs=["bold"]))

    return InferenceResponse(generated_text=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_api:app",  # Ensure this matches your filename
        host="0.0.0.0",
        port=args.port,  # Use the correct port
        reload=False,  # Keep reload False
    )
