from __future__ import annotations
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from ..model.inference import predict

app = FastAPI(title="AI Crop Disease Detector API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    result = predict(image)
    return JSONResponse(content=result)
