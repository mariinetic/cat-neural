import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import tensorflow as tf

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "catneural")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "catneural.keras")
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="CatNeural API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[DB_NAME]
analyses = db["analyses"]

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Execute backend/model/train.py primeiro."
        )

    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    # O modelo retorna a probabilidade da classe cat.
    probability = float(model.predict(array, verbose=0)[0][0])
    return probability


@app.get("/api/gallery")
def gallery(limit: int = 24):
    limit = max(1, min(limit, 100))
    rows = []
    if DATASET_DIR.exists():
        for image_path in sorted(DATASET_DIR.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS or len(rows) >= limit:
                continue
            relative = image_path.relative_to(DATASET_DIR).as_posix()
            try:
                with Image.open(image_path) as opened_image:
                    image_width, image_height = opened_image.size
            except (OSError, ValueError):
                image_width, image_height = 1, 1
            landmarks = []
            annotation_path = image_path.with_suffix(image_path.suffix + ".cat")
            if not annotation_path.exists():
                annotation_path = image_path.with_suffix(".cat")
            try:
                if annotation_path.exists():
                    values = [float(value) for value in annotation_path.read_text().split()]
                    count = int(values[0]) if values else 0
                    landmarks = [[values[i], values[i + 1]] for i in range(1, min(len(values) - 1, 1 + count * 2), 2)]
            except (ValueError, OSError):
                landmarks = []
            rows.append({"id": relative, "filename": image_path.name, "image_url": f"/api/gallery/image/{quote(relative)}", "landmarks": landmarks, "width": image_width, "height": image_height})
    return rows


@app.get("/api/gallery/image/{relative_path:path}")
def gallery_image(relative_path: str):
    requested = (DATASET_DIR / unquote(relative_path)).resolve()
    if DATASET_DIR.resolve() not in requested.parents or requested.suffix.lower() not in IMAGE_EXTENSIONS or not requested.is_file():
        raise HTTPException(status_code=404, detail="Imagem não encontrada na galeria.")
    return FileResponse(requested)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "database": DB_NAME,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem válida.")

    raw = await file.read()

    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="A imagem deve ter no máximo 10 MB.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem.")

    start = time.perf_counter()
    cat_probability = predict(image)
    elapsed = round(time.perf_counter() - start, 3)

    confidence = round(cat_probability * 100, 2)
    is_cat = cat_probability >= 0.5

    document = {
        "filename": file.filename,
        "prediction": "cat" if is_cat else "not_cat",
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc),
        "processing_time": elapsed,
    }

    result = analyses.insert_one(document)

    return {
        "id": str(result.inserted_id),
        "filename": file.filename,
        "prediction": document["prediction"],
        "confidence": confidence,
        "processing_time": elapsed,
        "message": "É um gatinho!" if is_cat else "Parece que não é um gatinho.",
    }


@app.get("/api/history")
def history(limit: int = 10):
    limit = max(1, min(limit, 50))
    rows = list(
        analyses.find(
            {},
            {"filename": 1, "prediction": 1, "confidence": 1,
             "created_at": 1, "processing_time": 1}
        ).sort("created_at", -1).limit(limit)
    )

    for row in rows:
        row["_id"] = str(row["_id"])
        if isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()

    return rows


@app.get("/api/stats")
def stats():
    total = analyses.count_documents({})
    cats = analyses.count_documents({"prediction": "cat"})

    pipeline = [
        {"$group": {"_id": None, "average": {"$avg": "$confidence"}}}
    ]
    result = list(analyses.aggregate(pipeline))
    average = round(float(result[0]["average"]), 2) if result else 0

    return {
        "total": total,
        "cats_detected": cats,
        "average_confidence": average,
    }
