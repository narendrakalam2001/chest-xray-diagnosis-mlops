# ============================================================
# CHEST X-RAY DIAGNOSIS API — FastAPI Serving
# ============================================================

import os
import io
import json
import time
import base64
import logging
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from src.model_loader import load_latest_model
from services.prediction_service import predict_image, predict_image_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chest X-Ray Diagnosis API",
    description=(
        "Multi-label chest X-ray pathology detection using NIH ChestX-ray14. "
        "Detects 14 conditions: Atelectasis, Cardiomegaly, Pneumonia, and more. "
        "Intended as decision support for radiologists — NOT for autonomous diagnosis."
    ),
    version="1.0.0"
)

# ── Load model on startup ─────────────────────────────────────
try:
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, thresholds, backbone = load_latest_model(device=device)
    logger.info("Model loaded successfully  |  backbone=%s  device=%s", backbone, device)
    MODEL_LOADED = True
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model        = None
    thresholds   = None
    backbone     = None
    device       = torch.device("cpu")
    MODEL_LOADED = False


# ============================================================
# INPUT SCHEMA
# ============================================================

class Base64ImageInput(BaseModel):
    """
    Accept X-ray image as base64-encoded string.
    Useful for web/mobile clients that can't do multipart upload.
    """
    image_base64:    str
    patient_id:      Optional[str] = None
    view_position:   Optional[str] = "PA"    # PA or AP
    return_gradcam:  bool = False


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message":  "Chest X-Ray Diagnosis API is live 🫁",
        "docs":     "/docs",
        "health":   "/health",
        "model_info": "/model_info",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": MODEL_LOADED,
        "device":       str(device),
        "backbone":     backbone or "not loaded",
    }


@app.get("/model_info")
def model_info():
    registry_path = "xray_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)

        # Also attach model card if available
        card_path = registry.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            registry["model_card"] = card

        return registry
    return {"error": "Model registry not found. Run train_model.py first."}


@app.get("/labels")
def get_labels():
    """Return list of 14 pathology labels and clinical thresholds."""
    from src.config import PATHOLOGY_LABELS, CLINICAL_THRESHOLDS
    return {
        "pathology_labels": PATHOLOGY_LABELS,
        "n_classes":        len(PATHOLOGY_LABELS),
        "clinical_thresholds": CLINICAL_THRESHOLDS,
    }


# ── File upload prediction ────────────────────────────────────

@app.post("/predict")
async def predict_upload(
    file:          UploadFile = File(...),
    return_gradcam: bool      = False,
):
    """
    Accept chest X-ray image file (PNG/JPG) and return
    multi-label diagnosis probabilities with risk assessment.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # ── Validate file type ────────────────────────────────────
    if not file.content_type in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PNG or JPEG."
        )

    start = time.time()

    try:
        image_bytes = await file.read()
        result = predict_image(
            model=model, image_bytes=image_bytes,
            thresholds=thresholds, device=device,
            return_gradcam=return_gradcam,
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # ── Log prediction ────────────────────────────────────────
    _log_prediction(file.filename, result)

    result["latency_seconds"] = round(time.time() - start, 4)
    result["filename"]        = file.filename

    return result


@app.post("/predict_base64")
async def predict_base64(payload: Base64ImageInput):
    """
    Accept X-ray image as base64-encoded string.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.time()

    try:
        result = predict_image_base64(
            model=model, image_b64=payload.image_base64,
            thresholds=thresholds, device=device,
            return_gradcam=payload.return_gradcam,
        )
    except Exception as e:
        logger.error("Base64 prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    if payload.patient_id:
        result["patient_id"] = payload.patient_id
    result["view_position"]   = payload.view_position
    result["latency_seconds"] = round(time.time() - start, 4)

    _log_prediction(payload.patient_id or "unknown", result)
    return result


# ============================================================
# PREDICTION LOGGING
# ============================================================

def _log_prediction(identifier: str, result: dict) -> None:
    """Append prediction to CSV log for monitoring."""
    import pandas as pd

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    findings  = result.get("findings",   [])
    mean_prob = result.get("mean_probability", 0)

    log_record = {
        "timestamp":       time.time(),
        "identifier":      identifier,
        "n_findings":      len(findings),
        "mean_probability": mean_prob,
        "critical_finding": result.get("critical_finding", False),
        "uncertainty_flag": result.get("uncertainty_flag",  False),
        "findings_str":    "|".join([f["label"] for f in findings]),
    }

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
