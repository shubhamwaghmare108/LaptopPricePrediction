"""Prediction endpoints. Model execution is confined to this backend."""
import json
from functools import lru_cache
from io import BytesIO
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.app.core.config import FEATURE_LIST_PATH, PREPROCESSOR_PATH, TRAIN_CSV_PATH
from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
from laptop_price.exception import PricePredictorException
from laptop_price.services.prediction_service import predict_dataframe
from laptop_price.repositories.artifact_repository import load_object

router = APIRouter(prefix="/predictions", tags=["predictions"])

@lru_cache(maxsize=1)
def _schema() -> dict:
    if not FEATURE_LIST_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model metadata is unavailable. Run the training pipeline.")
    metadata = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN_CSV_PATH) if TRAIN_CSV_PATH.is_file() else pd.DataFrame()
    try:
        preprocessor = load_object(PREPROCESSOR_PATH)
        ordered = list(getattr(preprocessor, "feature_names_in_", metadata.get("num_cols", []) + metadata.get("cat_cols", [])))
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Preprocessor unavailable: {error}") from error
    numeric, fields = set(metadata.get("num_cols", [])), []
    for name in ordered:
        values, default = [], 0.0
        if name in train:
            if name in numeric:
                series = pd.to_numeric(train[name], errors="coerce").dropna()
                if not series.empty:
                    default = float(series.median())
            else:
                values = sorted(train[name].dropna().astype(str).unique().tolist())
                default = values[0] if values else ""
        fields.append({"name": name, "kind": "number" if name in numeric else "category", "default": default, "options": values})
    return {"features": fields, "target": "Price_INR"}

@router.get("/schema")
def get_schema() -> dict:
    return _schema()

@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_dataframe(pd.DataFrame(request.records))
        return PredictionResponse(predictions=result.to_dict(orient="records"))
    except PricePredictorException as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@router.post("/batch", response_model=PredictionResponse)
async def predict_batch(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    try:
        result = predict_dataframe(pd.read_csv(BytesIO(content)))
        return PredictionResponse(predictions=result.to_dict(orient="records"))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid UTF-8 CSV.") from error
    except PricePredictorException as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
