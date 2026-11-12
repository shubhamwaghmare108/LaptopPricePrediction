"""Health and readiness endpoints."""
from fastapi import APIRouter
from backend.app.core.config import MODEL_PATH, PREPROCESSOR_PATH

router = APIRouter(tags=["health"])

@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_ready": MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file()}
