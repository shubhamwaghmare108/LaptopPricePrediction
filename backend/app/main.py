"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import CORS_ORIGINS
from backend.app.api.routes import health, predictions

app = FastAPI(title="Laptop Price Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])
app.include_router(health.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")

@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {"name": "Laptop Price Prediction API", "docs": "/docs", "health": "/api/v1/health"}
