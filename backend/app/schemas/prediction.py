"""Transport schemas for inference requests and responses."""
from typing import Any
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)

class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]
