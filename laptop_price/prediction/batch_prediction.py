"""Batch prediction adapter; delegates inference rules to prediction service."""
from pathlib import Path
from typing import Optional, Union
import joblib
import pandas as pd
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.services.prediction_service import predict_dataframe

logger = get_logger(__name__)

def _load_input(input_data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    path = Path(input_data)
    if not path.is_file():
        raise PricePredictorException(f"Input file not found: {path}")
    return pd.read_csv(path)

def batch_predict(input_data: Union[str, Path, pd.DataFrame], model_path: Optional[Union[str, Path]] = None, transformer_path: Optional[Union[str, Path]] = None, target_col: Optional[str] = "Price_INR", output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    try:
        frame = _load_input(input_data)
        if model_path is None and transformer_path is None:
            result = predict_dataframe(frame)
        elif model_path is not None and transformer_path is not None:
            model_file, transformer_file = Path(model_path), Path(transformer_path)
            if not model_file.is_file():
                raise PricePredictorException(f"Model file not found: {model_file}")
            if not transformer_file.is_file():
                raise PricePredictorException(f"Transformer file not found: {transformer_file}")
            result = predict_dataframe(frame, joblib.load(model_file), joblib.load(transformer_file))
        else:
            raise PricePredictorException("Provide both model_path and transformer_path, or neither")
        if target_col and target_col in result.columns and target_col != "Price_INR":
            result = result.drop(columns=[target_col])
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return result
    except PricePredictorException:
        raise
    except Exception as error:
        logger.exception("Batch prediction failed")
        raise PricePredictorException(f"Batch prediction failed: {error}") from error
