"""Business logic for validating and producing laptop price predictions."""
import pandas as pd
from laptop_price.exception import PricePredictorException
from laptop_price.repositories.artifact_repository import load_prediction_artifacts
TARGET_COLUMN = "Price_INR"
PREDICTION_COLUMN = "predicted_Price_INR"
def predict_dataframe(data: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    if data is None or data.empty:
        raise PricePredictorException("Prediction input is empty")
    if model is None or preprocessor is None:
        model, preprocessor = load_prediction_artifacts()
    features = data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    expected = list(getattr(preprocessor, "feature_names_in_", features.columns))
    missing = [column for column in expected if column not in features.columns]
    extra = [column for column in features.columns if column not in expected]
    if missing or extra:
        raise PricePredictorException(f"Feature schema mismatch. Missing: {missing}; unexpected: {extra}")
    try:
        predictions = model.predict(preprocessor.transform(features[expected]))
    except Exception as error:
        raise PricePredictorException(f"Prediction failed: {error}") from error
    result = data.copy()
    result[PREDICTION_COLUMN] = predictions
    return result
