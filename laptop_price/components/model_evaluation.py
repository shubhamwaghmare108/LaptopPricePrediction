import json
from pathlib import Path
import pandas as pd
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.metrics import calculate_regression_metrics
from laptop_price.utils import load_object
logger = get_logger(__name__)
def evaluate_model(model_path: str = "artifacts/model/best_model.joblib", test_csv: str = "artifacts/transformed/test.csv", transformer_path: str = "artifacts/transformed/preprocessor.joblib", target_col: str = "Price_INR", metrics_output_path: str = "artifacts/model/model_metrics.json") -> dict:
    try:
        model, preprocessor = load_object(Path(model_path)), load_object(Path(transformer_path))
        test_df = pd.read_csv(test_csv)
        if target_col not in test_df:
            raise PricePredictorException(f"Target column '{target_col}' missing from test data")
        X_test = test_df.drop(columns=[target_col])
        metrics = calculate_regression_metrics(test_df[target_col], model.predict(preprocessor.transform(X_test)))
        output = Path(metrics_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
        return metrics
    except Exception as error:
        logger.exception("Model evaluation failed")
        raise PricePredictorException(f"Model evaluation failed: {error}") from error
