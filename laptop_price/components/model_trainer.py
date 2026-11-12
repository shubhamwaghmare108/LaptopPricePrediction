from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from laptop_price.entity.artifact_entity import ModelTrainerArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.metrics import calculate_regression_metrics
from laptop_price.utils import load_object, save_object
logger = get_logger(__name__)
def _evaluate(y_true, y_pred) -> dict:
    return {key.lower(): value for key, value in calculate_regression_metrics(y_true, y_pred).items()}
def train_model(transformed_train_csv: str = "artifacts/transformed/train.csv", transformed_test_csv: str = "artifacts/transformed/test.csv", transformer_path: str = "artifacts/transformed/preprocessor.joblib", target_col: str = "Price_INR", model_output_path: str = "artifacts/model/best_model.joblib") -> ModelTrainerArtifact:
    try:
        train_df, test_df = pd.read_csv(transformed_train_csv), pd.read_csv(transformed_test_csv)
        if target_col not in train_df or target_col not in test_df:
            raise PricePredictorException(f"Target column '{target_col}' missing from train/test data")
        X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
        X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
        preprocessor = load_object(Path(transformer_path))
        X_train_t, X_test_t = preprocessor.transform(X_train), preprocessor.transform(X_test)
        candidates = {"LinearRegression": LinearRegression(), "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)}
        evaluated = {}
        for name, estimator in candidates.items():
            estimator.fit(X_train_t, y_train)
            evaluated[name] = (estimator, _evaluate(y_test, estimator.predict(X_test_t)))
            logger.info("%s metrics: %s", name, evaluated[name][1])
        chosen, (best_model, best_metrics) = min(evaluated.items(), key=lambda item: item[1][1]["rmse"])
        model_path = Path(model_output_path)
        save_object(best_model, model_path)
        logger.info("Selected %s (RMSE %.4f)", chosen, best_metrics["rmse"])
        return ModelTrainerArtifact(model_path, float(best_model.score(X_train_t, y_train)), float(best_model.score(X_test_t, y_test)))
    except Exception as error:
        logger.exception("Model training failed")
        raise PricePredictorException(f"Model training failed: {error}") from error
