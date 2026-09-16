import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from laptop_price.config import TRANSFORMED_DATA_DIR
from laptop_price.entity.artifact_entity import DataTransformationArtifact
from laptop_price.entity.config_entity import DataTransformationConfig
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.utils import save_df, save_object

logger = get_logger(__name__)


def transform(raw_path: Path, target_col: str = "Price_INR") -> DataTransformationArtifact:
    """Fit the feature preprocessor on training data and persist all outputs."""
    try:
        df = pd.read_csv(raw_path).drop_duplicates().reset_index(drop=True)

        for column in ("SKU", "Model"):
            if column in df.columns:
                df = df.drop(columns=[column])
                logger.info("Dropped identifier column: %s", column)

        if target_col not in df.columns:
            raise PricePredictorException(
                f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
            )

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", encoder),
            ]
        )
        preprocessor = ColumnTransformer(
            [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)],
            remainder="drop",
        )

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        preprocessor.fit(X_train)

        output_dir = Path(TRANSFORMED_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_path = output_dir / "preprocessor.joblib"
        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"
        feature_path = output_dir / "feature_list.json"

        save_object(preprocessor, preprocessor_path)

        # Persist readable, consistently imputed datasets for downstream training.
        X_train_out = X_train.copy()
        X_test_out = X_test.copy()
        for column in num_cols:
            median = X_train_out[column].median()
            X_train_out[column] = X_train_out[column].fillna(median)
            X_test_out[column] = X_test_out[column].fillna(median)
        for column in cat_cols:
            X_train_out[column] = X_train_out[column].fillna("missing")
            X_test_out[column] = X_test_out[column].fillna("missing")

        X_train_out[target_col] = train_df[target_col].to_numpy()
        X_test_out[target_col] = test_df[target_col].to_numpy()
        X_train_out.to_csv(train_path, index=False)
        X_test_out.to_csv(test_path, index=False)

        with feature_path.open("w", encoding="utf-8") as handle:
            json.dump({"num_cols": num_cols, "cat_cols": cat_cols}, handle, indent=2)

        logger.info("Saved transformation artifacts under %s", output_dir)
        return DataTransformationArtifact(
            transformed_path=output_dir / "transformed.npz",
            transformer_object_path=preprocessor_path,
        )
    except Exception as error:
        logger.exception("Data transformation failed")
        raise PricePredictorException(f"Data transformation failed: {error}") from error
