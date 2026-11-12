import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from laptop_price.entity.artifact_entity import DataTransformationArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.utils import save_object
logger = get_logger(__name__)
def transform(raw_path: Path, target_col: str = "Price_INR") -> DataTransformationArtifact:
    try:
        df = pd.read_csv(raw_path).drop_duplicates().reset_index(drop=True)
        identifiers = [column for column in ("SKU", "Model") if column in df.columns]
        if identifiers:
            df = df.drop(columns=identifiers)
        if target_col not in df.columns:
            raise PricePredictorException(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")
        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
        cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target_col]
        if not num_cols and not cat_cols:
            raise PricePredictorException("No usable feature columns found")
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocessor = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols), ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", encoder)]), cat_cols)], remainder="drop")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        preprocessor.fit(train_df.drop(columns=[target_col]))
        output_dir = Path("artifacts/transformed")
        output_dir.mkdir(parents=True, exist_ok=True)
        transformer_path = output_dir / "preprocessor.joblib"
        save_object(preprocessor, transformer_path)
        # Keep source feature values; the fitted pipeline owns imputation/scaling/encoding.
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        (output_dir / "feature_list.json").write_text(json.dumps({"num_cols": num_cols, "cat_cols": cat_cols}, indent=2), encoding="utf-8")
        return DataTransformationArtifact(transformer_object_path=transformer_path)
    except Exception as error:
        logger.exception("Data transformation failed")
        raise PricePredictorException(f"Data transformation failed: {error}") from error
