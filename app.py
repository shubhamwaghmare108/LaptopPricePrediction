"""Presentation tier: Streamlit delegates prediction to the service layer."""
import json
import pandas as pd
import streamlit as st
from laptop_price.exception import PricePredictorException
from laptop_price.repositories.artifact_repository import FEATURE_LIST_PATH, MODEL_PATH, PREPROCESSOR_PATH, TRAIN_CSV_PATH
from laptop_price.services.prediction_service import predict_dataframe
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("Laptop Price Predictor")
st.caption("Three-tier application: Presentation → Services → Repositories")
def load_feature_metadata():
    if FEATURE_LIST_PATH.is_file():
        metadata = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))
        return metadata.get("num_cols", []) + metadata.get("cat_cols", []), metadata.get("num_cols", []), metadata.get("cat_cols", [])
    if TRAIN_CSV_PATH.is_file():
        frame = pd.read_csv(TRAIN_CSV_PATH, nrows=5)
        features = [column for column in frame.columns if column != "Price_INR"]
        numeric = frame[features].select_dtypes(include="number").columns.tolist()
        return features, numeric, [column for column in features if column not in numeric]
    return [], [], []
features, numeric_columns, categorical_columns = load_feature_metadata()
if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
    st.error("Serving artifacts are missing. Run python main.py before starting the app.")
    st.stop()
st.sidebar.success("Serving artifacts found")
st.sidebar.caption(f"Model: {MODEL_PATH.name}")
st.sidebar.caption(f"Preprocessor: {PREPROCESSOR_PATH.name}")
if not features:
    st.warning("Feature metadata is missing. Run the training pipeline first.")
    st.stop()
training = pd.read_csv(TRAIN_CSV_PATH) if TRAIN_CSV_PATH.is_file() else pd.DataFrame()
st.subheader("Single laptop estimate")
with st.form("single_prediction"):
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(features):
        with columns[index % 3]:
            if feature in numeric_columns:
                default = float(training[feature].median()) if feature in training and not training[feature].dropna().empty else 0.0
                values[feature] = st.number_input(feature, value=default)
            else:
                choices = sorted(training[feature].dropna().astype(str).unique().tolist()) if feature in training else []
                values[feature] = st.selectbox(feature, choices) if choices else st.text_input(feature)
    submitted = st.form_submit_button("Estimate price")
if submitted:
    try:
        result = predict_dataframe(pd.DataFrame([values]))
        st.dataframe(result, use_container_width=True)
        st.download_button("Download estimate", result.to_csv(index=False), "laptop_estimate.csv", "text/csv")
    except PricePredictorException as error:
        st.error(str(error))
st.divider()
st.subheader("Batch prediction")
upload = st.file_uploader("Upload a feature CSV", type=["csv"])
if upload is not None:
    try:
        batch = pd.read_csv(upload)
        st.dataframe(batch.head(), use_container_width=True)
        if st.button("Run batch predictions"):
            result = predict_dataframe(batch)
            st.success(f"Predicted {len(result)} rows")
            st.dataframe(result.head(), use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False), "laptop_predictions.csv", "text/csv")
    except (ValueError, PricePredictorException) as error:
        st.error(f"Batch prediction failed: {error}")
