"""Streamlit frontend. It talks to FastAPI; it never loads model artifacts."""
import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("PREDICTION_API_URL", "http://localhost:8000").rstrip("/") + "/api/v1"
TIMEOUT = 60

def get_api(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def post_api(path: str, *, payload=None, files=None) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, files=files, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def show_error(error: Exception) -> None:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        st.error(f"API error ({error.response.status_code}): {detail}")
    else:
        st.error(f"Cannot reach prediction API at {API_URL}: {error}")

def main() -> None:
    st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")
    st.title("Laptop Price Predictor")
    st.caption("Streamlit frontend · FastAPI backend · Separate training pipeline")
    try:
        health = get_api("/health")
        if not health.get("model_ready"):
            st.warning("API is running, but serving artifacts are missing. Run the training pipeline and deploy its artifacts.")
            return
        schema = get_api("/predictions/schema")
    except requests.RequestException as error:
        show_error(error)
        st.info("Start the backend: uvicorn backend.app.main:app --reload --port 8000")
        return
    fields = schema.get("features", [])
    if not fields:
        st.error("The backend returned an empty feature schema.")
        return
    with st.sidebar:
        st.success("Prediction API ready")
        st.caption(API_URL)
        st.caption(f"{len(fields)} model features")
    st.subheader("Single laptop estimate")
    with st.form("single_prediction"):
        values = {}
        columns = st.columns(3)
        for index, field in enumerate(fields):
            with columns[index % 3]:
                if field["kind"] == "number":
                    values[field["name"]] = st.number_input(field["name"], value=float(field.get("default") or 0.0))
                elif field.get("options"):
                    values[field["name"]] = st.selectbox(field["name"], field["options"])
                else:
                    values[field["name"]] = st.text_input(field["name"])
        submitted = st.form_submit_button("Estimate price", type="primary")
    if submitted:
        try:
            response = post_api("/predictions", payload={"records": [values]})
            result = pd.DataFrame(response["predictions"])
            st.success(f"Estimated price: ₹{float(result.iloc[0]['predicted_price']):,.2f}")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download estimate", result.to_csv(index=False), "laptop_estimate.csv", "text/csv")
        except (requests.RequestException, KeyError, ValueError) as error:
            show_error(error)
    st.divider()
    st.subheader("Batch prediction")
    uploaded = st.file_uploader("Upload a CSV containing the model features", type=["csv"])
    if uploaded is not None:
        try:
            preview = pd.read_csv(uploaded)
            st.dataframe(preview.head(), use_container_width=True)
            uploaded.seek(0)
            if st.button("Run batch predictions"):
                result = post_api("/predictions/batch", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
                output = pd.DataFrame(result["predictions"])
                st.success(f"Predicted {len(output)} rows")
                st.dataframe(output.head(100), use_container_width=True)
                st.download_button("Download predictions", output.to_csv(index=False), "laptop_predictions.csv", "text/csv")
        except (requests.RequestException, pd.errors.ParserError, ValueError) as error:
            show_error(error)

if __name__ == "__main__":
    main()
