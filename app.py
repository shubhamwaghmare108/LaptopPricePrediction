from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#17212b; --muted:#66727e; --paper:#f7f8f5; --panel:#fff; --line:#e3e8e4; --teal:#137a72; --teal-dark:#0c5c58; --gold:#e8aa45; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:rgba(247,248,245,.88); }
    [data-testid="stSidebar"] { background:#eef3f0; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:2rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:0; color:var(--ink); }
    h1 { font-size:clamp(2.2rem,5vw,4.5rem); line-height:.98; margin:0; }
    h2 { font-size:1.45rem; margin-top:.4rem; }
    .block-container { max-width:1180px; padding:3.5rem 3rem 4rem; }
    .hero { padding:1.25rem 0 2.5rem; }
    .eyebrow,.section-kicker { color:var(--teal); font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .eyebrow { margin-bottom:.9rem; }
    .hero-copy { color:var(--muted); font-size:1.05rem; max-width:620px; line-height:1.6; margin-top:1.1rem; }
    .hero-rule { width:76px; height:5px; background:var(--gold); border-radius:3px; margin-top:1.5rem; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1rem 1.15rem; min-height:92px; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; margin-top:.35rem; }
    .section-gap { height:2.75rem; }
    .section-copy { color:var(--muted); margin:-.45rem 0 1.2rem; }
    .stForm,[data-testid="stFileUploader"] { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1.2rem; }
    .stButton > button,.stFormSubmitButton > button { border-radius:8px; border:1px solid var(--teal); background:var(--teal); color:white; font-weight:700; padding:.55rem 1.2rem; }
    .stButton > button:hover,.stFormSubmitButton > button:hover { background:var(--teal-dark); border-color:var(--teal-dark); color:white; }
    .stDownloadButton > button { border-radius:8px; border:1px solid var(--line); font-weight:600; }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    hr { border-color:var(--line); margin:2.2rem 0; }
    </style>
    """, unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "best_model.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

@st.cache_resource
def load_artifacts():
    required = {"preprocessor": PREPROCESSOR_PATH, "model": MODEL_PATH, "feature metadata": FEATURE_LIST_PATH}
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing serving artifacts: " + "; ".join(missing))
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    with FEATURE_LIST_PATH.open("r", encoding="utf-8") as file:
        info = json.load(file)
    num_cols = info.get("num_cols", [])
    cat_cols = info.get("cat_cols", [])
    features = num_cols + cat_cols
    if not features:
        raise ValueError("Feature metadata contains no input columns.")
    return preprocessor, model, features, num_cols, cat_cols

@st.cache_data
def load_training_data():
    return pd.read_csv(TRAIN_CSV_PATH) if TRAIN_CSV_PATH.is_file() else pd.DataFrame()

def load_training_unique_values(training_df, categorical_columns):
    result = {}
    for column in categorical_columns:
        if column in training_df.columns:
            result[column] = training_df[column].dropna().astype(str).value_counts().index.tolist()
    return result

def predict_df(df_input, preprocessor, model, features):
    df = df_input.drop(columns=["Price_INR"], errors="ignore").copy()
    missing = [column for column in features if column not in df.columns]
    extra = [column for column in df.columns if column not in features]
    if missing or extra:
        raise ValueError(f"Column mismatch. Missing: {missing}; unexpected: {extra}")
    df = df[features]
    predictions = np.asarray(model.predict(preprocessor.transform(df)), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Model returned a non-finite prediction.")
    result = df.copy()
    result["predicted_Price_INR"] = predictions
    return result

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

try:
    preprocessor, model, features, num_cols, cat_cols = load_artifacts()
except Exception as error:
    st.error(f"Error loading artifacts: {error}")
    st.stop()

training_df = load_training_data()
training_uniques = load_training_unique_values(training_df, cat_cols)

with st.sidebar:
    st.markdown("## Model desk")
    st.caption("Serving configuration")
    st.success("Active model ready", icon=":material/check_circle:")
    st.markdown("**Model**")
    st.code(MODEL_PATH.name)
    st.markdown("**Preprocessor**")
    st.code(PREPROCESSOR_PATH.name)
    st.caption(f"{len(features)} inputs | {len(num_cols)} numeric | {len(cat_cols)} categorical")
    with st.expander("View input schema"):
        st.write(features)

st.markdown("""<div class="hero"><div class="eyebrow">Laptop intelligence / price lab</div><h1>Find the right price<br>for every machine.</h1><div class="hero-copy">Build a laptop profile from the details that matter, then get a model-backed price estimate in seconds.</div><div class="hero-rule"></div></div>""", unsafe_allow_html=True)

metric_columns = st.columns(3)
for column, label, value in zip(metric_columns, ["Model status", "Input profile", "Workflow"], ["Ready to predict", f"{len(features)} signals", "Single + batch"]):
    with column:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Quick estimate</div><h2>Describe your laptop</h2><div class="section-copy">Choose the closest specifications to generate a price estimate.</div>', unsafe_allow_html=True)

with st.form("single_prediction_form"):
    input_values = {}
    left, right = st.columns(2)
    for index, column in enumerate(features):
        target = left if index % 2 == 0 else right
        if column in num_cols:
            default = float(training_df[column].median()) if column in training_df.columns and not training_df[column].dropna().empty else 0.0
            input_values[column] = target.number_input(column, value=default, key=f"num_{column}")
        else:
            options = training_uniques.get(column, [])
            input_values[column] = target.selectbox(column, options, key=f"cat_{column}") if options else target.text_input(column, key=f"text_{column}")
    submitted = st.form_submit_button("Estimate price")

if submitted:
    try:
        result = predict_df(pd.DataFrame([input_values]), preprocessor, model, features)
        st.dataframe(result)
        st.download_button("Download estimate CSV", data=to_csv_bytes(result), file_name="single_prediction.csv", mime="text/csv")
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.markdown('<div class="section-kicker">02 / Batch desk</div><h2>Price a full shortlist</h2><div class="section-copy">Upload a CSV with the same feature columns to score multiple laptops at once.</div>', unsafe_allow_html=True)
with st.expander("How to prepare your CSV"):
    guide_left, guide_right = st.columns(2)
    with guide_left:
        st.markdown("**Before you upload**")
        st.markdown("1. Keep one laptop per row.\n2. Use the exact column names shown below.\n3. Keep numeric values in numeric columns.\n4. Save the file as `.csv`.")
        st.caption("Price_INR is optional and will be ignored if included.")
    with guide_right:
        st.markdown("**Required columns**")
        st.code(", ".join(features), language="text")
        if not training_df.empty:
            st.download_button("Download CSV template", data=training_df[features].head(1).to_csv(index=False).encode("utf-8"), file_name="laptop_prediction_template.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload a CSV containing the model features", type="csv")
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())
        if st.button("Run batch predictions", type="primary"):
            result = predict_df(batch_df, preprocessor, model, features)
            st.success("Prediction finished.")
            st.dataframe(result.head())
            st.download_button("Download predictions CSV", data=to_csv_bytes(result), file_name="batch_predictions.csv", mime="text/csv")
    except Exception as error:
        st.error(f"Batch prediction failed: {error}")

st.caption("Streamlit app for the laptop-price end-to-end pipeline")
