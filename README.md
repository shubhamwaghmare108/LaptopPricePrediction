# Laptop Price Prediction — Three-Tier ML Application

This project predicts laptop prices in INR. Refactoring is isolated on the `refactor/three-tier-architecture` branch; `main` remains the baseline.

## Architecture

| Tier | Responsibility | Modules |
|---|---|---|
| Presentation | User interaction | `app.py`, `main.py`, `check_model.py` |
| Business / service | Prediction rules and training orchestration | `laptop_price/services/`, `laptop_price/pipeline/`, `laptop_price/components/`, `laptop_price/metrics.py` |
| Data access | Source data, artifact persistence and loading | `laptop_price/repositories/`, `laptop_price/config.py`, `laptop_price/utils.py` |

Streamlit delegates inference to `services/prediction_service.py`. The service validates the feature schema and uses the repository layer to load the model and preprocessor. Training components remain reusable business operations orchestrated by the training pipeline.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and configure MySQL if needed, or set `LAPTOP_DATA_CSV` to a local CSV containing `Price_INR`.

## Run

```bash
python main.py
streamlit run app.py
python -m pytest -q
python -m ruff check --select E4,E7,E9,F app.py laptop_price tests
```

## Artifact contract

Training creates `artifacts/transformed/train.csv`, `test.csv`, `preprocessor.joblib`, and `feature_list.json`. The model pusher publishes `prediction/models/current_model.joblib`. Train/test CSVs retain source feature values; the fitted preprocessor performs imputation, scaling, and encoding exactly once.

## Follow-up

The current trainer compares models using the test split. Introduce validation/cross-validation for model selection and reserve a separate holdout set for final unbiased evaluation. Never commit `.env` or credentials.
