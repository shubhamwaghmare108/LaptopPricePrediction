# Laptop Price Prediction

An end-to-end laptop price prediction project with **three independently runnable applications**: an offline training application, a FastAPI inference backend, and a Streamlit frontend. The UI never loads model artifacts or performs inference; predictions always travel over HTTP to FastAPI.

## Architecture

```text
Offline training job                         Online serving
Dataset -> Training pipeline -> model bundle
                                  | deploy    User
                                  v             |
                         FastAPI inference <--- Streamlit UI
                                  |
                         Prediction service
                                  |
                         Artifact repository
                                  |
                         Model + preprocessor
```

| Application | Responsibility | Entry point |
|---|---|---|
| Training | Ingestion, validation, transformation, model training/evaluation, artifact publication | `training/run.py` (also `python main.py`) |
| Backend | HTTP contracts, readiness/schema endpoints, prediction and batch prediction | `backend/app/main.py` |
| Frontend | Forms, CSV upload, result display and downloads | `frontend/streamlit_app.py` |
| ML core | Reusable training and prediction business logic | `laptop_price/` |

## Local setup

Use Python 3.11 (recommended) and create a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1. Train independently

```bash
python -m pip install -r requirements-training.txt
# Configure .env / MySQL, or set LAPTOP_DATA_CSV to a local CSV
python -m training.run
# or: python main.py
```

The pipeline publishes the active model to `prediction/models/current_model.joblib` and the preprocessor/schema to `artifacts/transformed/`. Deploy these files from the same training run as one compatible bundle:

- `prediction/models/current_model.joblib`
- `artifacts/transformed/preprocessor.joblib`
- `artifacts/transformed/feature_list.json`
- `artifacts/transformed/train.csv` (for form defaults and category choices)

### 2. Run the FastAPI backend

```bash
python -m pip install -r requirements-api.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

- `GET /api/v1/health` — API and artifact readiness
- `GET /api/v1/predictions/schema` — ordered feature schema and UI defaults
- `POST /api/v1/predictions` — JSON single/multi-record prediction
- `POST /api/v1/predictions/batch` — CSV upload prediction

JSON request shape (use the exact feature set from the schema endpoint):

```json
{"records": [{"Brand": "Dell", "RAM_GB": 16}]}
```

### 3. Run the Streamlit frontend

In a separate terminal:

```bash
python -m pip install -r requirements-frontend.txt
# PowerShell: $env:PREDICTION_API_URL = "http://127.0.0.1:8000"
# macOS/Linux: export PREDICTION_API_URL=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. The frontend only uses HTTP requests to FastAPI; it does not import `laptop_price`, read local artifacts, connect to MySQL, or run model inference.

## Docker Compose

After training artifacts have been generated:

```bash
docker compose up --build
```

Frontend: http://localhost:8501 · API docs: http://localhost:8000/docs. Compose mounts artifacts read-only into the backend. In production, use an immutable model bundle in an artifact store and deploy a pinned model/preprocessor pair.

## Separation rules

- Training runs manually, in CI, or as a scheduled job; never during API startup or prediction requests.
- FastAPI owns model loading, schema enforcement, single prediction and batch prediction.
- Streamlit owns presentation only and calls the backend through `PREDICTION_API_URL`.
- Configure secrets with environment variables or a secret manager. Do not commit `.env` or credentials.

## Checks

```bash
python -m compileall -q app.py main.py backend frontend training laptop_price
python -m ruff check --select E4,E7,E9,F app.py main.py backend frontend training laptop_price tests
python -m pytest -q
```

## Known ML follow-up

The current trainer still compares candidate models using the test split. Introduce validation/cross-validation for selection and reserve a final untouched holdout for unbiased evaluation.
