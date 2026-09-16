# Deployment guide

## Streamlit Cloud

1. Deploy `app.py` from the repository root.
2. Use Python 3.11.
3. Ensure these serving artifacts are present in the deployed revision:
   - `artifacts/transformed/preprocessor.joblib`
   - `artifacts/transformed/feature_list.json`
   - `prediction/models/current_model.joblib`
4. Do not run the training pipeline during Streamlit startup.
5. If artifacts are stored externally, download and validate them during a controlled build step before starting the app.

## Local validation

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py laptop_price tests
ruff check app.py laptop_price tests
pytest
```

## Release contract

A model release must contain a compatible model, preprocessor, and feature metadata generated from the same training run. The CI test suite checks that these files exist, can be loaded, and agree with the training schema.

The current CI workflow intentionally fails when required serving artifacts are absent. This prevents deploying an app that can start but cannot make predictions.
