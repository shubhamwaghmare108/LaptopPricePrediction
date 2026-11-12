# Training application

Training is an offline workflow, independent of the HTTP inference service and Streamlit UI.

From the repository root:

```bash
python -m pip install -r requirements-training.txt
python main.py
```

The pipeline ingests data, validates/transforms it, trains and evaluates candidate models, then publishes the active model to `prediction/models/current_model.joblib`. The serving preprocessor and feature metadata are written under `artifacts/transformed/`. Deploy the model, preprocessor, feature metadata, and training CSV together to the backend environment. Do not train inside the API process.
