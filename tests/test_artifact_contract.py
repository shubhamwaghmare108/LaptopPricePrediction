"""Artifact contracts are tested with generated temporary artifacts, not local files."""
import joblib

def test_model_and_preprocessor_are_loadable(fitted_artifacts):
    _, _, model_path, transformer_path = fitted_artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(transformer_path)
    assert callable(getattr(model, "predict", None))
    assert callable(getattr(preprocessor, "transform", None))

def test_preprocessor_schema_matches_training_features(fitted_artifacts):
    features, _, _, transformer_path = fitted_artifacts
    preprocessor = joblib.load(transformer_path)
    assert list(preprocessor.feature_names_in_) == list(features.columns)
