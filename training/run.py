"""CLI entry point for the offline training application."""
from laptop_price.pipeline.training_pipeline import run_pipeline

if __name__ == "__main__":
    print(run_pipeline())
