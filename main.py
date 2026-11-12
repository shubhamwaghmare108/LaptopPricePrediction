"""Backward-compatible entry point for the separate training application."""
from training.run import run_pipeline

if __name__ == "__main__":
    print(run_pipeline())
