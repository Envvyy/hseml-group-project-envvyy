"""Utility functions for the project."""

import random
import numpy as np
import pandas as pd
from typing import Any


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def save_experiment_results(
    experiment_name: str,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    filepath: str,
) -> None:
    """Save experiment results to CSV file.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        metrics: Dictionary with metric names and values
        params: Dictionary with model parameters
        filepath: Path to save results
    """
    results = {
        "experiment": experiment_name,
        "model": model_name,
        **metrics,
        "params": str(params),
    }
    
    df = pd.DataFrame([results])
    
    try:
        existing_df = pd.read_csv(filepath)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv(filepath, index=False)
