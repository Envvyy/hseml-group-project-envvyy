"""Data loading utilities."""

import pandas as pd
from pathlib import Path


def load_twitch_data(filepath: str | Path) -> pd.DataFrame:
    """Load Twitch dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with Twitch data
    """
    df = pd.read_csv(filepath)
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }
    return info
