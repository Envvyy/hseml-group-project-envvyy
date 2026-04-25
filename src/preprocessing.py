"""Data preprocessing functions."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'mean', 'median')
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == "drop":
        df_clean = df_clean.dropna()
    elif strategy == "mean":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == "median":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame without duplicates
    """
    return df.drop_duplicates()


def remove_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr") -> pd.DataFrame:
    """Remove outliers from specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to check for outliers
        method: Method for outlier detection ('iqr' or 'zscore')
        
    Returns:
        DataFrame without outliers
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == "iqr":
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == "zscore":
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < 3]
    
    return df_clean


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation, and test sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        val_size: Proportion of validation set from remaining data
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
