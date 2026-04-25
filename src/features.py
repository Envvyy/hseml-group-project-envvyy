"""Feature engineering functions."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engagement-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new engagement features
    """
    df_new = df.copy()
    
    # Average viewers per stream
    if "Watch time(Minutes)" in df_new.columns and "Stream time(minutes)" in df_new.columns:
        df_new["avg_viewers"] = df_new["Watch time(Minutes)"] / (df_new["Stream time(minutes)"] + 1)
    
    # Follower to viewer ratio
    if "Followers" in df_new.columns and "Average viewers" in df_new.columns:
        df_new["follower_viewer_ratio"] = df_new["Followers"] / (df_new["Average viewers"] + 1)
    
    # Peak to average ratio
    if "Peak viewers" in df_new.columns and "Average viewers" in df_new.columns:
        df_new["peak_avg_ratio"] = df_new["Peak viewers"] / (df_new["Average viewers"] + 1)
    
    return df_new


def encode_categorical_features(
    df: pd.DataFrame, categorical_columns: list[str]
) -> Tuple[pd.DataFrame, dict]:
    """Encode categorical features using Label Encoding.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Tuple of (scaled X_train, X_val, X_test, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
