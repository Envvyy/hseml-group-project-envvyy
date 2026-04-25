"""Machine learning models."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from typing import Any, Dict


def get_baseline_model() -> LinearRegression:
    """Get baseline Linear Regression model.
    
    Returns:
        Untrained Linear Regression model
    """
    return LinearRegression()


def get_models_dict(random_state: int = 42) -> Dict[str, Any]:
    """Get dictionary of models to train.
    
    Args:
        random_state: Random seed
        
    Returns:
        Dictionary with model names and instances
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
        ),
    }
    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    return metrics


def train_and_evaluate(model: Any, X_train, y_train, X_val, y_val) -> Dict[str, float]:
    """Train model and evaluate on validation set.
    
    Args:
        model: ML model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)
    return metrics
