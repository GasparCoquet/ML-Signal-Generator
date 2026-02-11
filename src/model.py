"""
Machine Learning Model Training Module

This module contains functions to train ML models for predicting next-day return direction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb


def time_series_split(
    data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets (time series aware).
    
    Args:
        data: DataFrame with features and target
        train_size: Proportion of data for training (default: 0.7)
        val_size: Proportion of data for validation (default: 0.15)
    
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    n = len(data)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    
    return train, val, test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
    
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    metrics = {
        'auc': auc,
        'train_accuracy': model.score(X_train, y_train),
        'val_accuracy': model.score(X_val, y_val)
    }
    
    return model, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
    
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='auc'
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    metrics = {
        'auc': auc,
        'train_accuracy': model.score(X_train, y_train),
        'val_accuracy': model.score(X_val, y_val)
    }
    
    return model, metrics


def train_walk_forward(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> float:
    """
    Walk-Forward Validation (expanding window) using TimeSeriesSplit.

    Fits a clone of the model on each train fold and evaluates on the next chunk.
    Returns the mean validation AUC across folds. The input model is not modified.

    Args:
        model: Estimator with fit and predict_proba (e.g. RandomForestClassifier, XGBClassifier).
        X: Feature DataFrame (time-ordered).
        y: Target Series (time-ordered).
        n_splits: Number of splits for TimeSeriesSplit.

    Returns:
        Mean ROC AUC across validation folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
    return float(np.mean(scores))


def get_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model (RandomForest or XGBoost)
        feature_names: List of feature names
    
    Returns:
        DataFrame with features and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = model.score(X_test, y_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'auc': auc,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results

