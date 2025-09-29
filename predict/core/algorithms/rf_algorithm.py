"""
Random Forest Algorithm
Ensemble learning method suitable for classification tasks
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
import time
from .base_algorithm import BaseAlgorithm

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    RF_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed, Random Forest algorithm unavailable")
    RF_AVAILABLE = False


class RandomForestAlgorithm(BaseAlgorithm):
    def __init__(self, n_estimators: int = 500, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', use_grid_search: bool = False):
        """Initialize Random Forest algorithm"""
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.use_grid_search = use_grid_search
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'RandomForestAlgorithm':
        """Train Random Forest model"""
        if not RF_AVAILABLE:
            raise ImportError("sklearn not installed, cannot use Random Forest algorithm")
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        
        input_shape = X_train_processed.shape[1]
        
        if input_shape > 100:
            max_features = 'log2'
        else:
            max_features = self.max_features
        
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        if self.use_grid_search:
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            self.model = GridSearchCV(
                rf, 
                param_grid, 
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
        else:
            self.model = rf
        
        self.model.fit(X_train_processed, y_train_processed)
        self.is_trained = True
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_test_processed, _ = self.prepare_data(X_test)
        
        return self.model.predict(X_test_processed)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        try:
            if hasattr(self.model, 'best_estimator_'):
                feature_importance = self.model.best_estimator_.feature_importances_
            else:
                feature_importance = self.model.feature_importances_
            
            return {f'feature_{i}': float(importance) for i, importance in enumerate(feature_importance)}
        except:
            return {} 