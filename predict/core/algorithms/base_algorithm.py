"""
Base algorithm class
Defines common interface for all algorithms
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
import time
import warnings
warnings.filterwarnings('ignore')


class BaseAlgorithm(ABC):
    def __init__(self, name: str):
        """Initialize algorithm"""
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_time = 0
        self.prediction_time = 0
        self.scaler = None
        
    def _ensure_standardized(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, force: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Ensure data is standardized"""
        if self.scaler is None or force:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = X_train.copy()
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseAlgorithm':
        """Train the algorithm"""
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def fit_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train algorithm and evaluate performance"""
        print(f"🔄 Training {self.name}...")
        
        start_time = time.time()
        self.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = self.predict(X_test)
        self.prediction_time = time.time() - start_time
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        result = {
            'algorithm': self.name,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            **metrics
        }
        
        print(f"   ✅ {self.name} completed")
        print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📊 F1 Score: {metrics['f1_score']:.4f}")
        print(f"   ⏱️ Training time: {self.training_time:.2f}s")
        
        return result
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def get_detailed_report(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Get detailed classification report"""
        return {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> bool:
        """Validate input data"""
        if X is None or X.empty:
            raise ValueError("Input data X cannot be empty")
        
        if np.any(np.isinf(X.select_dtypes(include=[np.number]))):
            raise ValueError("Input data contains infinite values")
        
        if y is not None:
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
        
        return True
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training/prediction"""
        self.validate_data(X, y)
        
        X_processed = X.copy()
        
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_columns] = X_processed[numeric_columns].fillna(X_processed[numeric_columns].median())
        
        if y is not None:
            y_processed = y.copy()
            return X_processed, y_processed
        
        return X_processed, None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by the algorithm"""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        } 