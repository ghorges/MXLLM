"""
Partial Least Squares (PLS) Algorithm
Suitable for high-dimensional data classification tasks
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
import time
from .base_algorithm import BaseAlgorithm

try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    PLS_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed, PLS algorithm unavailable")
    PLS_AVAILABLE = False


class PLSAlgorithm(BaseAlgorithm):
    def __init__(self, n_components: int = 10, use_grid_search: bool = True):
        """Initialize PLS algorithm"""
        super().__init__("PLS")
        self.n_components = n_components
        self.use_grid_search = use_grid_search
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'PLSAlgorithm':
        """Train PLS model"""
        if not PLS_AVAILABLE:
            raise ImportError("sklearn not installed, cannot use PLS algorithm")
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        X_train_scaled, _ = self._ensure_standardized(X_train_processed)
        
        n_components = min(self.n_components, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
        
        class PLSTransformer:
            def __init__(self, n_components=10):
                self.n_components = n_components
                self.pls = PLSRegression(n_components=n_components)
                
            def fit(self, X, y):
                self.pls.fit(X, y)
                return self
                
            def transform(self, X):
                X_transformed = self.pls.transform(X)
                if X_transformed.ndim == 3:
                    X_transformed = X_transformed.reshape(X_transformed.shape[0], -1)
                return X_transformed
        
        pipeline = Pipeline([
            ('pls', PLSTransformer(n_components=n_components)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        if self.use_grid_search:
            param_grid = {
                'pls__n_components': [min(5, n_components), 
                                    min(10, n_components), 
                                    min(20, n_components)],
                'classifier__C': [0.1, 1.0, 10.0]
            }
            
            self.model = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=min(5, 20),
                scoring='f1_weighted',
                n_jobs=-1
            )
        else:
            self.model = pipeline
        
        self.model.fit(X_train_scaled, y_train_processed)
        self.is_trained = True
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_test_processed, _ = self.prepare_data(X_test)
        X_test_scaled, _ = self._ensure_standardized(X_test_processed)
        
        return self.model.predict(X_test_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained or not hasattr(self.model, 'best_estimator_'):
            return {}
        
        try:
            if hasattr(self.model, 'best_estimator_'):
                pls_step = self.model.best_estimator_.named_steps['pls']
            else:
                pls_step = self.model.named_steps['pls']
            
            if hasattr(pls_step, 'pls') and hasattr(pls_step.pls, 'x_weights_'):
                weights = np.abs(pls_step.pls.x_weights_).mean(axis=1)
                return {f'feature_{i}': float(w) for i, w in enumerate(weights)}
        except:
            pass
        
        return {}


if __name__ == "__main__":
    # 测试PLS算法
    print("测试PLS算法...")
    
    if not PLS_AVAILABLE:
        print("sklearn未安装，跳过测试")
    else:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # 生成测试数据
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 转换为DataFrame和Series
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # 测试PLS算法
        pls_algo = PLSAlgorithm(n_components=5, use_grid_search=False)
        results = pls_algo.fit_and_evaluate(X_train_df, y_train_series, X_test_df, y_test_series)
        
        print(f"PLS测试结果: 准确率 = {results['accuracy']:.4f}")
        print("PLS算法测试完成！") 