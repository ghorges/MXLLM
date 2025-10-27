"""
XGBoost Algorithm
高效的梯度提升树算法，适用于分类任务，并支持特征重要性分析
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
import time
from .base_algorithm import BaseAlgorithm

try:
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    XGBOOST_AVAILABLE = True
except ImportError:
    print("警告: xgboost未安装, XGBoost算法不可用")
    XGBOOST_AVAILABLE = False


class XGBoostAlgorithm(BaseAlgorithm):
    def __init__(self, n_estimators: int = 300, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, use_grid_search: bool = False):
        """初始化XGBoost算法
        
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 每棵树使用的样本比例
            colsample_bytree: 每棵树使用的特征比例
            use_grid_search: 是否使用网格搜索优化参数
        """
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_grid_search = use_grid_search
        self.feature_names = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'XGBoostAlgorithm':
        """训练XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost未安装，无法使用XGBoost算法")
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        
        # 保存特征名称
        self.feature_names = X_train_processed.columns.tolist()
        
        # 创建XGBoost分类器
        xgb_clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        if self.use_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            self.model = GridSearchCV(
                xgb_clf, 
                param_grid, 
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        else:
            self.model = xgb_clf
        
        self.model.fit(X_train_processed, y_train_processed)
        self.is_trained = True
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_test_processed, _ = self.prepare_data(X_test)
        
        return self.model.predict(X_test_processed)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_test_processed, _ = self.prepare_data(X_test)
        
        return self.model.predict_proba(X_test_processed)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性
        
        Returns:
            特征名称到重要性分数的字典，按重要性降序排列
        """
        if not self.is_trained:
            return {}
        
        try:
            # 获取最佳模型（如果使用了网格搜索）
            if hasattr(self.model, 'best_estimator_'):
                model = self.model.best_estimator_
            else:
                model = self.model
            
            # 获取特征重要性
            feature_importance = model.feature_importances_
            
            # 创建特征名称到重要性的映射
            importance_dict = {}
            if self.feature_names:
                for i, importance in enumerate(feature_importance):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
            else:
                for i, importance in enumerate(feature_importance):
                    importance_dict[f'feature_{i}'] = float(importance)
            
            # 按重要性降序排序
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True))
            
            return importance_dict
        except Exception as e:
            print(f"获取特征重要性时出错: {e}")
            return {}
    
    def get_top_features(self, top_n: int = 20) -> Dict[str, float]:
        """获取前N个最重要的特征
        
        Args:
            top_n: 返回的特征数量
            
        Returns:
            前N个最重要特征的字典
        """
        all_importance = self.get_feature_importance()
        return dict(list(all_importance.items())[:top_n])

