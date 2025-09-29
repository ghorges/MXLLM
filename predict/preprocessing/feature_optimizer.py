"""
优化特征选择模块
基于PLS测试结果的最佳配置
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class PLSTransformer(BaseEstimator, TransformerMixin):
    """PLS转换器，用于降维"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        
    def fit(self, X, y):
        # 将分类标签转换为数值
        y_numeric = y.astype(int) if hasattr(y, 'astype') else np.array(y, dtype=int)
        self.pls.fit(X, y_numeric)
        return self
    
    def transform(self, X):
        # 使用PLS变换特征，只返回X的变换结果
        X_transformed = self.pls.transform(X)
        # 确保返回2D数组
        if X_transformed.ndim > 2:
            X_transformed = X_transformed.reshape(X_transformed.shape[0], -1)
        return X_transformed
    

class OptimizedFeatureSelector:
    """优化的特征选择器"""
    
    def __init__(self):
        self.task_configs = {
            'rl_class': {
                'feature_method': 'mutual_info',
                'n_features': 50,
                'pls_components': 1,
                'description': 'Mutual Info (K=50) + PLS-1'
            },
            'eab_class': {
                'feature_method': 'rf_importance', 
                'n_features': 20,
                'pls_components': 1,
                'description': 'RF-Top20 + PLS-1'
            }
        }
        
        self.selectors = {}
        self.scalers = {}
        self.pls_transformers = {}
        self.classifiers = {}
    
    def _select_features_mutual_info(self, X_train, y_train, X_test, n_features):
        """使用互信息选择特征"""
        selector = SelectKBest(mutual_info_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # 转换回DataFrame保持列名
        selected_features = X_train.columns[selector.get_support()]
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        return X_train_selected, X_test_selected, selector
    
    def _select_features_rf_importance(self, X_train, y_train, X_test, n_features):
        """使用随机森林特征重要性选择特征"""
        # 训练随机森林获取特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 获取特征重要性排序
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 选择top特征
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        
        return X_train_selected, X_test_selected, rf
    
    def optimize_dataset_features(self, X_train, y_train, X_test, task_name):
        """
        为特定任务优化特征
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            task_name: 任务名称
            
        Returns:
            优化后的训练和测试特征
        """
        if task_name not in self.task_configs:
            print(f"⚠️ 未找到任务 {task_name} 的配置，使用默认配置")
            task_name = 'rl_class'  # 使用默认配置
        
        config = self.task_configs[task_name]
        print(f"🎯 使用最佳配置: {config['description']}")
        
        # 1. 标准化特征
        print(f"   🔧 标准化特征...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换回DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # 2. 特征选择
        print(f"   🔧 特征选择: {config['feature_method']}, 选择 {config['n_features']} 个特征")
        if config['feature_method'] == 'mutual_info':
            X_train_selected, X_test_selected, selector = self._select_features_mutual_info(
                X_train_scaled, y_train, X_test_scaled, config['n_features']
            )
        else:  # rf_importance
            X_train_selected, X_test_selected, selector = self._select_features_rf_importance(
                X_train_scaled, y_train, X_test_scaled, config['n_features']
            )
        
        # 3. PLS降维
        print(f"   🔧 PLS降维: {config['pls_components']} 个组件")
        pls_transformer = PLSTransformer(n_components=config['pls_components'])
        X_train_pls = pls_transformer.fit_transform(X_train_selected, y_train)
        X_test_pls = pls_transformer.transform(X_test_selected)
        
        # 转换为DataFrame
        pls_columns = [f'PLS_Component_{i+1}' for i in range(config['pls_components'])]
        X_train_final = pd.DataFrame(X_train_pls, columns=pls_columns, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_pls, columns=pls_columns, index=X_test.index)
        
        # 保存组件
        self.scalers[task_name] = scaler
        self.selectors[task_name] = selector
        self.pls_transformers[task_name] = pls_transformer
        
        print(f"   ✅ 特征优化完成: {X_train_final.shape[1]} 个PLS组件")
        print(f"   📊 特征范围: [{X_train_final.min().min():.3f}, {X_train_final.max().max():.3f}]")
        
        return X_train_final, X_test_final
    
    def create_optimized_pipeline(self, task_name):
        """创建优化的预测管道"""
        if task_name not in self.task_configs:
            task_name = 'rl_class'
        
        config = self.task_configs[task_name]
        
        # 创建管道步骤
        steps = [
            ('scaler', StandardScaler()),
        ]
        
        # 添加特征选择步骤
        if config['feature_method'] == 'mutual_info':
            steps.append(('feature_selector', SelectKBest(mutual_info_classif, k=config['n_features'])))
        else:
            # 对于RF重要性，需要自定义选择器
            steps.append(('feature_selector', SelectKBest(mutual_info_classif, k=config['n_features'])))
        
        # 添加PLS降维
        steps.append(('pls', PLSTransformer(n_components=config['pls_components'])))
        
        # 添加分类器
        steps.append(('classifier', LogisticRegression(max_iter=1000)))
        
        pipeline = Pipeline(steps)
        
        return pipeline
    
    def get_feature_importance_analysis(self, X_train, y_train, task_name):
        """获取特征重要性分析"""
        if task_name not in self.task_configs:
            return None
        
        config = self.task_configs[task_name]
        
        if config['feature_method'] == 'rf_importance':
            # 使用随机森林分析特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(config['n_features'])
        
        elif config['feature_method'] == 'mutual_info':
            # 使用互信息分析特征重要性
            selector = SelectKBest(mutual_info_classif, k=config['n_features'])
            selector.fit(X_train, y_train)
            
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            
            return importance_df.head(config['n_features'])
        
        return None


def optimize_dataset_features(X_train, y_train, X_test, task_name):
    """
    便捷函数：为数据集优化特征
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        task_name: 任务名称
        
    Returns:
        优化后的训练和测试特征
    """
    optimizer = OptimizedFeatureSelector()
    return optimizer.optimize_dataset_features(X_train, y_train, X_test, task_name)


if __name__ == "__main__":
    # 测试代码
    print("🔬 测试优化特征选择器...")
    
    # 创建示例数据
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 50), columns=[f'feature_{i}' for i in range(50)])
    y_train = pd.Series(np.random.choice([0, 1], 100))
    X_test = pd.DataFrame(np.random.randn(30, 50), columns=[f'feature_{i}' for i in range(50)])
    
    optimizer = OptimizedFeatureSelector()
    
    # 测试两个任务
    for task in ['rl_class', 'eab_class']:
        print(f"\n🎯 测试任务: {task}")
        X_train_opt, X_test_opt = optimizer.optimize_dataset_features(X_train, y_train, X_test, task)
        print(f"   原始特征: {X_train.shape[1]} → 优化后: {X_train_opt.shape[1]}")
    
    print("\n✅ 优化特征选择器测试完成！") 