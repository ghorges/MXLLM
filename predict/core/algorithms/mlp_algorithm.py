"""
Multi-Layer Perceptron (MLP) Algorithm
Neural network implementation using sklearn
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
import time
from .base_algorithm import BaseAlgorithm

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    MLP_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed, MLP algorithm unavailable")
    MLP_AVAILABLE = False


class MLPAlgorithm(BaseAlgorithm):
    def __init__(self, hidden_layer_sizes=(300, 150, 75), max_iter=3000, alpha=0.01, 
                 learning_rate_init=0.001, use_grid_search=False):
        """Initialize MLP algorithm"""
        super().__init__("MLP")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.use_grid_search = use_grid_search
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'MLPAlgorithm':
        """Train MLP model"""
        if not MLP_AVAILABLE:
            raise ImportError("sklearn not installed, cannot use MLP algorithm")
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        X_train_scaled, _ = self._ensure_standardized(X_train_processed)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            learning_rate='constant',
            solver='adam',
            activation='relu',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            tol=1e-8,
            beta_1=0.9,
            beta_2=0.999
        )
        
        if self.use_grid_search:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            
            self.model = GridSearchCV(
                mlp, 
                param_grid, 
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
        else:
            self.model = mlp
        
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
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X_test: 测试特征
            
        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型未训练，无法预测")
        
        return self.model.predict_proba(X_test)
    
    def get_loss_curve(self) -> np.ndarray:
        """
        获取损失曲线
        
        Returns:
            损失曲线数组
        """
        if not self.is_trained:
            raise ValueError("模型未训练，无法获取损失曲线")
        
        if self.use_grid_search:
            mlp_model = self.model.best_estimator_
        else:
            mlp_model = self.model
        
        return mlp_model.loss_curve_
    
    def get_validation_score(self) -> float:
        """
        获取验证分数
        
        Returns:
            验证分数
        """
        if not self.is_trained:
            raise ValueError("模型未训练，无法获取验证分数")
        
        if self.use_grid_search:
            mlp_model = self.model.best_estimator_
        else:
            mlp_model = self.model
        
        if hasattr(mlp_model, 'best_validation_score_'):
            return mlp_model.best_validation_score_
        else:
            return 0.0
    
    def plot_loss_curve(self, save_path: str = None):
        """
        绘制损失曲线
        
        Args:
            save_path: 保存路径
        """
        if not self.is_trained:
            print("模型未训练，无法绘制损失曲线")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            loss_curve = self.get_loss_curve()
            
            plt.figure(figsize=(10, 6))
            plt.plot(loss_curve, label='训练损失')
            plt.title('MLP训练损失曲线')
            plt.xlabel('迭代次数')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                print(f"损失曲线已保存到: {save_path}")
            else:
                plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法绘制损失曲线")
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        获取网络信息
        
        Returns:
            网络信息字典
        """
        if not self.is_trained:
            return {}
        
        if self.use_grid_search:
            mlp_model = self.model.best_estimator_
        else:
            mlp_model = self.model
        
        info = {
            'hidden_layer_sizes': mlp_model.hidden_layer_sizes,
            'activation': mlp_model.activation,
            'solver': mlp_model.solver,
            'n_layers': mlp_model.n_layers_,
            'n_outputs': mlp_model.n_outputs_,
            'n_iter': mlp_model.n_iter_,
            'loss': mlp_model.loss_,
            'alpha': mlp_model.alpha,
            'learning_rate_init': mlp_model.learning_rate_init
        }
        
        if hasattr(mlp_model, 'best_validation_score_'):
            info['best_validation_score'] = mlp_model.best_validation_score_
        
        return info


if __name__ == "__main__":
    # 测试MLP算法
    print("测试MLP算法...")
    
    if not MLP_AVAILABLE:
        print("sklearn未安装，跳过测试")
    else:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 生成测试数据
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换为DataFrame和Series
        X_train_df = pd.DataFrame(X_train_scaled)
        X_test_df = pd.DataFrame(X_test_scaled)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # 测试MLP算法
        mlp_algo = MLPAlgorithm(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            use_grid_search=False
        )
        results = mlp_algo.fit_and_evaluate(X_train_df, y_train_series, X_test_df, y_test_series)
        
        print(f"MLP测试结果: 准确率 = {results['accuracy']:.4f}")
        
        # 测试网络信息
        network_info = mlp_algo.get_network_info()
        print(f"网络层数: {network_info.get('n_layers', 'N/A')}")
        print(f"迭代次数: {network_info.get('n_iter', 'N/A')}")
        
        print("MLP算法测试完成！") 