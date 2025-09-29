"""
数据平衡器
处理类别不平衡问题，使用SMOTE等技术
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("警告：imbalanced-learn未安装，将使用简单的平衡策略")
    IMBLEARN_AVAILABLE = False


class DataBalancer:
    def __init__(self):
        """初始化数据平衡器"""
        self.balancer = None
        self.strategy = None
        
    def analyze_imbalance(self, y: pd.Series) -> Dict[str, float]:
        """
        分析数据不平衡情况
        
        Args:
            y: 标签数据
            
        Returns:
            不平衡分析结果
        """
        value_counts = y.value_counts()
        majority_count = value_counts.max()
        minority_count = value_counts.min()
        
        imbalance_ratio = majority_count / minority_count
        minority_percentage = minority_count / len(y) * 100
        
        print(f"   📊 类别不平衡分析:")
        print(f"      类别分布: {value_counts.to_dict()}")
        print(f"      不平衡比例: {imbalance_ratio:.2f}:1")
        print(f"      少数类占比: {minority_percentage:.1f}%")
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'minority_percentage': minority_percentage,
            'majority_count': majority_count,
            'minority_count': minority_count
        }
    
    def should_balance(self, imbalance_ratio: float, minority_percentage: float) -> bool:
        """
        判断是否需要数据平衡
        
        Args:
            imbalance_ratio: 不平衡比例
            minority_percentage: 少数类占比
            
        Returns:
            是否需要平衡
        """
        # 更严格的平衡条件：比例>2.5或少数类<30%
        return imbalance_ratio > 2.5 or minority_percentage < 30.0
    
    def balance_dataset(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       method: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        平衡数据集
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            method: 平衡方法 ('auto', 'smote', 'oversample', 'undersample', 'none')
            
        Returns:
            平衡后的特征和标签
        """
        # 检查不平衡情况
        imbalance_info = self.check_imbalance(y_train)
        
        print(f"   📊 类别不平衡分析:")
        print(f"      类别分布: {imbalance_info['class_counts']}")
        print(f"      不平衡比例: {imbalance_info['imbalance_ratio']:.2f}:1")
        print(f"      少数类占比: {imbalance_info['minority_percentage']:.1f}%")
        
        # 如果不平衡比例不严重，不进行处理
        if imbalance_info['imbalance_ratio'] < 2.0:
            print(f"   ✅ 类别相对平衡，无需处理")
            return X_train, y_train
        
        # 自动选择策略
        if method == 'auto':
            if imbalance_info['imbalance_ratio'] > 5.0:
                method = 'smote' if IMBLEARN_AVAILABLE else 'oversample'
            elif imbalance_info['imbalance_ratio'] > 3.0:
                method = 'oversample'
            else:
                method = 'none'
        
        if method == 'none':
            print(f"   ✅ 跳过数据平衡")
            return X_train, y_train
        
        print(f"   🔧 使用 {method} 方法平衡数据...")
        
        try:
            if method == 'smote' and IMBLEARN_AVAILABLE:
                # SMOTE过采样
                # 限制生成的样本数量，避免过度平衡
                target_ratio = min(0.8, 1.0 / imbalance_info['imbalance_ratio'] * 2)
                
                smote = SMOTE(
                    sampling_strategy=target_ratio,
                    random_state=42,
                    k_neighbors=min(5, imbalance_info['class_counts'][imbalance_info['minority_class']] - 1)
                )
                X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
                self.balancer = smote
                self.strategy = 'smote'
                
            elif method == 'oversample':
                # 随机过采样 - 更温和的平衡
                target_ratio = min(0.7, 1.0 / imbalance_info['imbalance_ratio'] * 1.5)
                
                if IMBLEARN_AVAILABLE:
                    oversampler = RandomOverSampler(
                        sampling_strategy=target_ratio,
                        random_state=42
                    )
                    X_balanced, y_balanced = oversampler.fit_resample(X_train, y_train)
                    self.balancer = oversampler
                else:
                    # 简单的重复采样
                    X_balanced, y_balanced = self._simple_oversample(X_train, y_train, target_ratio)
                
                self.strategy = 'oversample'
                
            elif method == 'undersample' and IMBLEARN_AVAILABLE:
                # 欠采样
                undersampler = RandomUnderSampler(
                    sampling_strategy=0.8,
                    random_state=42
                )
                X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
                self.balancer = undersampler
                self.strategy = 'undersample'
                
            else:
                print(f"   ⚠️ 不支持的平衡方法: {method}")
                return X_train, y_train
            
            # 转换回DataFrame和Series
            if isinstance(X_balanced, np.ndarray):
                X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
                y_balanced = pd.Series(y_balanced, name=y_train.name)
            
            # 检查平衡后的结果
            balanced_info = self.check_imbalance(y_balanced)
            print(f"   ✅ 数据平衡完成:")
            print(f"      平衡后分布: {balanced_info['class_counts']}")
            print(f"      新的不平衡比例: {balanced_info['imbalance_ratio']:.2f}:1")
            print(f"      样本数变化: {len(y_train)} → {len(y_balanced)}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"   ❌ 数据平衡失败: {e}")
            return X_train, y_train
    
    def _simple_oversample(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          target_ratio: float) -> Tuple[pd.DataFrame, pd.Series]:
        """
        简单的过采样实现（当imblearn不可用时）
        """
        class_counts = y_train.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        minority_samples = X_train[y_train == minority_class]
        minority_labels = y_train[y_train == minority_class]
        
        # 计算需要生成的样本数
        target_minority_count = int(class_counts[majority_class] * target_ratio)
        samples_needed = target_minority_count - len(minority_samples)
        
        if samples_needed > 0:
            # 随机重复采样
            indices = np.random.choice(minority_samples.index, 
                                     size=samples_needed, 
                                     replace=True)
            additional_samples = minority_samples.loc[indices]
            additional_labels = minority_labels.loc[indices]
            
            # 合并数据
            X_balanced = pd.concat([X_train, additional_samples], ignore_index=True)
            y_balanced = pd.concat([y_train, additional_labels], ignore_index=True)
        else:
            X_balanced = X_train.copy()
            y_balanced = y_train.copy()
        
        return X_balanced, y_balanced


def balance_training_data(X_train: pd.DataFrame, y_train: pd.Series, 
                         method: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
    """
    便捷函数：平衡训练数据
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        method: 平衡方法
        
    Returns:
        平衡后的训练特征和标签
    """
    balancer = DataBalancer()
    return balancer.balance_dataset(X_train, y_train, method) 