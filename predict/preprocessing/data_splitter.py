"""
数据分割和预处理模块
功能：分割数据集，进行标准化和归一化
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataSplitter:
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        初始化数据分割器
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.imputers = {}
        
    def prepare_datasets(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        准备数据集，进行预处理但不分割
        
        Args:
            df: 输入数据框
            
        Returns:
            预处理后的完整数据集字典
        """
        datasets = {}
        
        # 检查可用的任务
        available_tasks = []
        if 'rl_class' in df.columns:
            available_tasks.append('rl_class')
        if 'eab_class' in df.columns:
            available_tasks.append('eab_class')
        
        if not available_tasks:
            print("❌ 没有找到可用的分类任务")
            return datasets
        
        print(f"📋 发现分类任务: {available_tasks}")
        
        # 为每个任务准备数据集
        for task in available_tasks:
            print(f"\n🎯 准备任务: {task}")
            
            # 筛选有效数据
            task_df = df[df[task].notna()].copy()
            
            if len(task_df) < 10:
                print(f"   ⚠️ {task} 数据量太少 ({len(task_df)} 条)，跳过")
                continue
            
            # 分离特征和标签
            label_columns = ['rl_class', 'eab_class', 'rl_value', 'eab_value']
            feature_columns = [col for col in task_df.columns 
                             if col not in label_columns + ['record_designation', 'doi', 'formula', 
                                                           'original_formula', 'components', 'main_component', 
                                                           'secondary_component', 'elemental_composition']]
            
            X = task_df[feature_columns].copy()
            y = task_df[task].copy()
            
            print(f"   📊 原始数据: {len(X)} 样本, {len(feature_columns)} 特征")
            print(f"   📊 标签分布: {y.value_counts().to_dict()}")
            
            # 数据清理和预处理
            X_processed, feature_names = self._preprocess_features(X, task)
            
            if X_processed is None or len(X_processed.columns) == 0:
                print(f"   ❌ {task} 特征预处理失败，跳过")
                continue
            
            # 存储完整的预处理后数据集（不分割）
            datasets[task] = {
                'X': X_processed,  # 完整的特征数据
                'y': y,           # 完整的标签数据
                'feature_names': feature_names
            }
            
            print(f"   ✅ {task} 数据集准备完成:")
            print(f"      - 总样本数: {len(X_processed)}")
            print(f"      - 特征数: {len(X_processed.columns)}")
            print(f"      - 特征范围: [{X_processed.min().min():.3f}, {X_processed.max().max():.3f}]")
        
        return datasets
    
    def split_data(self, df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data for a specific target
        
        Args:
            df: Input dataframe
            target_name: Target column name (e.g., 'rl_class', 'eab_class')
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Prepare features (exclude target columns and non-numeric columns)
        exclude_columns = ['rl_class', 'eab_class', 'doi', 'record_designation', 
                          'chemical_formula', 'formula', 'original_formula', 'main_component', 'components']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        y = df[target_name]
        
        # Convert boolean columns to numeric
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        print(f"   ✅ Features for training: {X.shape[1]} numeric features")
        print(f"   ✅ Target distribution: {y.value_counts().to_dict()}")
        
        # Split the data
        return self.split_dataset(X, y)
    
    def split_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割数据集为训练集和测试集
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            训练集和测试集的特征和标签
        """
        try:
            # 尝试分层抽样
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y
            )
            print(f"   ✅ 使用分层抽样分割数据集")
        except Exception as e:
            print(f"   ⚠️ 分层抽样失败，使用随机抽样: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
        
        print(f"   📊 数据分割完成:")
        print(f"      - 训练集: {len(X_train)} 样本")
        print(f"      - 测试集: {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        标准化特征（从已有的预处理器或重新训练）
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            task: 任务名称
            
        Returns:
            标准化后的训练集和测试集特征
        """
        # 如果已有预训练的scaler，直接使用
        if task in self.scalers:
            scaler = self.scalers[task]['scaler']
            scaler_name = self.scalers[task]['scaler_name']
            print(f"   🔧 使用已保存的{scaler_name}")
        else:
            # 重新训练scaler
            print(f"   🔧 开始特征标准化...")
            
            # 检查数据分布，选择合适的标准化方法
            skewness = X_train.skew().abs()
            high_skew_features = skewness[skewness > 2].index.tolist()
            
            print(f"   📊 数据分布分析:")
            print(f"      高偏度特征 (|skew| > 2): {len(high_skew_features)} 个")
            
            # 根据数据特点选择标准化方法
            if len(high_skew_features) > len(X_train.columns) * 0.3:
                scaler = RobustScaler()
                scaler_name = "RobustScaler"
                print(f"   🔧 选择 RobustScaler (适合有异常值的数据)")
            else:
                scaler = StandardScaler()
                scaler_name = "StandardScaler"
                print(f"   🔧 选择 StandardScaler (标准正态分布)")
            
            # 保存标准化器
            self.scalers[task] = {
                'scaler': scaler,
                'scaler_name': scaler_name
            }
        
        # 拟合并转换训练集
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # 转换测试集
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 验证标准化效果
        print(f"   ✅ 特征标准化完成 ({scaler_name}):")
        print(f"      训练集均值: {X_train_scaled.mean().mean():.6f}")
        print(f"      训练集标准差: {X_train_scaled.std().mean():.6f}")
        print(f"      测试集范围: [{X_test_scaled.min().min():.3f}, {X_test_scaled.max().max():.3f}]")
        
        return X_train_scaled, X_test_scaled
    
    def _preprocess_features(self, X: pd.DataFrame, task: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        预处理特征
        
        Args:
            X: 特征数据框
            task: 任务名称
            
        Returns:
            处理后的特征数据框和特征名称列表
        """
        print(f"   🔧 开始特征预处理...")
        
        # 检查数据类型
        print(f"   📊 数据类型分布:")
        dtype_counts = X.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"      {dtype}: {count} 列")
        
        # 选择数值型特征
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   📊 数值型特征: {len(numeric_columns)} 个")
        
        if len(numeric_columns) == 0:
            print(f"   ❌ 没有数值型特征")
            return None, []
        
        X_numeric = X[numeric_columns].copy()
        
        # 检查和处理无限值
        inf_cols = []
        for col in X_numeric.columns:
            if np.isinf(X_numeric[col]).any():
                inf_cols.append(col)
                X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)
        
        if inf_cols:
            print(f"   🔧 处理无限值: {len(inf_cols)} 列")
        
        # 检查缺失值
        missing_info = X_numeric.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print(f"   🔧 处理缺失值: {len(missing_cols)} 列")
            print(f"      缺失最多的5列: {missing_cols.nlargest(5).to_dict()}")
            
            # 使用中位数填充缺失值
            imputer = SimpleImputer(strategy='median')
            X_numeric_filled = pd.DataFrame(
                imputer.fit_transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
            self.imputers[task] = imputer
            print(f"   ✅ 缺失值填充完成")
        else:
            X_numeric_filled = X_numeric.copy()
            print(f"   ✅ 无缺失值")
        
        # 移除常数特征（方差为0的特征）
        constant_features = []
        for col in X_numeric_filled.columns:
            if X_numeric_filled[col].var() == 0:
                constant_features.append(col)
        
        if constant_features:
            print(f"   🔧 移除常数特征: {len(constant_features)} 个")
            X_numeric_filled = X_numeric_filled.drop(columns=constant_features)
        
        # 移除高度相关的特征
        if len(X_numeric_filled.columns) > 1:
            correlation_threshold = 0.99
            corr_matrix = X_numeric_filled.corr().abs()
            
            # 找到高度相关的特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > correlation_threshold:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        high_corr_pairs.append((col_i, col_j, corr_matrix.iloc[i, j]))
            
            # 移除高度相关的特征（保留第一个）
            features_to_remove = set()
            for col_i, col_j, corr_val in high_corr_pairs:
                features_to_remove.add(col_j)  # 移除第二个特征
            
            if features_to_remove:
                print(f"   🔧 移除高相关特征: {len(features_to_remove)} 个 (相关性 > {correlation_threshold})")
                X_numeric_filled = X_numeric_filled.drop(columns=list(features_to_remove))
        
        # 最终检查
        final_features = X_numeric_filled.columns.tolist()
        print(f"   ✅ 特征预处理完成: {len(final_features)} 个特征")
        
        if len(final_features) == 0:
            print(f"   ❌ 预处理后没有有效特征")
            return None, []
        
        return X_numeric_filled, final_features
    
    def save_datasets(self, datasets: Dict, cache_dir: str) -> None:
        """
        保存完整数据集到缓存目录
        
        Args:
            datasets: 数据集字典
            cache_dir: 缓存目录
        """
        os.makedirs(cache_dir, exist_ok=True)
        
        for task_name, data in datasets.items():
            task_dir = os.path.join(cache_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            
            # 保存完整数据集
            data['X'].to_pickle(os.path.join(task_dir, 'X_complete.pkl'))
            data['y'].to_pickle(os.path.join(task_dir, 'y_complete.pkl'))
            
            # 保存特征名称
            with open(os.path.join(task_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(data['feature_names']))
            
            # 保存预处理器
            if task_name in self.scalers:
                joblib.dump(self.scalers[task_name]['scaler'], 
                           os.path.join(task_dir, 'scaler.pkl'))
            
            if task_name in self.imputers:
                joblib.dump(self.imputers[task_name], 
                           os.path.join(task_dir, 'imputer.pkl'))
        
        print(f"✅ 完整数据集已缓存到: {cache_dir}")
    
    def load_datasets(self, cache_dir: str) -> Optional[Dict]:
        """
        从缓存目录加载完整数据集
        
        Args:
            cache_dir: 缓存目录
            
        Returns:
            完整数据集字典或None
        """
        if not os.path.exists(cache_dir):
            return None
        
        datasets = {}
        
        for task_name in os.listdir(cache_dir):
            task_dir = os.path.join(cache_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            
            try:
                # 加载完整数据集
                X_complete = pd.read_pickle(os.path.join(task_dir, 'X_complete.pkl'))
                y_complete = pd.read_pickle(os.path.join(task_dir, 'y_complete.pkl'))
                
                # 加载特征名称
                feature_names_file = os.path.join(task_dir, 'feature_names.txt')
                if os.path.exists(feature_names_file):
                    with open(feature_names_file, 'r', encoding='utf-8') as f:
                        feature_names = [line.strip() for line in f.readlines()]
                else:
                    feature_names = X_complete.columns.tolist()
                
                # 加载预处理器
                scaler_file = os.path.join(task_dir, 'scaler.pkl')
                if os.path.exists(scaler_file):
                    scaler = joblib.load(scaler_file)
                    self.scalers[task_name] = {
                        'scaler': scaler,
                        'scaler_name': type(scaler).__name__
                    }
                
                imputer_file = os.path.join(task_dir, 'imputer.pkl')
                if os.path.exists(imputer_file):
                    self.imputers[task_name] = joblib.load(imputer_file)
                
                datasets[task_name] = {
                    'X': X_complete,
                    'y': y_complete,
                    'feature_names': feature_names
                }
                
                print(f"✅ 加载任务数据集: {task_name}")
                print(f"   - 总样本数: {len(X_complete)}, {len(X_complete.columns)} 特征")
                
            except Exception as e:
                print(f"❌ 加载任务 {task_name} 失败: {e}")
                continue
        
        return datasets if datasets else None
    
    def check_cached_datasets(self, cache_dir: str) -> bool:
        """
        检查是否存在缓存的完整数据集
        
        Args:
            cache_dir: 缓存目录
            
        Returns:
            是否存在缓存
        """
        if not os.path.exists(cache_dir):
            return False
        
        # 检查是否有有效的任务目录
        for item in os.listdir(cache_dir):
            task_dir = os.path.join(cache_dir, item)
            if os.path.isdir(task_dir):
                required_files = ['X_complete.pkl', 'y_complete.pkl']
                if all(os.path.exists(os.path.join(task_dir, f)) for f in required_files):
                    return True
        
        return False 