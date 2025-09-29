"""
数据质量筛选器
基于PLSDA模型的交叉验证结果，筛选出预测效果好的样本
目标：提高模型性能，保持原始数据不变
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class PLSDAForFiltering:
    """用于数据筛选的PLSDA分类器"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        """训练模型"""
        y_encoded = self.label_encoder.fit_transform(y)
        self.pls.fit(X, y_encoded)
        return self
    
    def predict(self, X):
        """预测"""
        y_pred_continuous = self.pls.predict(X)
        y_pred_rounded = np.round(y_pred_continuous.flatten()).astype(int)
        y_pred_clipped = np.clip(y_pred_rounded, 0, len(self.label_encoder.classes_) - 1)
        return self.label_encoder.inverse_transform(y_pred_clipped)
    
    def predict_proba_like(self, X):
        """获取预测置信度（模拟概率）"""
        y_pred_continuous = self.pls.predict(X).flatten()
        # 将连续预测值转换为置信度分数
        distances = np.abs(y_pred_continuous - np.round(y_pred_continuous))
        confidences = 1 - distances  # 距离整数越近，置信度越高
        return confidences


class DataQualityFilter:
    """数据质量筛选器"""
    
    def __init__(self, cv_folds=5, n_components=10, random_state=42):
        self.cv_folds = cv_folds
        self.n_components = n_components
        self.random_state = random_state
        self.sample_scores_ = None
        self.feature_importance_ = None
        
    def _calculate_vip_scores(self, X, y):
        """计算VIP特征重要性分数"""
        print("   🔍 计算VIP特征重要性...")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练PLSDA获取VIP
        max_components = min(self.n_components, X.shape[1], len(np.unique(y)))
        plsda = PLSDAForFiltering(n_components=max_components)
        plsda.fit(X_scaled, y)
        
        # 计算VIP分数
        W = plsda.pls.x_weights_
        Q = plsda.pls.y_loadings_
        
        ss_y = []
        for i in range(max_components):
            if Q.ndim == 1:
                ss_y_i = Q[i] ** 2 if i < len(Q) else 0
            else:
                ss_y_i = np.sum(Q[:, i] ** 2)
            ss_y.append(ss_y_i)
        
        total_ss_y = sum(ss_y)
        if total_ss_y == 0:
            return np.ones(X.shape[1])
        
        p = X.shape[1]
        vip_scores = np.zeros(p)
        
        for j in range(p):
            numerator = 0
            for i in range(max_components):
                numerator += (W[j, i] ** 2) * ss_y[i]
            vip_scores[j] = np.sqrt(p * numerator / total_ss_y)
        
        return vip_scores
    
    def _evaluate_sample_quality(self, X, y, feature_subset=None):
        """使用交叉验证评估每个样本的预测质量"""
        print("   🎯 评估样本预测质量...")
        
        if feature_subset is not None:
            X_subset = X.iloc[:, feature_subset] if hasattr(X, 'iloc') else X[:, feature_subset]
        else:
            X_subset = X
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        # 交叉验证设置
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # 存储每个样本的评估结果
        sample_correct_predictions = np.zeros(len(X_scaled))
        sample_confidences = np.zeros(len(X_scaled))
        sample_prediction_counts = np.zeros(len(X_scaled))
        
        # 交叉验证
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            print(f"      📊 交叉验证 {fold+1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], \
                                       y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # 训练模型
            max_components = min(self.n_components, X_train_fold.shape[1], len(np.unique(y_train_fold)))
            plsda = PLSDAForFiltering(n_components=max_components)
            plsda.fit(X_train_fold, y_train_fold)
            
            # 预测验证集
            y_pred = plsda.predict(X_val_fold)
            confidences = plsda.predict_proba_like(X_val_fold)
            
            # 记录预测结果
            correct_mask = (y_pred == y_val_fold.values if hasattr(y_val_fold, 'values') else y_pred == y_val_fold)
            sample_correct_predictions[val_idx] += correct_mask
            sample_confidences[val_idx] += confidences
            sample_prediction_counts[val_idx] += 1
        
        # 计算最终分数
        # 准确率分数（每个样本被正确预测的比例）
        accuracy_scores = sample_correct_predictions / np.maximum(sample_prediction_counts, 1)
        
        # 平均置信度分数
        confidence_scores = sample_confidences / np.maximum(sample_prediction_counts, 1)
        
        # 综合质量分数（准确率 + 置信度）
        quality_scores = 0.7 * accuracy_scores + 0.3 * confidence_scores
        
        return {
            'accuracy_scores': accuracy_scores,
            'confidence_scores': confidence_scores,
            'quality_scores': quality_scores,
            'prediction_counts': sample_prediction_counts
        }
    
    def fit(self, X, y, use_feature_selection=True, top_features_ratio=0.5):
        """
        训练筛选器，评估样本质量
        
        参数:
        - X: 特征数据
        - y: 标签数据  
        - use_feature_selection: 是否使用特征选择
        - top_features_ratio: 使用多少比例的重要特征
        """
        print("🔍 开始数据质量评估...")
        print(f"   📊 数据形状: {X.shape}")
        print(f"   🏷️ 标签分布: {pd.Series(y).value_counts().to_dict()}")
        
        feature_subset = None
        
        if use_feature_selection and X.shape[1] > 10:
            # 计算特征重要性
            vip_scores = self._calculate_vip_scores(X, y)
            self.feature_importance_ = pd.DataFrame({
                'feature_idx': range(len(vip_scores)),
                'feature_name': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
                'vip_score': vip_scores
            }).sort_values('vip_score', ascending=False)
            
            # 选择重要特征
            n_top_features = max(5, int(X.shape[1] * top_features_ratio))
            feature_subset = self.feature_importance_.head(n_top_features)['feature_idx'].tolist()
            
            print(f"   ✅ 特征选择: 从{X.shape[1]}个特征中选择前{n_top_features}个")
            print(f"   🔝 最重要特征: {self.feature_importance_.head(3)['feature_name'].tolist()}")
        
        # 评估样本质量
        sample_eval = self._evaluate_sample_quality(X, y, feature_subset)
        
        # 存储结果
        self.sample_scores_ = pd.DataFrame({
            'sample_idx': range(len(X)),
            'accuracy_score': sample_eval['accuracy_scores'],
            'confidence_score': sample_eval['confidence_scores'], 
            'quality_score': sample_eval['quality_scores'],
            'prediction_count': sample_eval['prediction_counts']
        })
        
        print(f"   ✅ 样本质量评估完成")
        print(f"   📈 质量分数范围: {self.sample_scores_['quality_score'].min():.4f} - {self.sample_scores_['quality_score'].max():.4f}")
        print(f"   📊 平均质量分数: {self.sample_scores_['quality_score'].mean():.4f}")
        
        return self
    
    def filter_samples(self, X, y, quality_threshold=None, keep_ratio=0.8, min_samples_per_class=10):
        """
        筛选高质量样本
        
        参数:
        - X: 特征数据
        - y: 标签数据
        - quality_threshold: 质量阈值，None时自动确定
        - keep_ratio: 保留样本的比例
        - min_samples_per_class: 每个类别最少保留的样本数
        """
        if self.sample_scores_ is None:
            raise ValueError("请先调用fit()方法评估样本质量")
        
        print("🎯 开始筛选高质量样本...")
        
        # 确定质量阈值
        if quality_threshold is None:
            # 方法1: 按比例保留
            quality_threshold = self.sample_scores_['quality_score'].quantile(1 - keep_ratio)
        
        print(f"   📏 质量阈值: {quality_threshold:.4f}")
        
        # 按类别筛选，确保每个类别都有足够样本
        y_series = pd.Series(y, index=range(len(y)))
        filtered_indices = []
        
        for class_label in y_series.unique():
            class_mask = (y_series == class_label)
            class_indices = y_series[class_mask].index.tolist()
            class_scores = self.sample_scores_.loc[class_indices]
            
            # 确保每个类别至少保留min_samples_per_class个样本
            n_class_samples = len(class_indices)
            min_keep = min(min_samples_per_class, n_class_samples)
            
            # 按质量分数排序，保留高质量样本
            class_scores_sorted = class_scores.sort_values('quality_score', ascending=False)
            
            # 首先保留达到阈值的样本
            high_quality_mask = class_scores_sorted['quality_score'] >= quality_threshold
            high_quality_indices = class_scores_sorted[high_quality_mask]['sample_idx'].tolist()
            
            # 如果高质量样本不足最小要求，补充最好的样本
            if len(high_quality_indices) < min_keep:
                additional_needed = min_keep - len(high_quality_indices)
                remaining_indices = class_scores_sorted[~high_quality_mask]['sample_idx'].head(additional_needed).tolist()
                selected_indices = high_quality_indices + remaining_indices
            else:
                selected_indices = high_quality_indices
            
            filtered_indices.extend(selected_indices)
            
            print(f"   📊 类别 {class_label}: {n_class_samples} -> {len(selected_indices)} 样本")
            print(f"      平均质量: {class_scores.loc[class_scores['sample_idx'].isin(selected_indices), 'quality_score'].mean():.4f}")
        
        # 生成筛选后的数据
        filtered_indices = sorted(filtered_indices)
        X_filtered = X.iloc[filtered_indices] if hasattr(X, 'iloc') else X[filtered_indices]
        y_filtered = y.iloc[filtered_indices] if hasattr(y, 'iloc') else y[filtered_indices]
        
        # 重置索引
        if hasattr(X_filtered, 'reset_index'):
            X_filtered = X_filtered.reset_index(drop=True)
        if hasattr(y_filtered, 'reset_index'):
            y_filtered = y_filtered.reset_index(drop=True)
        
        print(f"   ✅ 筛选完成: {len(X)} -> {len(X_filtered)} 样本 ({len(X_filtered)/len(X)*100:.1f}%)")
        print(f"   📈 筛选后平均质量: {self.sample_scores_.loc[self.sample_scores_['sample_idx'].isin(filtered_indices), 'quality_score'].mean():.4f}")
        
        return X_filtered, y_filtered, filtered_indices
    
    def get_sample_scores(self):
        """获取样本质量分数"""
        return self.sample_scores_
    
    def get_feature_importance(self):
        """获取特征重要性"""
        return self.feature_importance_
    
    def save_quality_report(self, filepath, dataset_name):
        """保存质量评估报告"""
        print(f"💾 保存质量评估报告...")
        
        # 样本质量统计
        quality_stats = {
            'dataset_name': dataset_name,
            'total_samples': len(self.sample_scores_),
            'mean_quality': self.sample_scores_['quality_score'].mean(),
            'std_quality': self.sample_scores_['quality_score'].std(),
            'min_quality': self.sample_scores_['quality_score'].min(),
            'max_quality': self.sample_scores_['quality_score'].max(),
            'median_quality': self.sample_scores_['quality_score'].median(),
        }
        
        # 质量分布
        quality_bins = pd.cut(self.sample_scores_['quality_score'], bins=5, labels=['很低', '低', '中', '高', '很高'])
        quality_distribution = quality_bins.value_counts().to_dict()
        
        # 保存详细报告
        report = {
            'summary': quality_stats,
            'quality_distribution': quality_distribution,
            'sample_scores': self.sample_scores_.to_dict('records')
        }
        
        if self.feature_importance_ is not None:
            report['feature_importance'] = self.feature_importance_.to_dict('records')
        
        # 保存到文件
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 报告已保存到: {filepath}")
        return report 