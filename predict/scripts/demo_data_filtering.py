"""
数据质量筛选演示脚本
快速测试筛选功能是否正常工作
"""

import pandas as pd
import numpy as np
from data_quality_filter import DataQualityFilter
from data_splitter import DataSplitter
import os


def demo_data_filtering():
    """演示数据质量筛选功能"""
    print("🔬 数据质量筛选功能演示")
    print("=" * 50)
    
    # 选择一个数据集进行演示
    dataset_name = 'rl_class'  # 或者 'eab_class'
    dataset_path = f"datasets/{dataset_name}"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        print("请确保你已经运行了数据预处理步骤生成数据集")
        return
    
    try:
        print(f"📂 加载数据集: {dataset_name}")
        # 加载数据
        X_complete = pd.read_pickle(f"{dataset_path}/X_complete.pkl")
        y_complete = pd.read_pickle(f"{dataset_path}/y_complete.pkl")
        
        print(f"   ✅ 原始数据: {X_complete.shape[0]} 样本, {X_complete.shape[1]} 特征")
        print(f"   📊 标签分布: {pd.Series(y_complete).value_counts().to_dict()}")
        
        # 数据分割
        print(f"\n🔧 分割数据...")
        splitter = DataSplitter(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split_dataset(X_complete, y_complete)
        
        print(f"   训练集: {X_train.shape}")
        print(f"   测试集: {X_test.shape}")
        
        # 创建数据质量筛选器
        print(f"\n🔍 创建数据质量筛选器...")
        quality_filter = DataQualityFilter(
            cv_folds=3,  # 使用较少的fold以加快演示
            n_components=5,
            random_state=42
        )
        
        # 评估样本质量
        print(f"\n📊 评估训练集样本质量...")
        quality_filter.fit(X_train, y_train, use_feature_selection=True, top_features_ratio=0.3)
        
        # 查看质量分数
        sample_scores = quality_filter.get_sample_scores()
        print(f"   ✅ 质量评估完成")
        print(f"   📈 质量分数统计:")
        print(f"      平均值: {sample_scores['quality_score'].mean():.4f}")
        print(f"      标准差: {sample_scores['quality_score'].std():.4f}")
        print(f"      最小值: {sample_scores['quality_score'].min():.4f}")
        print(f"      最大值: {sample_scores['quality_score'].max():.4f}")
        
        # 显示质量分布
        quality_bins = pd.cut(sample_scores['quality_score'], bins=5, labels=['很低', '低', '中', '高', '很高'])
        quality_dist = quality_bins.value_counts()
        print(f"   📊 质量分布:")
        for level, count in quality_dist.items():
            print(f"      {level}: {count} 样本 ({count/len(sample_scores)*100:.1f}%)")
        
        # 筛选高质量样本
        print(f"\n🎯 筛选高质量样本...")
        keep_ratio = 0.8  # 保留80%
        min_samples_per_class = 5
        
        X_train_filtered, y_train_filtered, filtered_indices = quality_filter.filter_samples(
            X_train, y_train,
            keep_ratio=keep_ratio,
            min_samples_per_class=min_samples_per_class
        )
        
        # 显示筛选结果
        print(f"\n📈 筛选结果:")
        print(f"   原始训练集: {X_train.shape[0]} 样本")
        print(f"   筛选后: {X_train_filtered.shape[0]} 样本 ({X_train_filtered.shape[0]/X_train.shape[0]*100:.1f}%)")
        
        # 对比类别分布
        print(f"\n📊 类别分布对比:")
        original_dist = pd.Series(y_train).value_counts().sort_index()
        filtered_dist = pd.Series(y_train_filtered).value_counts().sort_index()
        
        for label in original_dist.index:
            orig_count = original_dist.get(label, 0)
            filt_count = filtered_dist.get(label, 0)
            retention_rate = filt_count / orig_count * 100 if orig_count > 0 else 0
            print(f"   类别 {label}: {orig_count} → {filt_count} ({retention_rate:.1f}%)")
        
        # 保存质量报告
        print(f"\n💾 保存质量评估报告...")
        quality_filter.save_quality_report(f"demo_quality_report_{dataset_name}.json", dataset_name)
        
        # 计算筛选后的平均质量
        filtered_quality_scores = sample_scores.loc[sample_scores['sample_idx'].isin(filtered_indices), 'quality_score']
        print(f"\n📈 质量提升效果:")
        print(f"   筛选前平均质量: {sample_scores['quality_score'].mean():.4f}")
        print(f"   筛选后平均质量: {filtered_quality_scores.mean():.4f}")
        print(f"   质量提升: {filtered_quality_scores.mean() - sample_scores['quality_score'].mean():+.4f}")
        
        print(f"\n✅ 演示完成！")
        print(f"📄 查看生成的JSON文件了解详细质量评估结果")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ 找不到数据文件")
        print(f"   期望文件: {dataset_path}/X_complete.pkl, {dataset_path}/y_complete.pkl")
        print("   请先运行数据预处理步骤生成这些文件")
        return False
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return False


if __name__ == "__main__":
    demo_data_filtering() 