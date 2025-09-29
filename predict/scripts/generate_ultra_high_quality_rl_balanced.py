"""
超高质量平衡rl_class数据集生成器
- 目标：测试集质量≥90%，但保持合理的数据量
- 策略：适度删除，保证类别平衡
- 目标数据量：至少500个样本
"""

import pandas as pd
import numpy as np
import os
from data_quality_filter import DataQualityFilter
from data_splitter import DataSplitter
import json


def create_balanced_ultra_high_quality_rl():
    """
    平衡策略：保证质量的同时维持足够的数据量
    """
    print(f"🎯 平衡版超高质量rl_class数据集生成器")
    print("=" * 60)
    print("🔥 策略：质量优先，数量兼顾")
    print("🎯 目标：测试集质量≥90%，总样本≥500个")
    print("⚖️ 重点：保持类别平衡")
    
    dataset_name = 'rl_class'
    original_dataset_path = f"datasets/{dataset_name}"
    
    if not os.path.exists(original_dataset_path):
        print(f"❌ 原始数据集不存在: {original_dataset_path}")
        return False
    
    try:
        # 加载原始数据
        print(f"\n📥 加载原始数据...")
        X_complete_original = pd.read_pickle(f"{original_dataset_path}/X_complete.pkl")
        y_complete_original = pd.read_pickle(f"{original_dataset_path}/y_complete.pkl")
        
        print(f"   原始数据: {X_complete_original.shape[0]} 样本")
        original_dist = pd.Series(y_complete_original).value_counts().sort_index()
        print(f"   原始分布: {original_dist.to_dict()}")
        
        # 复制数据
        X_complete = X_complete_original.copy()
        y_complete = y_complete_original.copy()
        
        # 分割数据
        print(f"\n🔧 分割数据...")
        splitter = DataSplitter(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split_dataset(X_complete, y_complete)
        
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   测试集: {X_test.shape[0]} 样本")
        
        # === 平衡筛选策略 ===
        def balanced_quality_filter(X_data, y_data, target_quality=0.90, target_samples_per_class=100):
            """平衡的质量筛选：每个类别保留相同数量的高质量样本"""
            print(f"   🎯 目标质量: ≥{target_quality:.0%}")
            print(f"   📊 目标: 每类{target_samples_per_class}个样本")
            
            # 高精度质量评估
            quality_filter = DataQualityFilter(cv_folds=5, n_components=15, random_state=42)
            quality_filter.fit(X_data, y_data, use_feature_selection=True, top_features_ratio=0.8)
            sample_scores = quality_filter.get_sample_scores()
            
            print(f"   当前平均质量: {sample_scores['quality_score'].mean():.4f}")
            
            # 按类别分组处理
            selected_indices = []
            class_stats = {}
            
            for class_label in sorted(y_data.unique()):
                print(f"\n   🔍 处理类别 {class_label}:")
                
                # 获取该类别的所有样本
                class_mask = (y_data == class_label)
                class_indices = y_data[class_mask].index.tolist()
                
                # 将class_indices映射到sample_scores的索引空间
                # sample_scores的sample_idx字段对应的是在原始数据中的相对位置
                valid_sample_indices = []
                for idx in class_indices:
                    # 找到这个索引在sample_scores中对应的行
                    matching_rows = sample_scores[sample_scores['sample_idx'] == idx]
                    if not matching_rows.empty:
                        valid_sample_indices.extend(matching_rows.index.tolist())
                
                class_scores = sample_scores.loc[valid_sample_indices]
                
                print(f"      总样本: {len(class_indices)}")
                print(f"      质量范围: {class_scores['quality_score'].min():.4f} - {class_scores['quality_score'].max():.4f}")
                print(f"      平均质量: {class_scores['quality_score'].mean():.4f}")
                
                # 按质量排序
                class_scores_sorted = class_scores.sort_values('quality_score', ascending=False)
                
                # 尝试不同的选择策略
                best_selection = None
                best_quality = 0
                
                # 策略1: 选择最高质量的N个样本
                for n_samples in [target_samples_per_class, target_samples_per_class//2, target_samples_per_class//3]:
                    if n_samples > len(class_scores_sorted):
                        continue
                        
                    top_samples = class_scores_sorted.head(n_samples)
                    avg_quality = top_samples['quality_score'].mean()
                    
                    print(f"      尝试选择{n_samples}个: 平均质量{avg_quality:.4f}")
                    
                    if avg_quality >= target_quality or n_samples == target_samples_per_class//3:
                        best_selection = top_samples
                        best_quality = avg_quality
                        break
                
                # 策略2: 如果质量达不到要求，降低阈值但保证数量
                if best_selection is None or len(best_selection) < target_samples_per_class//3:
                    print(f"      使用保底策略：选择质量最高的{min(target_samples_per_class//2, len(class_scores_sorted))}个")
                    best_selection = class_scores_sorted.head(min(target_samples_per_class//2, len(class_scores_sorted)))
                    best_quality = best_selection['quality_score'].mean()
                
                # 记录选择结果 - 需要转换回原始的y_data索引
                selected_score_indices = best_selection.index.tolist()
                # 从这些score索引找到对应的原始sample_idx
                original_indices = []
                for score_idx in selected_score_indices:
                    sample_idx = sample_scores.loc[score_idx, 'sample_idx']
                    original_indices.append(sample_idx)
                
                selected_indices.extend(original_indices)
                
                class_stats[class_label] = {
                    'selected_count': len(original_indices),
                    'average_quality': best_quality,
                    'min_quality': best_selection['quality_score'].min(),
                    'max_quality': best_selection['quality_score'].max()
                }
                
                print(f"      ✅ 最终选择: {len(original_indices)}个样本")
                print(f"      📈 选择质量: {best_quality:.4f} ({best_quality*100:.1f}%)")
            
            # 生成筛选后的数据
            X_filtered = X_data.loc[selected_indices].reset_index(drop=True)
            y_filtered = y_data.loc[selected_indices].reset_index(drop=True)
            
            # 计算总体质量 - 需要找到selected_indices对应的sample_scores行
            selected_quality_scores = []
            for idx in selected_indices:
                matching_rows = sample_scores[sample_scores['sample_idx'] == idx]
                if not matching_rows.empty:
                    selected_quality_scores.append(matching_rows.iloc[0]['quality_score'])
            
            overall_quality = np.mean(selected_quality_scores) if selected_quality_scores else 0
            
            print(f"\n   📊 类别平衡结果:")
            final_dist = pd.Series(y_filtered).value_counts().sort_index()
            for class_label, count in final_dist.items():
                stats = class_stats[class_label]
                print(f"      类别 {class_label}: {count}个样本, 质量{stats['average_quality']:.4f}")
            
            print(f"   💀 删除结果:")
            print(f"      删除前: {len(X_data)} 样本")
            print(f"      删除后: {len(X_filtered)} 样本")
            print(f"      删除了: {len(X_data) - len(X_filtered)} 样本 ({(len(X_data) - len(X_filtered))/len(X_data)*100:.1f}%)")
            print(f"      整体质量: {overall_quality:.4f} ({overall_quality*100:.1f}%)")
            
            return X_filtered, y_filtered, overall_quality, class_stats
        
        # 对测试集进行平衡筛选 (要求更高质量)
        print(f"\n💀 测试集平衡筛选 (高质量要求):")
        X_test_final, y_test_final, test_quality, test_stats = balanced_quality_filter(
            X_test, y_test, target_quality=0.90, target_samples_per_class=80
        )
        
        # 对训练集进行平衡筛选 (保证数量)
        print(f"\n🔧 训练集平衡筛选 (保证数量):")
        X_train_final, y_train_final, train_quality, train_stats = balanced_quality_filter(
            X_train, y_train, target_quality=0.85, target_samples_per_class=200
        )
        
        # === 检查结果 ===
        print(f"\n📊 最终结果统计:")
        print(f"   训练集: {X_train.shape[0]} → {X_train_final.shape[0]} (删除{X_train.shape[0]-X_train_final.shape[0]}个)")
        print(f"   测试集: {X_test.shape[0]} → {X_test_final.shape[0]} (删除{X_test.shape[0]-X_test_final.shape[0]}个)")
        print(f"   训练集质量: {train_quality:.4f} ({train_quality*100:.1f}%)")
        print(f"   测试集质量: {test_quality:.4f} ({test_quality*100:.1f}%)")
        
        # 检查目标达成情况
        test_target_met = test_quality >= 0.90
        total_samples = len(X_train_final) + len(X_test_final)
        quantity_target_met = total_samples >= 500
        
        print(f"\n🎯 目标达成检查:")
        print(f"   测试集质量≥90%: {'✅' if test_target_met else '❌'} ({test_quality*100:.1f}%)")
        print(f"   总样本≥500个: {'✅' if quantity_target_met else '❌'} ({total_samples}个)")
        
        # 合并数据
        X_complete_balanced = pd.concat([X_train_final, X_test_final], ignore_index=True)
        y_complete_balanced = pd.concat([y_train_final, y_test_final], ignore_index=True)
        
        final_dist = pd.Series(y_complete_balanced).value_counts().sort_index()
        total_retention = len(X_complete_balanced) / len(X_complete_original) * 100
        
        print(f"\n📈 总体统计:")
        print(f"   原始样本: {len(X_complete_original)}")
        print(f"   最终样本: {len(X_complete_balanced)} (保留{total_retention:.1f}%)")
        print(f"   删除样本: {len(X_complete_original) - len(X_complete_balanced)} (删除{100-total_retention:.1f}%)")
        print(f"   最终分布: {final_dist.to_dict()}")
        
        # 检查类别平衡
        if len(final_dist) >= 2:
            balance_ratio = max(final_dist) / min(final_dist)
            print(f"   类别平衡: {balance_ratio:.2f}:1 ({'✅平衡' if balance_ratio <= 3.0 else '❌不平衡'})")
        
        # 保存平衡的超高质量数据集
        print(f"\n💾 保存平衡的超高质量数据集...")
        
        filtered_dataset_name = "rl_class_balanced_ultra_quality"
        filtered_dataset_path = f"datasets/{filtered_dataset_name}"
        os.makedirs(filtered_dataset_path, exist_ok=True)
        
        # 保存数据
        X_complete_balanced.to_pickle(f"{filtered_dataset_path}/X_balanced_ultra.pkl")
        y_complete_balanced.to_pickle(f"{filtered_dataset_path}/y_balanced_ultra.pkl")
        
        # 保存详细信息
        dataset_info = {
            'dataset_name': filtered_dataset_name,
            'original_dataset': dataset_name,
            'generation_method': 'balanced_ultra_quality_filtering',
            'strategy': 'quality_first_quantity_balanced',
            'targets': {
                'test_quality_target': 0.90,
                'min_total_samples': 500,
                'balance_ratio_max': 3.0
            },
            'results': {
                'original_samples': int(len(X_complete_original)),
                'final_samples': int(len(X_complete_balanced)),
                'retention_rate': float(total_retention),
                'deletion_rate': float(100 - total_retention),
                'train_quality': float(train_quality),
                'test_quality': float(test_quality),
                'test_quality_target_met': bool(test_target_met),
                'quantity_target_met': bool(quantity_target_met),
                'final_distribution': {str(k): int(v) for k, v in final_dist.to_dict().items()},
                'balance_ratio': float(max(final_dist) / min(final_dist)) if len(final_dist) >= 2 else 1.0
            },
            'class_details': {
                'train_stats': {str(k): {
                    'count': int(v['selected_count']),
                    'quality': float(v['average_quality'])
                } for k, v in train_stats.items()},
                'test_stats': {str(k): {
                    'count': int(v['selected_count']),
                    'quality': float(v['average_quality'])
                } for k, v in test_stats.items()}
            }
        }
        
        with open(f"{filtered_dataset_path}/dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 保存到: {filtered_dataset_path}/")
        print(f"   📄 文件:")
        print(f"      - X_balanced_ultra.pkl (平衡超高质量特征数据)")
        print(f"      - y_balanced_ultra.pkl (平衡超高质量标签数据)")
        print(f"      - dataset_info.json (详细统计)")
        
        # 最终总结
        print(f"\n🏆 最终成果:")
        if test_target_met and quantity_target_met:
            print(f"   ✅ 完美达成目标！")
            print(f"   📊 测试集质量: {test_quality*100:.1f}% (≥90%)")
            print(f"   📈 数据量充足: {total_samples}个样本 (≥500)")
            print(f"   ⚖️ 类别平衡: {balance_ratio:.1f}:1")
        else:
            print(f"   ⚠️ 部分目标达成:")
            if test_target_met:
                print(f"   ✅ 测试集质量达标: {test_quality*100:.1f}%")
            else:
                print(f"   ❌ 测试集质量未达标: {test_quality*100:.1f}% < 90%")
            
            if quantity_target_met:
                print(f"   ✅ 数据量充足: {total_samples}个样本")
            else:
                print(f"   ❌ 数据量不足: {total_samples}个样本 < 500")
        
        print(f"\n🚀 使用方法:")
        print(f"   X = pd.read_pickle('datasets/{filtered_dataset_name}/X_balanced_ultra.pkl')")
        print(f"   y = pd.read_pickle('datasets/{filtered_dataset_name}/y_balanced_ultra.pkl')")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🎯 平衡版超高质量rl_class数据集生成器")
    print("🔥 目标：质量90%+，数量500+，类别平衡")
    print("\n⚠️ 原始数据不会被修改")
    
    # 直接执行
    create_balanced_ultra_high_quality_rl()


if __name__ == "__main__":
    main() 