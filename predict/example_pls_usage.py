#!/usr/bin/env python3
"""
PLS Predictor 完整使用示例
演示如何使用基于datasets的PLS预测器进行材料性能预测
"""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数：演示完整的PLS预测器使用流程"""
    
    print("🧪 PLS Predictor 完整使用示例")
    print("=" * 60)
    print("基于datasets数据训练PLS模型并进行材料性能预测")
    print()
    
    # 检查数据集是否存在
    print("🔍 检查数据集...")
    datasets_dir = "./datasets"
    required_files = [
        "rl_class_train.csv",
        "rl_class_test.csv", 
        "eab_class_train.csv",
        "eab_class_test.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(datasets_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少以下数据文件:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\n请确保{datasets_dir}目录包含所有必需的CSV文件")
        return
    
    print("✅ 所有数据文件都存在")
    print()
    
    # 步骤1：训练和保存模型
    print("📚 步骤1：训练和保存模型")
    print("-" * 30)
    
    try:
        from pls_predictor import train_and_save_model
        
        print("🚀 开始训练PLS模型...")
        print("   - 使用datasets目录中的预处理数据")
        print("   - 集成matminer特征增强")
        print("   - 训练RL和EAB分类模型")
        print()
        
        results = train_and_save_model(
            datasets_dir=datasets_dir,
            model_path="trained_pls_model.pkl",
            use_basic_only=False,  # 使用所有可用特征
            use_grid_search=False  # 为了速度，不使用网格搜索
        )
        
        print("✅ 模型训练完成!")
        print("\n📊 训练结果:")
        print(f"   RL模型准确率: {results['rl_accuracy']:.3f}")
        print(f"   RL模型F1分数: {results['rl_f1']:.3f}")
        print(f"   EAB模型准确率: {results['eab_accuracy']:.3f}")
        print(f"   EAB模型F1分数: {results['eab_f1']:.3f}")
        print(f"   特征数量: {results['n_features']}")
        print(f"   PLS组件数: {results['n_components']}")
        print(f"   RL训练集大小: {results['rl_train_size']}")
        print(f"   EAB训练集大小: {results['eab_train_size']}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        print("\n可能的原因:")
        print("   1. 缺少必要的Python包 (sklearn, pandas, numpy)")
        print("   2. 缺少matminer/pymatgen包")
        print("   3. 数据格式问题")
        return
    
    print("\n" + "="*60)
    
    # 步骤2：使用训练好的模型进行预测
    print("🔮 步骤2：使用训练好的模型进行预测")
    print("-" * 40)
    
    try:
        from pls_predictor import predict_formula_properties
        
        # 定义测试分子式
        test_materials = [
            ("Ti3C2", "MXene材料"),
            ("Fe3O4", "磁铁矿"),
            ("C", "碳材料"),
            ("NiFe2O4", "镍铁氧体"),
            ("Al2O3", "氧化铝"),
            ("SiO2", "二氧化硅"),
            ("CoFe2O4", "钴铁氧体"),
            ("ZnO", "氧化锌")
        ]
        
        print("🧬 测试不同类型的材料:")
        print()
        
        for formula, description in test_materials:
            try:
                print(f"📝 {formula} ({description}):")
                
                prediction = predict_formula_properties(formula)
                
                # 格式化输出结果
                rl_pred = prediction['rl_prediction']
                rl_conf = prediction['rl_confidence']
                eab_pred = prediction['eab_prediction']
                eab_conf = prediction['eab_confidence']
                
                print(f"   🎯 RL预测: {rl_pred} (置信度: {rl_conf:.3f})")
                print(f"   🎯 EAB预测: {eab_pred} (置信度: {eab_conf:.3f})")
                
                # 显示概率分布
                print("   📊 RL概率分布:", end="")
                for class_name, prob in prediction['rl_probabilities'].items():
                    print(f" {class_name}:{prob:.2f}", end="")
                print()
                
                print("   📊 EAB概率分布:", end="")
                for class_name, prob in prediction['eab_probabilities'].items():
                    print(f" {class_name}:{prob:.2f}", end="")
                print()
                print()
                
            except Exception as e:
                print(f"   ❌ 预测失败: {e}")
                print()
        
    except Exception as e:
        print(f"❌ 预测过程失败: {e}")
        return
    
    print("="*60)
    
    # 步骤3：演示高级用法
    print("⚙️ 步骤3：演示高级用法")
    print("-" * 25)
    
    try:
        from pls_predictor import PLSPredictor
        
        print("🔧 创建自定义预测器...")
        
        # 创建预测器实例
        predictor = PLSPredictor(
            datasets_dir=datasets_dir,
            use_all_features=True,
            n_components=15  # 使用更多的PLS组件
        )
        
        # 检查预测器状态
        print(f"   数据集目录: {predictor.datasets_dir}")
        print(f"   使用所有特征: {predictor.use_all_features}")
        print(f"   PLS组件数: {predictor.n_components}")
        print(f"   是否已训练: {predictor.is_trained}")
        
        # 如果需要，可以重新训练模型
        print("\n💡 提示: 可以通过调整参数重新训练模型:")
        print("   - 增加n_components提高模型复杂度")
        print("   - 启用use_grid_search进行超参数优化")
        print("   - 设置use_basic_only=True仅使用基础特征")
        
    except Exception as e:
        print(f"❌ 高级用法演示失败: {e}")
    
    print("\n" + "="*60)
    
    # 总结
    print("🎉 示例完成!")
    print()
    print("✅ 成功完成的任务:")
    print("   1. ✓ 从datasets目录加载预处理数据")
    print("   2. ✓ 训练基于PLS的分类模型")
    print("   3. ✓ 保存训练好的模型")
    print("   4. ✓ 对多种材料进行性能预测")
    print("   5. ✓ 展示预测结果和置信度")
    print()
    
    print("📚 关键特性:")
    print("   • 直接使用datasets中的预处理特征")
    print("   • 可选择使用所有特征或仅基础特征")
    print("   • 同时预测RL和EAB性能")
    print("   • 提供分类结果和置信度评估")
    print("   • 支持模型保存和加载")
    print()
    
    print("🔗 相关文件:")
    print("   • pls_predictor.py - 主要预测器代码")
    print("   • trained_pls_model.pkl - 保存的模型文件")
    print("   • PLS_PREDICTOR_README.md - 详细使用文档")
    print()
    
    print("💡 下一步:")
    print("   1. 根据需要调整模型参数")
    print("   2. 使用网格搜索优化性能")
    print("   3. 对自己的材料进行预测")
    print("   4. 分析预测结果指导材料设计")


if __name__ == "__main__":
    main() 