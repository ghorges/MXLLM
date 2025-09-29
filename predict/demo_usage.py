"""
演示如何使用材料性能预测函数
"""

# 导入预测函数
from predict_formula import get_material_properties, predict_properties_simple

def demo_basic_usage():
    """演示基本用法"""
    print("🔬 基本用法演示")
    print("-" * 30)
    
    # 预测单个材料
    formula = "Ti3C2"
    result = get_material_properties(formula)
    
    if result['success']:
        print(f"材料: {result['formula']}")
        print(f"EAB (有效吸收带宽): {result['eab']}")
        print(f"RL (反射损失): {result['rl']}")
        print(f"EAB置信度: {result['eab_confidence']:.3f}")
        print(f"RL置信度: {result['rl_confidence']:.3f}")
    else:
        print(f"预测失败: {result.get('error', '未知错误')}")


def demo_batch_prediction():
    """演示批量预测"""
    print("\n📊 批量预测演示")
    print("-" * 30)
    
    # 测试多种材料
    materials = [
        "Ti3C2",      # MXene材料
        "Fe3O4",      # 磁铁矿  
        "CoFe2O4",    # 钴铁氧体
        "NiFe2O4",    # 镍铁氧体
        "C",          # 碳材料
        "Al2O3",      # 氧化铝
        "ZnO",        # 氧化锌
        "SiO2"        # 二氧化硅
    ]
    
    results = []
    
    for formula in materials:
        result = get_material_properties(formula)
        results.append(result)
        
        if result['success']:
            print(f"{formula:8} | EAB: {result['eab']:9} | RL: {result['rl']:9}")
        else:
            print(f"{formula:8} | 预测失败")
    
    return results


def demo_simple_interface():
    """演示简化接口"""
    print("\n⚡ 简化接口演示")
    print("-" * 30)
    
    formulas = ["Ti3C2", "Fe3O4", "C"]
    
    for formula in formulas:
        eab, rl = predict_properties_simple(formula)
        print(f"{formula}: EAB={eab}, RL={rl}")


def demo_integration_example():
    """演示如何在其他代码中集成"""
    print("\n🔗 集成示例")
    print("-" * 30)
    
    def analyze_material_performance(formula):
        """分析材料性能的示例函数"""
        result = get_material_properties(formula)
        
        if not result['success']:
            return f"无法分析 {formula}: {result.get('error', '未知错误')}"
        
        # 性能评估逻辑
        eab = result['eab']
        rl = result['rl']
        
        if eab == 'excellent' and rl == 'excellent':
            performance = "优秀"
        elif eab in ['excellent', 'good'] and rl in ['excellent', 'good']:
            performance = "良好"
        else:
            performance = "一般"
        
        return f"{formula} 的微波吸收性能: {performance} (EAB: {eab}, RL: {rl})"
    
    # 测试集成函数
    test_materials = ["Ti3C2", "Fe3O4", "Al2O3"]
    
    for material in test_materials:
        analysis = analyze_material_performance(material)
        print(analysis)


def demo_error_handling():
    """演示错误处理"""
    print("\n🛡️ 错误处理演示")
    print("-" * 30)
    
    # 测试无效分子式
    invalid_formulas = ["XYZ123", "", "InvalidFormula"]
    
    for formula in invalid_formulas:
        result = get_material_properties(formula)
        
        if result['success']:
            print(f"{formula}: 预测成功")
        else:
            print(f"{formula}: 预测失败 - {result.get('error', '未知错误')}")


if __name__ == "__main__":
    print("🎯 材料性能预测函数使用演示")
    print("=" * 50)
    
    # 运行所有演示
    demo_basic_usage()
    demo_batch_prediction() 
    demo_simple_interface()
    demo_integration_example()
    demo_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 演示完成!")
    print("\n📚 主要函数:")
    print("1. get_material_properties(formula) - 主要接口，返回详细结果")
    print("2. predict_properties_simple(formula) - 简化接口，返回(eab, rl)元组")
    print("\n🔧 特性:")
    print("• 自动处理模型训练（如果模型不存在）")
    print("• 自动使用matminer进行特征增强")
    print("• 完善的错误处理")
    print("• 简单易用的接口")
    print("\n💡 使用方法:")
    print("```python")
    print("from predict_formula import get_material_properties")
    print("result = get_material_properties('Ti3C2')")
    print("print(f\"EAB: {result['eab']}, RL: {result['rl']}\")")
    print("```") 