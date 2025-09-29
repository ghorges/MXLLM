"""
简单的分子式预测函数
输入分子式，返回 EAB 和 RL 预测结果
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_result_meaning(prediction_value: int, prediction_type: str) -> str:
    """
    获取预测结果的说明
    
    Args:
        prediction_value (int): 预测值 (0 or 1)
        prediction_type (str): 预测类型 ('eab' or 'rl')
    
    Returns:
        str: 结果说明
    """
    meanings = {
        'eab': {
            0: "差 - 有效吸收带宽 ≤ 4 GHz，微波吸收性能不佳",
            1: "好 - 有效吸收带宽 > 4 GHz，微波吸收性能良好"
        },
        'rl': {
            0: "好 - 反射损耗 ≤ -50 dB，微波吸收效果良好", 
            1: "差 - 反射损耗 > -50 dB，微波吸收效果不佳"
        }
    }
    
    if prediction_type in meanings and prediction_value in meanings[prediction_type]:
        return meanings[prediction_type][prediction_value]
    else:
        return "未知"

def predict_properties(formula: str, model_path: str = None) -> dict:
    """
    预测材料的微波吸收性能
    
    Args:
        formula (str): 化学分子式，例如 "Ti3C2", "Fe3O4", "C"
        model_path (str, optional): 模型文件路径，默认使用当前目录下的 trained_pls_model.pkl
    
    Returns:
        dict: 包含预测结果的字典
        {
            'formula': str,           # 输入的分子式
            'eab': int,              # EAB预测 (0=差, 1=好)
            'rl': int,               # RL预测 (0=好, 1=差)
            'eab_confidence': float, # EAB预测置信度 (0-1)
            'rl_confidence': float,  # RL预测置信度 (0-1)
            'eab_meaning': str,      # EAB结果说明
            'rl_meaning': str,       # RL结果说明
            'success': bool          # 是否预测成功
        }
    
    EAB和RL结果说明:
        EAB (有效吸收带宽):
        - 0: 差 - 有效吸收带宽 ≤ 4 GHz，微波吸收性能不佳
        - 1: 好 - 有效吸收带宽 > 4 GHz，微波吸收性能良好
        
        RL (反射损耗):
        - 0: 好 - 反射损耗 ≤ -50 dB，微波吸收效果良好
        - 1: 差 - 反射损耗 > -50 dB，微波吸收效果不佳
    
    Example:
        >>> result = predict_properties("Ti3C2")
        >>> print(f"EAB: {result['eab']} ({result['eab_meaning']})")
        >>> print(f"RL: {result['rl']} ({result['rl_meaning']})")
        EAB: 1 (好 - 吸收带宽较宽，微波吸收性能良好)
        RL: 0 (差 - 反射损耗较小，微波吸收效果不佳)
    """
    
    # 默认模型路径
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "trained_pls_model.pkl")
    
    try:
        # 导入预测器
        from pls_predictor import predict_formula_properties
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            return {
                'formula': formula,
                'eab': -1,
                'rl': -1, 
                'eab_confidence': 0.0,
                'rl_confidence': 0.0,
                'eab_meaning': "未知 - 模型文件不存在",
                'rl_meaning': "未知 - 模型文件不存在",
                'success': False,
                'error': f"模型文件不存在: {model_path}"
            }
        
        # 进行预测
        prediction = predict_formula_properties(formula, model_path)
        
        # 格式化返回结果
        eab_pred = prediction['eab_prediction']
        rl_pred = prediction['rl_prediction']
        
        result = {
            'formula': prediction['formula'],
            'eab': eab_pred,
            'rl': rl_pred,
            'eab_confidence': prediction['eab_confidence'],
            'rl_confidence': prediction['rl_confidence'],
            'eab_meaning': get_result_meaning(eab_pred, 'eab'),
            'rl_meaning': get_result_meaning(rl_pred, 'rl'),
            'success': True
        }
        
        return result
        
    except Exception as e:
        # 预测失败时返回错误信息
        return {
            'formula': formula,
            'eab': -1,
            'rl': -1,
            'eab_confidence': 0.0,
            'rl_confidence': 0.0,
            'eab_meaning': f"未知 - 预测失败: {str(e)[:50]}",
            'rl_meaning': f"未知 - 预测失败: {str(e)[:50]}",
            'success': False,
            'error': str(e)
        }


def predict_properties_simple(formula: str) -> tuple:
    """
    简化版预测函数，只返回 EAB 和 RL 结果
    
    Args:
        formula (str): 化学分子式
    
    Returns:
        tuple: (eab, rl) 预测结果元组
        eab 取值说明:
        - 0: 差 (有效吸收带宽 ≤ 4 GHz)  
        - 1: 好 (有效吸收带宽 > 4 GHz)
        - -1: 未知 (预测失败)
        
        rl 取值说明:
        - 0: 好 (反射损耗 ≤ -50 dB)
        - 1: 差 (反射损耗 > -50 dB)
        - -1: 未知 (预测失败)
        
    Example:
        >>> eab, rl = predict_properties_simple("Ti3C2")
        >>> print(f"EAB: {eab}, RL: {rl}")
        EAB: 1, RL: 0
    """
    result = predict_properties(formula)
    
    if result['success']:
        return result['eab'], result['rl']
    else:
        return -1, -1


def train_model_if_needed() -> bool:
    """
    如果模型不存在，则训练模型
    
    Returns:
        bool: 训练是否成功
    """
    model_path = os.path.join(os.path.dirname(__file__), "trained_pls_model.pkl")
    
    # 如果模型已存在，直接返回
    if os.path.exists(model_path):
        return True
    
    try:
        print("🚀 模型不存在，开始训练...")
        from pls_predictor import train_and_save_model
        
        # 训练并保存模型
        results = train_and_save_model(
            datasets_dir="./datasets",
            model_path=model_path,
            use_basic_only=False,
            use_grid_search=False
        )
        
        print(f"✅ 模型训练完成!")
        print(f"   RL准确率: {results['rl_accuracy']:.3f}")
        print(f"   EAB准确率: {results['eab_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return False


# 主要的公开接口函数
def get_material_properties(formula: str) -> dict:
    """
    获取材料的微波吸收性能预测（主要接口函数）
    
    这是推荐使用的主要函数，会自动处理模型训练和预测
    
    Args:
        formula (str): 化学分子式
    
    Returns:
        dict: 预测结果，包含以下字段:
        {
            'formula': str,           # 输入的分子式
            'eab': int,              # EAB预测 (0=差, 1=好, -1=未知)
            'rl': int,               # RL预测 (0=好, 1=差, -1=未知)
            'eab_confidence': float, # EAB预测置信度 (0-1)
            'rl_confidence': float,  # RL预测置信度 (0-1)
            'eab_meaning': str,      # EAB结果说明
            'rl_meaning': str,       # RL结果说明
            'success': bool          # 是否预测成功
        }
    
    EAB和RL结果说明:
        EAB (有效吸收带宽):
        - 0: 差 - 有效吸收带宽 ≤ 4 GHz，微波吸收性能不佳
        - 1: 好 - 有效吸收带宽 > 4 GHz，微波吸收性能良好
        
        RL (反射损耗):
        - 0: 好 - 反射损耗 ≤ -50 dB，微波吸收效果良好
        - 1: 差 - 反射损耗 > -50 dB，微波吸收效果不佳
    
    Example:
        >>> props = get_material_properties("Ti3C2")
        >>> print(f"EAB: {props['eab']} ({props['eab_meaning']})")
        >>> print(f"RL: {props['rl']} ({props['rl_meaning']})")
        EAB: 0 (差 - 吸收带宽较窄，微波吸收性能不佳)
        RL: 1 (好 - 反射损耗较大，微波吸收效果良好)
    """
    
    # 确保模型存在
    if not train_model_if_needed():
        return {
            'formula': formula,
            'eab': -1,
            'rl': -1,
            'eab_confidence': 0.0,
            'rl_confidence': 0.0,
            'eab_meaning': "未知 - 模型训练失败",
            'rl_meaning': "未知 - 模型训练失败",
            'success': False,
            'error': '模型训练失败'
        }
    
    # 进行预测
    return predict_properties(formula)


if __name__ == "__main__":
    # 测试函数
    print("🧪 测试材料性能预测函数")
    print("=" * 40)
    
    # 测试材料列表
    test_formulas = [
        "Ti3C2",     # MXene
        "Fe3O4",     # 磁铁矿
        "C",         # 碳
        "NiFe2O4",   # 镍铁氧体
        "Al2O3"      # 氧化铝
    ]
    
    for formula in test_formulas:
        print(f"\n📝 测试 {formula}:")
        
        # 使用主要接口函数
        result = get_material_properties(formula)
        
        if result['success']:
            print(f"   EAB: {result['eab']} ({result['eab_meaning']}) - 置信度: {result['eab_confidence']:.3f}")
            print(f"   RL: {result['rl']} ({result['rl_meaning']}) - 置信度: {result['rl_confidence']:.3f}")
        else:
            print(f"   ❌ 预测失败: {result.get('error', '未知错误')}")
        
        # 使用简化版函数
        eab, rl = predict_properties_simple(formula)
        print(f"   简化结果: EAB={eab}, RL={rl}")
    
    print("\n🎉 测试完成!")
    print("\n💡 使用方法:")
    print("   from predict_formula import get_material_properties")
    print("   result = get_material_properties('Ti3C2')")
    print("   print(f\"EAB: {result['eab']}, RL: {result['rl']}\")") 