# PLS Predictor 使用说明

## 概述

PLS Predictor 是一个基于偏最小二乘法(PLS)的材料属性预测系统，专门用于预测材料的微波吸收性能，包括反射损失(RL)和有效吸收带宽(EAB)。

## 主要特性

- ✅ **使用 datasets 目录中的预处理数据**
- ✅ **仅使用分子式、is_heterostructure、is_supported 作为基础特征**
- ✅ **集成 matminer 进行材料特征增强**
- ✅ **自动保存训练好的模型**
- ✅ **提供简单的预测函数，输入分子式返回 eab 和 rl 预测**

## 安装依赖

```bash
pip install pandas numpy scikit-learn
pip install pymatgen matminer  # 可选，用于材料特征增强
```

## 使用方法

### 🚀 最简单的使用方法 (推荐)

```python
from predict_formula import get_material_properties

# 预测材料性能（会自动训练模型，自动使用matminer）
result = get_material_properties("Ti3C2")

print(f"EAB: {result['eab']}, RL: {result['rl']}")
print(f"置信度: EAB={result['eab_confidence']:.3f}, RL={result['rl_confidence']:.3f}")
```

### ⚡ 更简单的接口

```python
from predict_formula import predict_properties_simple

# 只返回预测结果
eab, rl = predict_properties_simple("Ti3C2")
print(f"EAB: {eab}, RL: {rl}")
```

### 🔧 高级用法：手动训练模型

```python
from pls_predictor import train_and_save_model

# 训练模型并保存
results = train_and_save_model(
    datasets_dir="./datasets",           # datasets目录路径
    model_path="trained_pls_model.pkl",  # 保存的模型文件路径
    use_basic_only=False,                # 是否仅使用基础特征
    use_grid_search=False                # 是否使用网格搜索优化
)

print("训练结果:", results)
```

### 📊 批量预测

```python
from predict_formula import get_material_properties

materials = ["Ti3C2", "Fe3O4", "CoFe2O4", "C"]

for formula in materials:
    result = get_material_properties(formula)
    if result['success']:
        print(f"{formula}: EAB={result['eab']}, RL={result['rl']}")
```

### 3. 高级用法

```python
from pls_predictor import PLSPredictor

# 创建预测器实例
predictor = PLSPredictor(
    datasets_dir="./datasets",
    use_matminer=True,
    n_components=10
)

# 训练模型
results = predictor.train(use_grid_search=False)

# 保存模型
predictor.save_model("my_model.pkl")

# 预测多个分子式
test_formulas = ["Ti3C2", "Fe3O4", "C", "NiFe2O4"]
for formula in test_formulas:
    prediction = predictor.predict_from_formula(formula)
    print(f"{formula}: RL={prediction['rl_prediction']}, EAB={prediction['eab_prediction']}")
```

## 数据要求

### datasets 目录结构

```
datasets/
├── rl_class_train.csv      # RL分类训练数据
├── rl_class_test.csv       # RL分类测试数据
├── eab_class_train.csv     # EAB分类训练数据
└── eab_class_test.csv      # EAB分类测试数据
```

### CSV文件格式

每个CSV文件应包含以下列：

1. **基础特征列**：
   - `is_heterostructure`: 是否为异质结构 (0/1)
   - `is_supported`: 是否为负载结构 (0/1)

2. **元素分数列**：
   - 所有元素符号作为列名 (H, He, Li, Be, B, C, N, O, ...)
   - 值为该元素在化学式中的分数

3. **目标列**：
   - `target`: 分类标签

4. **可选的matminer特征列**：
   - 各种材料特征（如果已预计算）

## 工作原理

### 1. 特征提取

1. **基础特征**：从数据中提取 `is_heterostructure` 和 `is_supported`
2. **分子式重构**：从元素分数列重构化学分子式
3. **matminer特征**：使用重构的分子式生成材料特征

### 2. 模型训练

1. **数据预处理**：标准化特征，编码标签
2. **PLS降维**：使用偏最小二乘法进行特征降维
3. **分类预测**：使用逻辑回归进行最终分类

### 3. 预测流程

1. **输入分子式**：用户提供化学分子式
2. **特征生成**：使用matminer提取特征
3. **模型预测**：使用训练好的模型进行预测
4. **结果输出**：返回分类结果和置信度

## 输出格式

### 训练结果

```python
{
    'rl_accuracy': 0.85,        # RL模型准确率
    'rl_f1': 0.83,              # RL模型F1分数
    'eab_accuracy': 0.82,       # EAB模型准确率
    'eab_f1': 0.80,             # EAB模型F1分数
    'rl_train_size': 957,       # RL训练集大小
    'rl_test_size': 411,        # RL测试集大小
    'eab_train_size': 957,      # EAB训练集大小
    'eab_test_size': 411,       # EAB测试集大小
    'n_features': 150,          # 特征数量
    'n_components': 10          # PLS组件数量
}
```

### 预测结果

```python
{
    'formula': 'Ti3C2',                           # 输入分子式
    'rl_prediction': 'good',                      # RL预测类别
    'eab_prediction': 'excellent',                # EAB预测类别
    'rl_probabilities': {                         # RL各类别概率
        'poor': 0.1, 
        'good': 0.7, 
        'excellent': 0.2
    },
    'eab_probabilities': {                        # EAB各类别概率
        'poor': 0.05, 
        'good': 0.25, 
        'excellent': 0.7
    },
    'rl_confidence': 0.7,                         # RL预测置信度
    'eab_confidence': 0.7                         # EAB预测置信度
}
```

## 性能优化

### 1. 使用网格搜索

```python
results = train_and_save_model(
    use_grid_search=True  # 启用网格搜索优化超参数
)
```

### 2. 调整PLS组件数

```python
predictor = PLSPredictor(n_components=20)  # 增加PLS组件数
```

### 3. 禁用matminer（如果内存不足）

```python
predictor = PLSPredictor(use_matminer=False)  # 仅使用基础特征
```

## 故障排除

### 1. 依赖问题

```bash
# 确保安装所有必需的包
pip install pandas numpy scikit-learn pymatgen matminer
```

### 2. 数据格式问题

- 确保CSV文件包含所需的列
- 检查目标列的标签格式
- 验证元素分数列的数值范围

### 3. 内存问题

- 减少PLS组件数量
- 禁用matminer特征
- 使用较小的数据集进行测试

### 4. 预测问题

- 确保输入的分子式格式正确
- 检查模型是否已正确训练
- 验证特征提取是否成功

## 示例代码

完整的使用示例：

```python
#!/usr/bin/env python3
"""
PLS Predictor 完整使用示例
"""

from pls_predictor import train_and_save_model, predict_formula_properties

def main():
    print("🚀 开始训练PLS模型...")
    
    # 1. 训练和保存模型
    try:
        results = train_and_save_model(
            datasets_dir="./datasets",
            model_path="trained_pls_model.pkl",
            use_matminer=True,
            use_grid_search=False
        )
        
        print("✅ 模型训练完成!")
        print(f"RL模型准确率: {results['rl_accuracy']:.3f}")
        print(f"EAB模型准确率: {results['eab_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    # 2. 测试预测功能
    print("\n🔮 测试预测功能...")
    
    test_formulas = [
        "Ti3C2",      # MXene材料
        "Fe3O4",      # 铁氧体
        "C",          # 碳材料
        "NiFe2O4",    # 镍铁氧体
        "Al2O3"       # 氧化铝
    ]
    
    for formula in test_formulas:
        try:
            prediction = predict_formula_properties(formula)
            print(f"\n📝 {formula}:")
            print(f"   RL: {prediction['rl_prediction']} (置信度: {prediction['rl_confidence']:.3f})")
            print(f"   EAB: {prediction['eab_prediction']} (置信度: {prediction['eab_confidence']:.3f})")
            
        except Exception as e:
            print(f"   ❌ 预测失败: {e}")
    
    print("\n🎉 示例完成!")

if __name__ == "__main__":
    main()
```

## 参与贡献

如需改进或报告问题，请：

1. 检查现有的issues
2. 创建详细的bug报告
3. 提供可重现的测试案例
4. 考虑性能优化建议

## 🚀 快速开始

如果你只是想快速使用，只需要这几行代码：

```python
# 1. 导入函数
from predict_formula import get_material_properties

# 2. 预测材料性能
result = get_material_properties("Ti3C2")

# 3. 查看结果
if result['success']:
    print(f"EAB: {result['eab']}")  # 有效吸收带宽
    print(f"RL: {result['rl']}")    # 反射损失
else:
    print(f"预测失败: {result['error']}")
```

**就这么简单！** 函数会自动：
- 检查模型是否存在，不存在就自动训练
- 使用 matminer 自动提取材料特征
- 返回 EAB 和 RL 的预测结果

## 🎯 主要接口函数

| 函数名 | 用途 | 返回值 |
|--------|------|--------|
| `get_material_properties(formula)` | 主要接口，完整预测 | dict (包含详细信息) |
| `predict_properties_simple(formula)` | 简化接口 | tuple (eab, rl) |

## 🔗 集成到你的代码

```python
def analyze_materials(formulas_list):
    """分析多个材料的性能"""
    from predict_formula import get_material_properties
    
    results = []
    for formula in formulas_list:
        result = get_material_properties(formula)
        results.append(result)
    
    return results

# 使用示例
materials = ["Ti3C2", "Fe3O4", "CoFe2O4"]
analysis = analyze_materials(materials)
```

## 许可证

请参考项目根目录的LICENSE文件。 