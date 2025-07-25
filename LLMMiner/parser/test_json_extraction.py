import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from LLMMiner.parser.utils import extract_json_from_text, extract_list_from_text

def test_json_extraction():
    """测试修改后的JSON提取函数"""
    
    # 测试用例: 格式 -> 文本
    test_cases = {
        "纯JSON对象": '{"name": "MXene", "formula": "Ti3C2Tx"}',
        
        "JSON数组": '[{"name": "MXene1"}, {"name": "MXene2"}, {"name": "MXene3"}]',
        
        "代码块中的JSON对象": """
        这是一些描述
        
        ```
        {"name": "MXene", "formula": "Ti3C2Tx"}
        ```
        
        更多描述
        """,
        
        "代码块中的JSON数组": """
        这是一些描述
        
        ```
        [{"name": "MXene1"}, {"name": "MXene2"}]
        ```
        
        更多描述
        """,
        
        "带语言标识的代码块": """
        这是一些描述
        
        ```json
        {"name": "MXene", "formula": "Ti3C2Tx"}
        ```
        
        更多描述
        """,
        
        "带嵌套对象": """
        分析结果如下:
        
        {
          "name": "MXene",
          "properties": {
            "conductivity": "high",
            "2D": true
          }
        }
        
        希望这对您有所帮助!
        """,
        
        "混合内容中的JSON对象": """
        有两个JSON对象:
        
        {"name": "MXene"}
        
        {"formula": "Ti3C2Tx"}
        """,
        
        "复杂数组": """
        材料数据:
        
        [
          {
            "name": "MXene",
            "type": "2D",
            "properties": {
              "conductivity": "high"
            }
          },
          {
            "name": "Graphene",
            "type": "2D"
          }
        ]
        """
    }
    
    print("===== 测试JSON提取 =====")
    for name, text in test_cases.items():
        json_str = extract_json_from_text(text)
        print(f"\n{name}:")
        print(f"原始文本: {text[:50]}..." if len(text) > 50 else f"原始文本: {text}")
        print(f"提取结果: {json_str[:50]}..." if len(json_str) > 50 else f"提取结果: {json_str}")
        try:
            # 尝试解析为JSON验证
            import json
            result = json.loads(json_str)
            print("✅ JSON有效")
        except json.JSONDecodeError as e:
            print(f"❌ JSON无效: {e}")
    
if __name__ == "__main__":
    test_json_extraction() 