from typing import List, Dict, Any
import yaml

# 导入工具函数
from utils import call_llm_with_list_output
from LLMMiner.parser.utils import log

class FieldExtractor:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        self.system_message = self.prompts.get('field_extractor', {}).get('system_message', '')
    
    def extract(self, text: str, model_name: str = "deepseek", temperature: float = 0.0) -> List[str]:
        """
        Extract fields from the text according to the predefined field list
        
        Args:
            text (str): The text to analyze
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            
        Returns:
            List[str]: List of field names found in the text
        """
        # 使用AI模型提取字段
        try:
            # 准备用户提示
            user_prompt = f"请从以下科学文献片段中提取相关字段:\n\n{text}"
            
            # 调用LLM并获取列表输出
            result = call_llm_with_list_output(
                system_prompt=self.system_message,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature
            )
            
            # 验证结果是否为有效的字符串列表
            valid_fields = []
            for item in result:
                if isinstance(item, str):
                    valid_fields.append(item)
                elif isinstance(item, dict) and 'field' in item:
                    valid_fields.append(item['field'])
            
            return valid_fields
            
        except Exception as e:
            log.info(f"调用AI提取字段时出错: {e}")
            # 发生错误时返回空列表
            return [] 