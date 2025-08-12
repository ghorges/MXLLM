from typing import List, Dict, Any
import yaml
import re

# 导入工具函数
from utils import call_llm_with_json_output, call_llm_with_json_stream, log  # 导入流式函数

class FieldExtractor:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        self.system_message = self.prompts.get('field_extractor', {}).get('system_message', '')
    
    def extract(self, text: str, model_name: str = "deepseek", temperature: float = 0.0, use_streaming: bool = True) -> List[str]:
        """
        Extract fields from the text according to the predefined field list
        
        Args:
            text (str): The text to analyze
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            use_streaming (bool): Whether to use streaming output
            
        Returns:
            List[str]: List of field names found in the text
        """
        # 使用AI模型提取字段
        try:
            # 准备用户提示
            user_prompt = f"请从以下科学文献片段中提取相关字段:\n\n{text}"
            
            # 根据use_streaming参数决定使用哪个函数
            if use_streaming:
                print("\n===== 开始提取字段 (流式输出) =====")
                result = call_llm_with_json_stream(
                    system_prompt=self.system_message,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    temperature=temperature
                )
                print("\n===== 字段提取完成 =====")
            else:
                # 使用原来的非流式函数
                result = call_llm_with_json_output(
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