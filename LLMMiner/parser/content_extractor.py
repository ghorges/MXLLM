from typing import List, Dict, Any
import yaml
import re

# 导入工具函数
from utils import call_llm_with_json_output
from LLMMiner.parser.utils import log

class ContentExtractor:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        self.system_message = self.prompts.get('content_extractor', {}).get('system_message', '')
    
    def extract(self, text: str, fields: List[str], model_name: str = "deepseek", temperature: float = 0.0) -> List[Dict[str, Any]]:
        """
        Extract detailed content for each field from the text
        
        Args:
            text (str): The text to analyze
            fields (List[str]): List of fields to extract content for
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            
        Returns:
            List[Dict[str, Any]]: List of records, each containing extracted content
                for the specified fields
        """
        # 使用AI模型提取详细内容
        try:
            # 准备用户提示
            fields_str = ", ".join(fields)
            user_prompt = f"""请从以下科学文献片段中提取详细内容:

科学文献原文:
{text}

预设字段列表:
{fields_str}

请确保提取的内容符合系统提示中的格式要求，并包含所有可能的记录。"""
            

            # 调用LLM并获取JSON输出
            result = call_llm_with_json_output(
                system_prompt="提取",
                user_prompt=self.system_message + "\n" + user_prompt,
                model_name=model_name,
                temperature=temperature
            )
            log.info(result)
            exit(0)
            # 处理结果
            if isinstance(result, list):
                # 如果已经是列表格式，直接返回
                return result
            elif isinstance(result, dict):
                # 如果是字典格式，检查是否包含records字段
                if "records" in result:
                    return result["records"]
                else:
                    # 将单个记录包装为列表
                    return [result]
            else:
                log.warning(f"警告: AI响应格式不正确: {type(result)}")
                return []
            
        except Exception as e:
            log.error(f"调用AI提取详细内容时出错: {e}")
            # 发生错误时返回一个基本记录
            return [{
                "record_designation": "Sample",
                "general_properties": {
                    "chemical_formula": "Unknown"
                }
            }] 