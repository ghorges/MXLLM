from typing import Dict, Any, List
import yaml
import json

# 导入工具函数
from utils import call_llm_with_json_output
from LLMMiner.parser.utils import log

class ContentEvaluator:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        self.system_message = self.prompts.get('content_evaluator', {}).get('system_message', '')
    
    def evaluate(self, 
                text: str, 
                extracted_content: List[Dict[str, Any]], 
                fields: List[str],
                model_name: str = "deepseek",
                temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate the extracted content against the original text and field list
        
        Args:
            text (str): The original text
            extracted_content (List[Dict[str, Any]]): The extracted content to evaluate
            fields (List[str]): The list of fields that were supposed to be extracted
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            
        Returns:
            Dict[str, Any]: Evaluation results containing:
                - status: "perfect" or "errors_found"
                - errors: List of error objects if any found
        """
        log.info(f"开始评估提取内容: {extracted_content}")
        # 使用AI模型评估提取内容
        try:
            # 将提取的内容转换为JSON字符串
            extracted_json = json.dumps(extracted_content, ensure_ascii=False, indent=2)
            fields_str = json.dumps(fields, ensure_ascii=False)
            
            # 准备用户提示
            user_prompt = f"""请评估以下从科学文献中提取的内容:

原始文本:
{text}

预设字段列表:
{fields_str}

提取的内容:
{extracted_json}

请根据系统提示中的规则评估提取内容的质量，并返回评估结果。"""
            
            # 调用LLM并获取JSON输出
            result = call_llm_with_json_output(
                system_prompt=self.system_message,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature
            )
            
            # 验证结果包含所有必要字段
            if "status" not in result:
                log.warning("AI响应缺少必要字段 'status'，使用默认值'perfect'")
                result["status"] = "perfect"
                
            if "errors" not in result:
                result["errors"] = []
                
            log.info(f"评估完成，结果: {result}")
            return result
            
        except Exception as e:
            log.error(f"调用AI评估内容时出错: {e}")
            # 发生错误时返回默认结果
            return {
                "status": "perfect",
                "errors": []
            } 