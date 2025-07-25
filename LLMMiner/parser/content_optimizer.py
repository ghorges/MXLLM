from typing import Dict, Any, List
import yaml
import json
import copy

# 导入工具函数
from utils import call_llm_with_json_output
from LLMMiner.parser.utils import log

class ContentOptimizer:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        self.system_message = self.prompts.get('content_optimization', {}).get('system_message', '')
    
    def optimize(self, 
                content: List[Dict[str, Any]], 
                evaluation_result: Dict[str, Any],
                model_name: str = "deepseek",
                temperature: float = 0.1) -> List[Dict[str, Any]]:
        """
        Optimize the content based on evaluation results
        
        Args:
            content (List[Dict[str, Any]]): The original extracted content
            evaluation_result (Dict[str, Any]): The evaluation results containing errors
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            
        Returns:
            List[Dict[str, Any]]: The optimized content
        """
        # If no errors found, return original content
        if evaluation_result["status"] == "perfect":
            return content
            
        # 使用AI模型优化内容
        try:
            # 将原始内容和评估结果转换为JSON字符串
            content_json = json.dumps(content, ensure_ascii=False, indent=2)
            evaluation_json = json.dumps(evaluation_result, ensure_ascii=False, indent=2)
            
            # 准备用户提示
            user_prompt = f"""请根据评估结果优化以下从科学文献中提取的内容:

原始内容:
{content_json}

评估结果:
{evaluation_json}

请根据系统提示中的规则优化内容，并返回完整的优化后内容。"""
            
            # 调用LLM并获取JSON输出
            result = call_llm_with_json_output(
                system_prompt=self.system_message,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature
            )
            
            # 验证结果是否为列表
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # 检查是否包含records字段
                if "records" in result:
                    return result["records"]
                else:
                    # 将单个记录包装为列表
                    return [result]
            else:
                log.info(f"警告: AI响应格式不正确: {type(result)}")
                return content  # 返回原始内容
            
        except Exception as e:
            log.info(f"调用AI优化内容时出错: {e}")
            # 发生错误时返回原始内容
            return content 