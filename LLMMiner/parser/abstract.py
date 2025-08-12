"""
摘要分析器模块

这个模块提供了摘要分析器的实现，用于分析材料科学文献摘要，
特别是识别与MXene电磁波吸收相关的研究。
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import yaml
import sys

# LangChain导入
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# 导入工具函数
from utils import call_llm_with_json_output, call_llm_with_json_stream, log  # 导入流式函数


# 定义结构化输出模型
class MxeneIdentificationResult(BaseModel):
    """MXene材料识别结果结构"""
    is_mxene_material: bool = Field(description="是否涉及MXene材料")
    is_absorption_study: bool = Field(description="是否研究了电磁波吸收性能")


class AbstractAnalyzer:
    def __init__(self, prompt_file: str = "prompt.yaml"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        # 正确获取absorption_identifier部分的system_message
        self.system_message = self.prompts.get('absorption_identifier', {}).get('system_message', '')
    
    def analyze(self, text: str, model_name: str = "kimi", temperature: float = 0.0, use_streaming: bool = True) -> Dict[str, bool]:
        """
        Analyze the abstract text and return classification results
        
        Args:
            text (str): The abstract text to analyze
            model_name (str): The name of the model to use
            temperature (float): The temperature parameter for the model
            use_streaming (bool): Whether to use streaming output
            
        Returns:
            Dict[str, bool]: Classification results containing:
                - is_mxene_material
                - is_absorption_study 
                - is_review_paper
                - is_emi_shielding
        """
        # 使用AI模型进行分析
        try:
            # 准备用户提示
            user_prompt = f"请分析以下摘要:\n\n{text}"
            
            # 根据use_streaming参数决定使用哪个函数
            if use_streaming:
                print("\n===== 开始分析摘要 (流式输出) =====")
                result = call_llm_with_json_stream(
                    system_prompt=self.system_message,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    temperature=temperature
                )
                print("\n===== 摘要分析完成 =====")
            else:
                # 使用原来的非流式函数
                result = call_llm_with_json_output(
                    system_prompt=self.system_message,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    temperature=temperature
                )
            
            # 验证结果包含所有必要字段
            required_fields = ["is_mxene_material", "is_absorption_study", "is_review_paper", "is_emi_shielding"]
            for field in required_fields:
                if field not in result:
                    log.warning(f"AI响应缺少必要字段 '{field}'，使用默认值False")
                    result[field] = False
            
            return result
            
        except Exception as e:
            log.error(f"调用AI进行摘要分析时出错: {e}")
            # 发生错误时返回默认结果
            return {
                "is_mxene_material": False,
                "is_absorption_study": False,
                "is_review_paper": False,
                "is_emi_shielding": False
            }


# 导出类
__all__ = ['AbstractAnalyzer', 'MxeneIdentificationResult']


# 如果直接运行该文件，则执行测试
if __name__ == "__main__":
    # 测试摘要
    test_abstract = """
    MXene materials, particularly Ti3C2Tx, have garnered significant attention due to their exceptional electromagnetic wave absorption properties. 
    In this study, we prepared Ti3C2Tx/polymer composites and investigated their microwave absorption performance in the frequency range of 2-18 GHz. 
    The results showed a maximum reflection loss of -45.6 dB at 12.4 GHz with a thickness of 1.8 mm, and an effective absorption bandwidth of 4.2 GHz. 
    The outstanding absorption properties can be attributed to the unique 2D layered structure of MXene, which facilitates multiple reflections and interfacial polarization.
    """
    
    try:
        # 初始化分析器
        analyzer = AbstractAnalyzer()
        
        # 分析摘要
        result = analyzer.analyze(test_abstract)
        
        # 打印结果
        log.info("分析结果:")
        log.info(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 判断是否为MXene电磁波吸收研究
        if result.get("is_mxene_material") and result.get("is_absorption_study"):
            log.info("\n✅ 该摘要确实涉及MXene材料的电磁波吸收研究")
        else:
            log.info("\n❌ 该摘要不涉及MXene材料的电磁波吸收研究")
            
    except Exception as e:
        log.error(f"测试过程中发生错误: {e}") 