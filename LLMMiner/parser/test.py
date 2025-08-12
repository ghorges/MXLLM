"""
测试Gemini API
"""

import sys
import os
from pathlib import Path

# 导入本地模块
from utils import get_llm, setup_logging
from langchain_core.messages import SystemMessage, HumanMessage

def test_gemini_api():
    """测试Gemini API"""
    # 设置日志
    log = setup_logging()
    
    try:
        # 获取Gemini模型
        log.info("正在初始化Gemini模型...")
        llm = get_llm(model_name="gemini", temperature=0.0)
        
        # 测试简单问候
        log.info("发送测试消息...")
        response = llm.invoke([
            SystemMessage(content="你是一个有用的AI助手。"),
            HumanMessage(content="你好")
        ])
        
        # 打印结果
        log.info(f"Gemini API响应:\n{response.content}")
        
        return True
    except Exception as e:
        log.error(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试Gemini API...")
    success = test_gemini_api()
    if success:
        print("测试成功完成!")
    else:
        print("测试失败!") 