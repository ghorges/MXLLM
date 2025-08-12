"""
测试Gemini API的流式传输功能
"""

from utils import call_llm_with_stream, setup_logging

def main():
    """测试流式传输功能"""
    # 设置日志
    log = setup_logging()
    log.info("开始测试流式传输...")
    
    try:
        # 定义提示
        system_prompt = "你是一个有用的AI助手。"
        user_prompt = "解释MXene材料的结构和特性，解释得详细一些。"
        
        # 调用流式API
        log.info("正在调用流式API...")
        stream = call_llm_with_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gemini",
            temperature=0.7  # 稍高的温度以获得更多变化
        )
        
        print("流式响应开始 (每个词将逐步显示)：\n" + "-" * 50)
        
        # 处理流式响应
        for chunk in stream:
            # 直接打印每个文本块，不换行
            print(chunk.text, end="", flush=True)
            
        print("\n" + "-" * 50)
        print("流式响应结束")
        
    except Exception as e:
        log.error(f"测试过程中发生错误: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 