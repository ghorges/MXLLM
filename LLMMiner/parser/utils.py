"""
工具函数模块

提供配置加载、AI模型调用等通用功能
"""

import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import logging
from logging.handlers import RotatingFileHandler
import sys

# LangChain导入
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# 初始化一个默认日志对象，确保log变量总是有效的
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("LLMMiner.parser.default")

# =====================================================================
# 临时逻辑：API密钥轮换管理 (之后需要删除)
# =====================================================================
# 全局API密钥列表 - 请填入您的API密钥
GEMINI_API_KEYS = [
    # 请在这里添加您的Gemini API密钥
    # "your-api-key-1",
    # "your-api-key-2", 
    # "your-api-key-3",
    # "AIzaSyAS95-FLdHTyvXmEs3TQ6F8iNgB9XstZDA",
    "AIzaSyA03aENIfFcE_cwbAhrr3kXWTxWZTL7nxc",
    "AIzaSyDn190pDhQ8w0NcE_Zin0B5hhecUOPYC_U",
    "AIzaSyBqdTdLqYTX9FO3l0f6bfVJBPHC5Fh1uFA",
    "AIzaSyDfaNHMzPfMHwcEyWpfZy08BB29rHFQw4Q",
]

# 全局计数器和索引
_api_call_counter = 0
_current_api_key_index = 0
_calls_per_key = 91  # 每个API密钥使用95次后切换

def get_rotated_gemini_api_key():
    """
    获取轮换的Gemini API密钥
    每调用95次后切换到下一个API密钥
    
    Returns:
        str: 当前使用的API密钥
    """
    global _api_call_counter, _current_api_key_index
    
    if not GEMINI_API_KEYS:
        # 如果没有设置全局密钥列表，返回None让原有逻辑处理
        return None
    
    # 检查是否需要切换API密钥
    if _api_call_counter >= _calls_per_key:
        _current_api_key_index = (_current_api_key_index + 1) % len(GEMINI_API_KEYS)
        _api_call_counter = 0
        log.info(f"切换到API密钥索引: {_current_api_key_index}")
    
    _api_call_counter += 1
    current_key = GEMINI_API_KEYS[_current_api_key_index]
    
    log.debug(f"使用API密钥索引 {_current_api_key_index}, 调用次数: {_api_call_counter}/{_calls_per_key}")
    
    return current_key
# =====================================================================

def setup_logging(log_level: str = "info", log_dir: str = None) -> logging.Logger:
    """
    设置日志系统
    Args:
        log_level: 日志级别，默认为info
        log_dir: 日志目录，如果为None则使用parser目录
    Returns:
        logging.Logger: 日志对象
    """
    global log
    # 直接返回已初始化的日志对象
    if len(log.handlers) > 0:
        return log
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    # 日志目录为parser目录
    current_dir = Path(__file__).parent.absolute()
    if log_dir is None:
        log_dir = current_dir
    else:
        log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    level = log_levels.get(log_level.lower(), logging.INFO)
    log.setLevel(level)
    # 添加文件处理器和控制台处理器
    log_file = os.path.join(log_dir, "parser.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    log.info("日志系统初始化完成")
    return log

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则自动查找
        
    Returns:
        Dict[str, Any]: 配置信息字典
    """
    try:
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent.parent
        
        # 如果没有指定配置路径，尝试多个可能的位置
        if config_path is None:
            possible_paths = [
                os.path.join(project_root, "config.yaml"),  # 项目根目录
                os.path.join(current_dir, "config.yaml"),   # 当前目录
                os.path.join(current_dir, "../../config.yaml"),  # 相对于当前目录的项目根目录
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
                    
            if config_path is None:
                if log:
                    log.warning("未找到配置文件，将使用默认配置")
                return {
                    "api_keys": {},
                    "models": {
                        "default": "deepseek",
                        "temperatures": {
                            "abstract_analysis": 0.0,
                            "field_extraction": 0.0,
                            "content_extraction": 0.0,
                            "content_evaluation": 0.0,
                            "content_optimization": 0.1
                        }
                    },
                    "system": {
                        "max_retries": 3,
                        "log_level": "info"
                    }
                }
        # 如果提供的是相对路径，转换为绝对路径
        elif not os.path.isabs(config_path):
            config_path = os.path.join(current_dir, config_path)
            
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 日志系统初始化
        log_level = config.get("system", {}).get("log_level", "info")
        setup_logging(log_level)
        # log.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        if log is None:
            setup_logging("info")
        log.error(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            "api_keys": {},
            "models": {
                "default": "deepseek",
                "temperatures": {
                    "abstract_analysis": 0.0,
                    "field_extraction": 0.0,
                    "content_extraction": 0.0,
                    "content_evaluation": 0.0,
                    "content_optimization": 0.1
                }
            },
            "system": {
                "max_retries": 3,
                "log_level": "info"
            }
        }

def get_temperature(stage: str) -> float:
    """
    获取特定处理阶段的温度参数
    
    Args:
        stage: 处理阶段名称，如 'abstract_analysis', 'field_extraction' 等
        
    Returns:
        float: 温度参数
    """
    config = load_config()
    temperatures = config.get("models", {}).get("temperatures", {})
    
    # 获取指定阶段的温度，如果不存在则使用默认值
    if stage in temperatures:
        return temperatures[stage]
    
    # 默认温度映射
    default_temps = {
        "abstract_analysis": 0.0,
        "field_extraction": 0.0,
        "content_extraction": 0.0,
        "content_evaluation": 0.0,
        "content_optimization": 0.1,
        "default": 0.0
    }
    
    return default_temps.get(stage, 0.0)

def get_max_retries() -> int:
    """
    获取最大重试次数
    
    Returns:
        int: 最大重试次数
    """
    config = load_config()
    return config.get("system", {}).get("max_retries", 3)

def get_llm(model_name: str = "kimi", temperature: float = 0.0) -> BaseChatModel:
    """
    获取指定的LLM模型实例
    
    Args:
        model_name: 模型名称，支持 "deepseek", "moonshot" 等
        temperature: 温度参数，控制随机性
        
    Returns:
        BaseChatModel: LangChain模型实例
    """
    # 加载配置
    config = load_config()
    
    # 根据模型名称选择相应的实现
    if model_name.lower() == "deepseek":
        api_key = config.get("api_keys", {}).get("deepseek")
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            
        if not api_key:
            raise ValueError("未找到DeepSeek API密钥，请在config.yaml中配置或设置环境变量")
            
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=temperature,
            api_key=api_key
        )
    elif model_name.lower() == "moonshot" or model_name.lower() == "kimi":
        api_key = config.get("api_keys", {}).get("moonshot")
        if not api_key:
            api_key = os.environ.get("MOONSHOT_API_KEY")
            
        if not api_key:
            raise ValueError("未找到Moonshot API密钥，请在config.yaml中配置或设置环境变量")
            
        # 使用ChatOpenAI类接入Moonshot
        model_version = config.get("models", {}).get("moonshot_version", "kimi-k2-0711-preview")
        return ChatOpenAI(
            model_name=model_version,
            openai_api_base="https://api.moonshot.cn/v1",
            openai_api_key=api_key,
            temperature=temperature
        )
    elif model_name.lower() in ["geimi", "gemini"]:
        
        # =====================================================================
        # 临时逻辑：优先使用轮换的API密钥 (之后需要删除)
        # =====================================================================
        api_key = get_rotated_gemini_api_key()
        # 如果环境变量中没有设置API密钥，设置它
        os.environ["GOOGLE_API_KEY"] = api_key

        # 如果没有设置全局密钥列表，则使用原有逻辑
        if not api_key:
            # 从配置文件或环境变量获取API密钥
            api_key = config.get("api_keys", {}).get("geimi")
            if not api_key:
                api_key = config.get("api_keys", {}).get("gemini")
            if not api_key:
                api_key = os.environ.get("GOOGLE_API_KEY")
        # =====================================================================
        # 检查最终是否有API密钥
        if not api_key:
            raise ValueError("未找到Google API密钥，请在config.yaml中配置或设置环境变量GOOGLE_API_KEY")
            
        # 使用ChatGoogleGenerativeAI类接入Gemini
        model_version = config.get("models", {}).get("geimi_version", "gemini-2.5-pro")
        max_tokens = config.get("models", {}).get("geimi_max_tokens", None)
        max_retries = config.get("system", {}).get("max_retries", 3)
        
        # 设置环境变量（确保总是使用最新的API密钥）
        if not api_key:
            log.error("无法获取有效的Google API密钥，请检查配置文件或环境变量")
            raise ValueError("Google API密钥不能为空")
        
        # 重要：清除旧的环境变量，设置新的环境变量
        # 注意：Gemini API只接受环境变量方式设置密钥
        os.environ["GOOGLE_API_KEY"] = api_key
        
        try:
            # 不直接传递API密钥，让库从环境变量获取
            return ChatGoogleGenerativeAI(
                model=model_version,
                api_key=api_key,
                convert_system_message_to_human=False,  # 确保正确处理系统消息
                disable_streaming=False,  # 启用流式输出 (注意：参数名是disable_streaming而不是streaming)
                timeout=1800.0  # 设置超时时间为30分钟 (1800秒)
            )
        except Exception as e:
            log.error(f"初始化ChatGoogleGenerativeAI失败: {e}")
            raise
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def extract_json_from_text(text: str) -> str:
    """
    从文本中提取JSON字符串，处理代码块标记
    
    支持以下格式:
    1. JSON对象: {}
    2. JSON数组: [{}, {}, ...]
    3. 代码块中的JSON: ```{} 或 ```[{}]
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        str: 提取出的JSON字符串
    """
    # 首先处理整个字符串可能被代码块包裹的情况
    text = text.strip()
    
    # 处理开头的代码块标记
    if text.startswith("```json"):
        text = text[7:].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
        
    # 处理结尾的代码块标记
    if text.endswith("```"):
        text = text[:-3].rstrip()
    
    # 同时检查对象和数组的起始位置
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    # 确定哪一个先出现 (如果都找到了)
    if first_brace != -1 and first_bracket != -1:
        # 哪个索引更小，哪个就是起始点
        if first_brace < first_bracket:
            # 对象在前
            last_brace = text.rfind('}')
            if last_brace > first_brace:
                object_content = text[first_brace:last_brace+1]
                try:
                    json.loads(object_content)
                    return object_content
                except json.JSONDecodeError:
                    # 尝试扩展搜索范围
                    try:
                        larger_content = text[first_brace:]
                        # 尝试找到有效的 JSON 结束点
                        brace_count = 0
                        for i, char in enumerate(larger_content):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    valid_json = larger_content[:i+1]
                                    json.loads(valid_json)
                                    return valid_json
                    except:
                        pass
        else:
            # 数组在前
            last_bracket = text.rfind(']')
            if last_bracket > first_bracket:
                array_content = text[first_bracket:last_bracket+1]
                try:
                    json.loads(array_content)
                    return array_content
                except json.JSONDecodeError:
                    # 尝试扩展搜索范围
                    try:
                        larger_content = text[first_bracket:]
                        # 尝试找到有效的 JSON 结束点
                        bracket_count = 0
                        for i, char in enumerate(larger_content):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    valid_json = larger_content[:i+1]
                                    json.loads(valid_json)
                                    return valid_json
                    except:
                        pass
    
    # 如果只有一种类型存在，或者两种都没找到，单独处理
    if first_brace != -1:
        last_brace = text.rfind('}')
        if last_brace > first_brace:
            object_content = text[first_brace:last_brace+1]
            try:
                json.loads(object_content)
                return object_content
            except json.JSONDecodeError:
                # 尝试扩展搜索
                pass
    
    if first_bracket != -1:
        last_bracket = text.rfind(']')
        if last_bracket > first_bracket:
            array_content = text[first_bracket:last_bracket+1]
            try:
                json.loads(array_content)
                return array_content
            except json.JSONDecodeError:
                # 尝试扩展搜索
                pass
    
    # 如果无法找到有效的JSON，尝试做进一步的处理
    log.warning(f"无法从文本中提取有效的JSON")
    
    # 作为最后的尝试，直接返回整个文本
    return text

def extract_list_from_text(text: str) -> str:
    """
    从文本中提取列表字符串，处理代码块标记
    
    支持以下格式:
    1. 列表: ["item1", "item2", ...]
    2. 代码块中的列表: ```["item1", ...] 
    
    Args:
        text: 包含列表的文本
        
    Returns:
        str: 提取出的列表字符串
    """
    # 首先处理整个字符串可能被代码块包裹的情况
    text = text.strip()
    
    # 处理开头的代码块标记
    if text.startswith("```json"):
        text = text[7:].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
        
    # 处理结尾的代码块标记
    if text.endswith("```"):
        text = text[:-3].rstrip()
    
    # 查找列表
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        list_content = text[first_bracket:last_bracket+1]
        try:
            # 尝试解析为JSON，验证是否有效
            json.loads(list_content)
            return list_content
        except json.JSONDecodeError:
            # 即使不是有效的JSON列表，仍返回最可能的列表部分
            return list_content
    
    # 没有找到列表结构，返回原始文本
    return text

def call_llm_with_json_output(system_prompt: str, 
                             user_prompt: str, 
                             model_name: str = "kimi",
                             temperature: float = 0.0,
                             max_retries: int = None) -> Dict[str, Any]:
    """
    调用LLM并解析JSON输出
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        model_name: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数，如果为None则使用配置文件中的值
        
    Returns:
        Dict[str, Any]: 解析后的JSON结果
    """
    # 如果未指定最大重试次数，从配置中获取
    if max_retries is None:
        max_retries = get_max_retries()
        
    llm = get_llm(model_name, temperature)
    
    for attempt in range(max_retries):
        try:
            # 调用LLM
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # 获取响应内容
            content = response.content
            # log.info(f"Extracted fields: {content}")
            # 提取JSON字符串
            json_str = extract_json_from_text(content)
            log.info(f"Extracted json_str fields: {json_str}")
            # 尝试解析JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    log.warning(f"JSON解析失败: {e}, 重试 ({attempt+1}/{max_retries})")
                    continue
                else:
                    log.error(f"JSON解析失败: {e}, 返回原始响应")
                    return {"error": "JSON解析失败", "raw_text": content}
                
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"调用LLM时发生错误: {e}, 重试 ({attempt+1}/{max_retries})")
                continue
            else:
                log.error(f"调用LLM失败: {e}")
                return {"error": f"调用LLM失败: {e}"}
    
    log.error("达到最大重试次数，调用LLM失败")
    return {"error": "达到最大重试次数，调用LLM失败"}

def call_llm_with_list_output(system_prompt: str, 
                             user_prompt: str, 
                             model_name: str = "deepseek",
                             temperature: float = 0.0,
                             max_retries: int = None) -> List[Any]:
    """
    调用LLM并解析列表输出
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        model_name: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数，如果为None则使用配置文件中的值
        
    Returns:
        List[Any]: 解析后的列表结果
    """
    # 如果未指定最大重试次数，从配置中获取
    if max_retries is None:
        max_retries = get_max_retries()
        
    llm = get_llm(model_name, temperature)
    
    for attempt in range(max_retries):
        try:
            # 调用LLM
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # 获取响应内容
            content = response.content
            
            # 提取列表字符串
            list_str = extract_list_from_text(content)
            
            # 尝试解析列表
            try:
                result = json.loads(list_str)
                if isinstance(result, list):
                    return result
                else:
                    return [result]
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    log.warning(f"列表解析失败: {e}, 重试 ({attempt+1}/{max_retries})")
                    continue
                else:
                    log.error(f"列表解析失败: {e}, 返回空列表")
                    return []
                
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"调用LLM时发生错误: {e}, 重试 ({attempt+1}/{max_retries})")
                continue
            else:
                log.error(f"调用LLM失败: {e}, 返回空列表")
                return []
    
    log.error("达到最大重试次数，调用LLM失败")
    return []

def call_llm_with_stream(system_prompt: str, 
                       user_prompt: str, 
                       model_name: str = "gemini",
                       temperature: float = 0.0,
                       max_retries: int = None):
    """
    流式调用LLM，适合交互式输出
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        model_name: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数，如果为None则使用配置文件中的值
        
    Returns:
        stream: 生成流对象
    """
    # 如果未指定最大重试次数，从配置中获取
    if max_retries is None:
        max_retries = get_max_retries()
        
    llm = get_llm(model_name, temperature)
    
    for attempt in range(max_retries):
        try:
            # 流式调用LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            return llm.stream(messages)
                
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"流式调用LLM时发生错误: {e}, 重试 ({attempt+1}/{max_retries})")
                continue
            else:
                log.error(f"流式调用LLM失败: {e}")
                raise
    
    log.error("达到最大重试次数，流式调用LLM失败")
    raise ValueError("达到最大重试次数，流式调用LLM失败")

def call_llm_with_json_stream(system_prompt: str, 
                             user_prompt: str, 
                             model_name: str = "gemini",
                             temperature: float = 0.0,
                             max_retries: int = None) -> Dict[str, Any]:
    """
    流式调用LLM并实时处理JSON输出，同时打印生成过程
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        model_name: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数，如果为None则使用配置文件中的值
        
    Returns:
        Dict[str, Any]: 解析后的JSON结果
    """
    # 如果未指定最大重试次数，从配置中获取
    if max_retries is None:
        max_retries = get_max_retries()
        
    llm = get_llm(model_name, temperature)
    
    for attempt in range(max_retries):
        try:
            # 打印状态信息
            print(f"\n正在使用{model_name}模型生成JSON...")
            
            # 流式调用LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            stream = llm.stream(messages)
            
            # 收集完整响应
            accumulated_text = ""
            content_started = False  # 标记是否已经开始显示内容
            
            # 使用一行来打印进度
            # print("生成中: ", end="", flush=True)
            for i, chunk in enumerate(stream):
                # 处理不同流式接口的兼容性
                if hasattr(chunk, 'text'):
                    if callable(chunk.text):  # 如果text是方法而不是属性
                        chunk_text = chunk.text()
                    else:  # 如果text是属性
                        chunk_text = chunk.text
                elif hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)  # 兜底方案
                
                accumulated_text += chunk_text
                
                # 如果还没开始显示内容，就显示进度
                if not content_started:
                    # 检查这个块是否包含JSON开始标记
                    if "{" in chunk_text or "[" in chunk_text:
                        # 清除进度显示行并换行
                        # print("\n\n生成的JSON内容:")
                        content_started = True
                    else:
                        # 只显示进度点
                        # print(".", end="", flush=True)
                        pass
                
                # 如果已经开始显示内容，则打印该块
                if content_started:
                    # sys.stdout.write(chunk_text)
                    # sys.stdout.flush()
                    pass
            
            if not content_started:
                # print("\n\n生成的JSON内容:")
                pass
                
            print("\n\n生成完成!")
            
            # 提取JSON字符串
            json_str = extract_json_from_text(accumulated_text)
            
            # 尝试美化打印JSON
            try:
                parsed_json = json.loads(json_str)
                # print("\n格式化的JSON结果:")
                # print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            except:
                pass  # 如果无法解析，则跳过美化打印
                
            log.info(f"Extracted json_str fields: {json_str}")
            
            # 尝试解析JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    log.warning(f"JSON解析失败: {e}, 重试 ({attempt+1}/{max_retries})")
                    continue
                else:
                    log.error(f"JSON解析失败: {e}, 返回原始响应")
                    return {"error": "JSON解析失败", "raw_text": accumulated_text}
                
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"调用LLM时发生错误: {e}, 重试 ({attempt+1}/{max_retries})")
                continue
            else:
                log.error(f"调用LLM失败: {e}")
                return {"error": f"调用LLM失败: {e}"}
    
    log.error("达到最大重试次数，调用LLM失败")
    return {"error": "达到最大重试次数，调用LLM失败"}

def normalize_formula(formula: str) -> str:
    """
    规范化化学式
    
    Args:
        formula: 原始化学式
        
    Returns:
        str: 规范化后的化学式
    """
    # 这里可以添加规范化化学式的逻辑
    # 例如将 "TiO2" 转换为 "TiO₂"
    return formula

def extract_metrics(text: str) -> Dict[str, Any]:
    """
    从文本中提取数值指标
    
    Args:
        text: 原始文本
        
    Returns:
        Dict[str, Any]: 提取的指标
    """
    # 这里可以添加提取数值指标的逻辑
    # 例如从 "反射损耗为-45.6 dB" 提取 {"reflection_loss": -45.6, "unit": "dB"}
    return {} 