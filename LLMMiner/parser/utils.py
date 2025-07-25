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

# LangChain导入
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import ChatOpenAI

# 日志初始化
log = None

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
    if log is not None:
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
    logger = logging.getLogger("LLMMiner.parser")
    level = log_levels.get(log_level.lower(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
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
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    log = logger
    log.info("日志系统初始化完成")
    return logger

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
        log.info(f"成功加载配置文件: {config_path}")
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
    
    # 检查是对象还是数组
    # 先查找JSON数组 - 形如 [{"key": value}, {"key": value}, ...]
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    
    # 如果找到了可能的数组边界，检查内部是否包含对象
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        array_content = text[first_bracket:last_bracket+1]
        # 检查数组内是否包含花括号，这表示可能是对象数组
        if '{' in array_content and '}' in array_content:
            try:
                # 尝试解析为JSON，验证是否有效
                json.loads(array_content)
                return array_content
            except json.JSONDecodeError:
                # 不是有效的JSON数组，继续检查其他可能性
                pass
    
    # 查找单个JSON对象
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        object_content = text[first_brace:last_brace+1]
        try:
            # 尝试解析为JSON，验证是否有效
            json.loads(object_content)
            return object_content
        except json.JSONDecodeError:
            # 不是有效的JSON对象
            pass
    
    # 如果上述尝试都失败，返回最可能的JSON部分（即使可能无效）
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        return text[first_bracket:last_bracket+1]
    elif first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return text[first_brace:last_brace+1]
    
    # 无法提取任何可能的JSON
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
            
            # 提取JSON字符串
            json_str = extract_json_from_text(content)
            
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