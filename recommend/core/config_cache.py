"""
配置缓存模块
负责保存和加载API配置信息
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import base64

logger = logging.getLogger(__name__)


class ConfigCache:
    """配置缓存管理器"""
    
    def __init__(self):
        """初始化配置缓存"""
        # 配置文件路径
        self.config_dir = Path.home() / ".mxene_recommender"
        self.config_file = self.config_dir / "config.json"
        
        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info(f"配置文件路径: {self.config_file}")
    
    def save_config(self, config: Dict[str, str]) -> bool:
        """
        保存配置到本地文件
        
        Args:
            config: 包含api_key, api_base, model的配置字典
            
        Returns:
            是否保存成功
        """
        try:
            # 对敏感信息进行简单编码（非加密，只是避免明文显示）
            encoded_config = {
                'api_key': self._encode_sensitive(config.get('api_key', '')),
                'api_base': config.get('api_base', ''),
                'model': config.get('model', ''),
                '_version': '1.0'
            }
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(encoded_config, f, indent=2, ensure_ascii=False)
            
            logger.info("配置已保存到本地")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def load_config(self) -> Optional[Dict[str, str]]:
        """
        从本地文件加载配置
        
        Returns:
            配置字典，如果加载失败返回None
        """
        try:
            if not self.config_file.exists():
                logger.info("配置文件不存在")
                return None
            
            # 读取配置文件
            with open(self.config_file, 'r', encoding='utf-8') as f:
                encoded_config = json.load(f)
            
            # 解码配置
            config = {
                'api_key': self._decode_sensitive(encoded_config.get('api_key', '')),
                'api_base': encoded_config.get('api_base', 'https://api.openai.com/v1'),
                'model': encoded_config.get('model', 'gpt-3.5-turbo')
            }
            
            # 验证配置完整性
            if config['api_key'] and config['api_base'] and config['model']:
                logger.info("成功加载本地配置")
                return config
            else:
                logger.warning("配置文件不完整")
                return None
                
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return None
    
    def clear_config(self) -> bool:
        """
        清除本地配置
        
        Returns:
            是否清除成功
        """
        try:
            if self.config_file.exists():
                self.config_file.unlink()
                logger.info("本地配置已清除")
            return True
        except Exception as e:
            logger.error(f"清除配置失败: {e}")
            return False
    
    def _encode_sensitive(self, text: str) -> str:
        """
        对敏感信息进行简单编码
        
        Args:
            text: 原始文本
            
        Returns:
            编码后的文本
        """
        if not text:
            return ''
        
        try:
            # 使用base64编码（非加密，只是避免明文显示）
            encoded_bytes = base64.b64encode(text.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"编码失败: {e}")
            return text
    
    def _decode_sensitive(self, encoded_text: str) -> str:
        """
        解码敏感信息
        
        Args:
            encoded_text: 编码后的文本
            
        Returns:
            解码后的文本
        """
        if not encoded_text:
            return ''
        
        try:
            # 使用base64解码
            decoded_bytes = base64.b64decode(encoded_text.encode('utf-8'))
            return decoded_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"解码失败: {e}")
            return encoded_text
    
    def has_config(self) -> bool:
        """
        检查是否存在有效配置
        
        Returns:
            是否存在有效配置
        """
        config = self.load_config()
        return config is not None and bool(config.get('api_key'))


# 全局配置缓存实例
config_cache = ConfigCache() 