"""
数据加载器模块
负责从JSON文件中加载MXene相关的数据
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类，用于加载和预处理MXene数据"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，默认为项目根目录的json和Data目录
        """
        if data_dir is None:
            # 默认使用recommend目录下的data目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            recommend_root = os.path.dirname(current_dir)
            self.json_dir = os.path.join(recommend_root, "data")
            self.data_dir = os.path.join(recommend_root, "data")
        else:
            self.json_dir = data_dir
            self.data_dir = data_dir
            
        self.data_cache = {}
        
    def load_all_json(self) -> Optional[List[Dict[str, Any]]]:
        """
        加载all.json文件
        
        Returns:
            包含所有原始数据的列表
        """
        cache_key = "all_json"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        all_json_path = os.path.join(self.json_dir, "all.json")
        
        try:
            with open(all_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data_cache[cache_key] = data
                logger.info(f"成功加载all.json，包含{len(data)}条记录")
                return data
        except FileNotFoundError:
            logger.error(f"未找到文件: {all_json_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            logger.error(f"加载all.json时发生错误: {e}")
            return None
    
    def load_extracted_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        加载所有的extracted.json文件
        
        Returns:
            字典，键为数据源名称，值为对应的数据列表
        """
        cache_key = "extracted_data"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        extracted_files = [
            "acs_extracted.json",
            "elsevier_extracted.json", 
            "rsc_extracted.json",
            "springer_extracted.json",
            "wiley_extracted.json"
        ]
        
        extracted_data = {}
        
        for filename in extracted_files:
            filepath = os.path.join(self.data_dir, filename)
            source_name = filename.replace("_extracted.json", "")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    extracted_data[source_name] = data
                    logger.info(f"成功加载{filename}，包含{len(data)}条记录")
            except FileNotFoundError:
                logger.warning(f"未找到文件: {filepath}")
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误 {filename}: {e}")
                continue
            except Exception as e:
                logger.error(f"加载{filename}时发生错误: {e}")
                continue
        
        self.data_cache[cache_key] = extracted_data
        return extracted_data
    
    def get_mxene_data(self) -> List[Dict[str, Any]]:
        """
        获取所有MXene相关的数据，现在只从data/all_mxene.json读取
        
        Returns:
            包含化学式、合成工艺、测试流程的数据列表
        """
        cache_key = "mxene_data"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # 从data/all_mxene.json加载数据
        mxene_data = self.load_all_mxene_json()
        if not mxene_data:
            logger.warning("未能加载all_mxene.json数据")
            return []
        
        # 处理数据
        processed_data = []
        for item in mxene_data:
            if self._is_mxene_related(item):
                processed_item = self._process_raw_data(item)
                if processed_item:
                    processed_data.append(processed_item)
        
        # 去重
        seen_dois = set()
        unique_data = []
        for item in processed_data:
            doi = item.get('doi', '')
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                unique_data.append(item)
            elif not doi:  # 没有DOI的项目也添加
                unique_data.append(item)
        
        self.data_cache[cache_key] = unique_data
        logger.info(f"成功处理MXene数据，包含{len(unique_data)}条去重记录")
        return unique_data
    
    def load_all_mxene_json(self) -> Optional[List[Dict[str, Any]]]:
        """
        加载data/all_mxene.json文件
        
        Returns:
            包含所有MXene数据的列表
        """
        cache_key = "all_mxene_json"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        all_mxene_path = os.path.join(self.data_dir, "all_mxene.json")
        
        try:
            with open(all_mxene_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data_cache[cache_key] = data
                logger.info(f"成功加载all_mxene.json，包含{len(data)}条记录")
                return data
        except FileNotFoundError:
            logger.error(f"未找到文件: {all_mxene_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            logger.error(f"加载all_mxene.json时发生错误: {e}")
            return None
    
    def _is_mxene_related(self, item: Dict[str, Any]) -> bool:
        """
        判断数据项是否与MXene相关
        
        Args:
            item: 数据项
            
        Returns:
            是否与MXene相关
        """
        text_fields = []
        
        # 获取所有可能包含文本的字段
        if 'abstract' in item:
            text_fields.append(str(item['abstract']).lower())
        if 'text' in item:
            if isinstance(item['text'], list):
                text_fields.extend([str(t).lower() for t in item['text']])
            else:
                text_fields.append(str(item['text']).lower())
        if 'title' in item:
            text_fields.append(str(item['title']).lower())
        if 'name' in item:
            text_fields.append(str(item['name']).lower())
        
        # 检查MXene相关关键词
        mxene_keywords = [
            'mxene', 'mxenes', 'ti3c2', 'ti2c', 'v2c', 'nb2c', 'ta2c',
            'microwave absorption', 'electromagnetic wave', 'emw', 'absorption',
            'electromagnetic interference', 'emi', 'shielding'
        ]
        
        combined_text = ' '.join(text_fields)
        return any(keyword in combined_text for keyword in mxene_keywords)
    
    def _process_raw_data(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理原始数据项
        
        Args:
            item: 原始数据项
            
        Returns:
            处理后的数据项
        """
        try:
            processed = {
                'doi': item.get('doi', ''),
                'publisher': item.get('publisher', ''),
                'title': item.get('name', ''),
                'abstract': item.get('abstract', ''),
                'content': item.get('text', []),
                'source': 'raw_data',
                'chemical_formula': self._extract_chemical_formula(item),
                'synthesis_method': self._extract_synthesis_method(item),
                'testing_procedure': self._extract_testing_procedure(item)
            }
            return processed
        except Exception as e:
            logger.error(f"处理原始数据时出错: {e}")
            return None
    
    def _process_extracted_data(self, item: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        处理提取数据项
        
        Args:
            item: 提取数据项
            source: 数据源名称
            
        Returns:
            处理后的数据项
        """
        try:
            processed = {
                'doi': item.get('doi', ''),
                'publisher': source,
                'title': item.get('title', ''),
                'abstract': item.get('abstract', ''),
                'content': item.get('content', item.get('text', [])),
                'source': f'extracted_{source}',
                'chemical_formula': item.get('chemical_formula', ''),
                'synthesis_method': item.get('synthesis_method', ''),
                'testing_procedure': item.get('testing_procedure', '')
            }
            return processed
        except Exception as e:
            logger.error(f"处理提取数据时出错: {e}")
            return None
    
    def _extract_chemical_formula(self, item: Dict[str, Any]) -> str:
        """
        从数据项中提取化学式信息
        """
        # 简单的化学式提取逻辑，可以根据需要扩展
        content = str(item.get('abstract', '')) + ' ' + str(item.get('text', ''))
        
        # 常见的MXene化学式模式
        import re
        formula_patterns = [
            r'\bTi\d*C\d*T?x?\b',
            r'\bV\d*C\d*T?x?\b', 
            r'\bNb\d*C\d*T?x?\b',
            r'\bMo\d*C\d*T?x?\b',
            r'\bTa\d*C\d*T?x?\b'
        ]
        
        formulas = []
        for pattern in formula_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            formulas.extend(matches)
        
        return ', '.join(list(set(formulas))) if formulas else ''
    
    def _extract_synthesis_method(self, item: Dict[str, Any]) -> str:
        """
        从数据项中提取合成方法信息
        """
        content = str(item.get('abstract', '')) + ' ' + str(item.get('text', ''))
        
        # 合成方法相关关键词
        synthesis_keywords = [
            'hydrothermal', 'solvothermal', 'chemical vapor deposition', 'cvd',
            'etching', 'hf etching', 'molten salt', 'mechanical exfoliation',
            'sonication', 'ultrasonic', 'synthesis', 'preparation'
        ]
        
        methods = []
        for keyword in synthesis_keywords:
            if keyword.lower() in content.lower():
                methods.append(keyword)
        
        return ', '.join(methods) if methods else ''
    
    def _extract_testing_procedure(self, item: Dict[str, Any]) -> str:
        """
        从数据项中提取测试流程信息
        """
        content = str(item.get('abstract', '')) + ' ' + str(item.get('text', ''))
        
        # 测试方法相关关键词
        testing_keywords = [
            'xrd', 'x-ray diffraction', 'sem', 'tem', 'transmission electron microscopy',
            'scanning electron microscopy', 'xps', 'x-ray photoelectron spectroscopy',
            'raman spectroscopy', 'ftir', 'uv-vis', 'network analyzer',
            'reflection loss', 'electromagnetic measurement', 'impedance',
            'microwave absorption', 'vector network analyzer', 'vna'
        ]
        
        tests = []
        for keyword in testing_keywords:
            if keyword.lower() in content.lower():
                tests.append(keyword)
        
        return ', '.join(tests) if tests else '' 