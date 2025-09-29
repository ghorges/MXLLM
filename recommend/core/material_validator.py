"""
材料验证器 - 集成数据库查找和性能预测
优先从数据库查找真实数据，如果没有则使用预测系统
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import re

# 添加recommend根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
recommend_root = os.path.dirname(current_dir)
project_root = os.path.dirname(recommend_root)
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


class MaterialValidator:
    """材料验证器 - 查找真实数据或预测性能"""
    
    def __init__(self, data_loader=None):
        """
        初始化材料验证器
        
        Args:
            data_loader: 数据加载器实例
        """
        from .data_loader import DataLoader
        
        self.data_loader = data_loader or DataLoader()
        
        # 尝试导入预测系统
        self.predictor = None
        try:
            self._load_predictor()
        except Exception as e:
            logger.warning(f"⚠️ 预测器加载失败: {e}")
        
        # 加载材料数据缓存
        self.material_cache = {}
        try:
            self._load_material_data()
        except Exception as e:
            logger.warning(f"⚠️ 材料数据加载失败: {e}")
            # 即使材料数据加载失败，也要保证基本功能可用
    
    def _load_predictor(self):
        """加载PLS预测器"""
        try:
            # 导入根目录下的预测系统
            predict_path = os.path.join(project_root, 'predict')
            if predict_path not in sys.path:
                sys.path.insert(0, predict_path)
            
            from pls_predictor import PLSPredictor
            
            # 尝试加载已训练的模型
            model_path = os.path.join(predict_path, 'trained_pls_model.pkl')
            if os.path.exists(model_path):
                self.predictor = PLSPredictor()
                self.predictor.load_model(model_path)
                logger.info("✅ 成功加载PLS预测模型")
            else:
                logger.warning("⚠️ 未找到训练好的PLS模型，预测功能不可用")
                
        except Exception as e:
            logger.error(f"❌ 加载预测系统失败: {e}")
            self.predictor = None
    
    def _load_material_data(self):
        """加载并缓存材料数据"""
        try:
            # 加载所有MXene数据
            mxene_data = self.data_loader.get_mxene_data()
            
            if mxene_data:
                # 建立化学式到材料数据的映射
                for item in mxene_data:
                    formulas = self._extract_formulas_from_item(item)
                    for formula in formulas:
                        if formula not in self.material_cache:
                            self.material_cache[formula] = []
                        self.material_cache[formula].append(item)
                
                logger.info(f"✅ 缓存了{len(self.material_cache)}种化学式的材料数据")
            else:
                logger.warning("⚠️ 未找到材料数据，数据库查找功能不可用")
            
        except Exception as e:
            logger.error(f"❌ 加载材料数据失败: {e}")
            # 确保material_cache始终是字典类型
            self.material_cache = {}
    
    def _extract_formulas_from_item(self, item: Dict[str, Any]) -> List[str]:
        """从数据项中提取化学式"""
        formulas = []
        
        # 从不同字段提取化学式
        fields_to_check = ['chemical_formula', 'formula', 'title', 'abstract', 'content']
        
        for field in fields_to_check:
            if field in item and item[field]:
                extracted = self._extract_formulas_from_text(str(item[field]))
                formulas.extend(extracted)
        
        # 去重并返回
        return list(set(formulas))
    
    def _extract_formulas_from_text(self, text: str) -> List[str]:
        """从文本中提取化学式"""
        # MXene化学式模式
        patterns = [
            r'\b[A-Z][a-z]?\d*[A-Z][a-z]?\d*[A-Z][a-z]?\d*\b',  # 通用化学式
            r'\b(Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te)\d*[A-Z]\d*[A-Z]?\d*\b',  # MXene模式
        ]
        
        formulas = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            formulas.extend(matches)
        
        # 过滤掉过短或明显不是化学式的结果
        filtered = []
        for formula in formulas:
            if len(formula) >= 2 and any(c.isdigit() for c in formula):
                filtered.append(formula)
        
        return filtered
    
    def validate_material(self, formula: str, enable_prediction: bool = True) -> Dict[str, Any]:
        """
        验证材料性能
        
        Args:
            formula: 化学式
            enable_prediction: 是否启用预测功能
            
        Returns:
            包含材料信息和性能数据的字典
        """
        logger.info(f"🔍 验证材料: {formula}")
        
        # 1. 首先在数据库中查找
        db_result = self._search_in_database(formula)
        
        if db_result['found']:
            logger.info(f"✅ 在数据库中找到材料: {formula} - 使用实验数据，跳过预测")
            return {
                'formula': formula,
                'source': 'database',
                'found_in_db': True,
                'experimental_data': db_result['data'],
                'prediction': None,
                'confidence': 'high',  # 实验数据置信度高
                'summary': f"找到{len(db_result['data'])}条实验数据"
            }
        
        # 2. 如果数据库中没有，且启用预测，则使用预测系统
        elif enable_prediction:
            logger.info(f"💭 数据库中未找到{formula}，使用预测系统")
            prediction = self._predict_properties(formula)
            
            return {
                'formula': formula,
                'source': 'prediction',
                'found_in_db': False,
                'experimental_data': None,
                'prediction': prediction,
                'confidence': prediction.get('confidence', 'medium') if prediction else 'low',
                'summary': "使用AI预测模型预测" if prediction else "无法获取性能数据"
            }
        
        # 3. 如果数据库中没有，且不启用预测
        else:
            logger.info(f"📋 数据库中未找到{formula}，预测功能已禁用")
            return {
                'formula': formula,
                'source': 'none',
                'found_in_db': False,
                'experimental_data': None,
                'prediction': None,
                'confidence': 'low',
                'summary': "数据库中无数据，未进行预测"
            }
    
    def _search_in_database(self, formula: str) -> Dict[str, Any]:
        """在数据库中搜索材料"""
        try:
            # 标准化化学式
            normalized_formula = self._normalize_formula(formula)
            
            # 搜索匹配的材料
            found_materials = []
            
            # 精确匹配
            if formula in self.material_cache:
                found_materials.extend(self.material_cache[formula])
            
            # 标准化匹配
            if normalized_formula != formula and normalized_formula in self.material_cache:
                found_materials.extend(self.material_cache[normalized_formula])
            
            # 模糊匹配
            for cached_formula in self.material_cache.keys():
                if self._formulas_similar(formula, cached_formula):
                    found_materials.extend(self.material_cache[cached_formula])
            
            if found_materials:
                # 处理找到的材料数据
                processed_data = []
                for material in found_materials[:5]:  # 最多返回5条
                    processed_item = self._process_material_data(material)
                    if processed_item:
                        processed_data.append(processed_item)
                
                return {
                    'found': True,
                    'data': processed_data
                }
            
            return {'found': False, 'data': []}
            
        except Exception as e:
            logger.error(f"❌ 数据库搜索失败: {e}")
            return {'found': False, 'data': []}
    
    def _normalize_formula(self, formula: str) -> str:
        """标准化化学式"""
        # 移除空格和特殊字符
        normalized = re.sub(r'[^\w]', '', formula)
        return normalized
    
    def _formulas_similar(self, formula1: str, formula2: str) -> bool:
        """判断两个化学式是否相似"""
        # 简单的相似度判断
        norm1 = self._normalize_formula(formula1).lower()
        norm2 = self._normalize_formula(formula2).lower()
        
        # 完全匹配
        if norm1 == norm2:
            return True
        
        # 包含关系（用于处理不同写法）
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        return False
    
    def _process_material_data(self, material: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理材料数据，提取关键信息"""
        try:
            processed = {
                'title': material.get('title', '未知标题'),
                'doi': material.get('doi', ''),
                'source': material.get('source', '未知来源'),
                'properties': {},
                'synthesis_method': material.get('synthesis_method', ''),
                'testing_procedure': material.get('testing_procedure', ''),
                'abstract': material.get('abstract', '')[:200] + '...' if material.get('abstract') else ''
            }
            
            # 提取性能数据
            properties = {}
            
            # 查找EAB和RL数据
            content_text = str(material.get('content', '')) + ' ' + str(material.get('abstract', ''))
            
            # 提取数值型性能数据
            eab_match = re.search(r'EAB[:\s]*([0-9.]+)\s*GHz', content_text, re.IGNORECASE)
            if eab_match:
                properties['EAB'] = f"{eab_match.group(1)} GHz"
            
            rl_match = re.search(r'RL[:\s]*([0-9.-]+)\s*dB', content_text, re.IGNORECASE)
            if rl_match:
                properties['RL'] = f"{rl_match.group(1)} dB"
            
            # 查找其他性能指标
            thickness_match = re.search(r'thickness[:\s]*([0-9.]+)\s*mm', content_text, re.IGNORECASE)
            if thickness_match:
                properties['thickness'] = f"{thickness_match.group(1)} mm"
            
            processed['properties'] = properties
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ 处理材料数据失败: {e}")
            return None
    
    def _predict_properties(self, formula: str) -> Optional[Dict[str, Any]]:
        """使用预测系统预测材料性能"""
        if not self.predictor:
            logger.warning("⚠️ 预测系统不可用")
            return None
        
        try:
            # 使用PLS预测器预测
            prediction = self.predictor.predict_from_formula(formula)
            
            # 格式化预测结果
            result = {
                'rl_prediction': prediction['rl_prediction'],
                'eab_prediction': prediction['eab_prediction'],
                'rl_confidence': prediction['rl_confidence'],
                'eab_confidence': prediction['eab_confidence'],
                'rl_probabilities': prediction['rl_probabilities'],
                'eab_probabilities': prediction['eab_probabilities'],
                'confidence': min(prediction['rl_confidence'], prediction['eab_confidence'])  # 取较低的置信度
            }
            
            # 添加性能解释
            result['rl_meaning'] = self._interpret_rl_prediction(prediction['rl_prediction'])
            result['eab_meaning'] = self._interpret_eab_prediction(prediction['eab_prediction'])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            return None
    
    def _interpret_rl_prediction(self, rl_value) -> str:
        """解释RL预测结果"""
        # 根据预测值给出含义（需要根据实际模型调整）
        if str(rl_value) == '0':
            return "优秀 - 反射损耗 ≤ -50 dB，微波吸收效果优秀"
        elif str(rl_value) == '1':
            return "良好 - 反射损耗 -50 ~ -20 dB，微波吸收效果良好"
        elif str(rl_value) == '2':
            return "一般 - 反射损耗 -20 ~ -10 dB，微波吸收效果一般"
        else:
            return "差 - 反射损耗 > -10 dB，微波吸收效果不佳"
    
    def _interpret_eab_prediction(self, eab_value) -> str:
        """解释EAB预测结果"""
        # 根据预测值给出含义（需要根据实际模型调整）
        if str(eab_value) == '0':
            return "差 - 有效吸收带宽 ≤ 4 GHz，频带覆盖不足"
        elif str(eab_value) == '1':
            return "一般 - 有效吸收带宽 4-8 GHz，频带覆盖一般"
        elif str(eab_value) == '2':
            return "良好 - 有效吸收带宽 8-12 GHz，频带覆盖良好"
        else:
            return "优秀 - 有效吸收带宽 > 12 GHz，频带覆盖优秀"
    
    def batch_validate_materials(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """批量验证材料"""
        results = []
        for formula in formulas:
            try:
                result = self.validate_material(formula)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ 验证材料{formula}失败: {e}")
                results.append({
                    'formula': formula,
                    'source': 'error',
                    'found_in_db': False,
                    'experimental_data': None,
                    'prediction': None,
                    'confidence': 'low',
                    'summary': f"验证失败: {str(e)}"
                })
        
        return results
    
    def format_validation_result(self, result: Dict[str, Any]) -> str:
        """格式化验证结果为可读文本"""
        formula = result['formula']
        
        if result['found_in_db']:
            # 实验数据格式化
            data_list = result['experimental_data']
            formatted = f"🧪 **{formula}** (实验数据)\n\n"
            
            for i, data in enumerate(data_list[:3], 1):  # 最多显示3条
                formatted += f"**来源 {i}**: {data['title']}\n"
                if data['doi']:
                    formatted += f"**DOI**: {data['doi']}\n"
                
                if data['properties']:
                    formatted += "**性能数据**:\n"
                    for prop, value in data['properties'].items():
                        formatted += f"  • {prop}: {value}\n"
                
                if data['synthesis_method']:
                    formatted += f"**合成方法**: {data['synthesis_method']}\n"
                
                formatted += "\n"
            
        else:
            # 预测数据格式化
            formatted = f"🔮 **{formula}** (AI预测)\n\n"
            
            if result['prediction']:
                pred = result['prediction']
                formatted += "**性能预测**:\n"
                formatted += f"• **EAB**: {pred['eab_prediction']} ({pred['eab_meaning']})\n"
                formatted += f"• **RL**: {pred['rl_prediction']} ({pred['rl_meaning']})\n"
                formatted += f"• **置信度**: {pred['confidence']:.2f}\n\n"
                formatted += "⚠️ *此为AI预测结果，仅供参考*\n"
            else:
                formatted += "❌ 无法获取性能数据\n"
        
        return formatted 