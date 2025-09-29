"""
特征增强模块
使用matminer增加材料特征信息
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from pymatgen.core import Composition
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        Stoichiometry, ElementFraction,
        ElementProperty, ValenceOrbital
    )
    MATMINER_AVAILABLE = True
except ImportError:
    print("警告：matminer或pymatgen未安装，将跳过材料特征增强")
    MATMINER_AVAILABLE = False


class FeatureEnhancer:
    def __init__(self):
        """初始化特征增强器"""
        self.featurizer = None
        self.feature_labels = []
        self.log_file = "化学式解析日志.txt"
        
        # 初始化日志文件
        self._init_log_file()
        
        if MATMINER_AVAILABLE:
            self._setup_featurizer()
    
    def _init_log_file(self):
        """初始化日志文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=== 化学式解析日志 ===\n\n")
        except Exception as e:
            print(f"警告：无法创建日志文件 {self.log_file}: {e}")
    
    def _log_message(self, message):
        """记录消息到日志文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"警告：无法写入日志文件: {e}")
    
    def _setup_featurizer(self):
        """设置matminer特征提取器"""
        try:
            # 构造多个组分特征
            self.featurizer = MultipleFeaturizer([
                Stoichiometry(),
                ElementFraction(),
                ElementProperty.from_preset("magpie", impute_nan=True),
                ValenceOrbital()
            ])
            
            self.feature_labels = self.featurizer.feature_labels()
            print(f"✅ matminer特征提取器设置完成，总共 {len(self.feature_labels)} 个特征")
            
        except Exception as e:
            print(f"❌ 设置matminer特征提取器失败: {e}")
            self.featurizer = None
    
    def _simplify_formula_for_matminer(self, formula: str) -> str:
        """
        将复杂化学式简化为matminer可以处理的格式
        
        Args:
            formula: 复杂化学式
            
        Returns:
            简化后的化学式
        """
        try:
            from pymatgen.core import Composition
            import re
            
            working_formula = formula.strip()
            
            # 首先尝试直接解析
            try:
                comp = Composition(working_formula)
                result = comp.reduced_formula
                # 如果成功且没有括号，直接返回
                if '(' not in result:
                    message = f"🔄 简化化学式: '{formula}' → '{result}' (直接解析)"
                    print(message)
                    self._log_message(message)
                    return result
                else:
                    # 如果还有括号，记录但继续处理
                    message = f"🔄 pymatgen解析结果仍有括号: '{formula}' → '{result}'"
                    print(message)
                    self._log_message(message)
            except Exception as e:
                message = f"🔄 pymatgen直接解析失败: '{formula}' → {e}"
                print(message)
                self._log_message(message)
            
            # 强制展开所有括号 - 新的强力方法
            def force_expand_parentheses(formula_str):
                """强制展开所有括号，不依赖pymatgen"""
                max_iterations = 20
                iteration = 0
                
                while '(' in formula_str and iteration < max_iterations:
                    iteration += 1
                    old_formula = formula_str
                    
                    # 找到最内层括号
                    pattern = r'\(([^()]+)\)(\d*)'
                    
                    def expand_match(match):
                        content = match.group(1)  # 括号内容，如 CS2
                        multiplier = int(match.group(2)) if match.group(2) else 1  # 乘数，如 2
                        
                        # 解析括号内的元素
                        element_pattern = r'([A-Z][a-z]?)(\d*)'
                        elements = re.findall(element_pattern, content)
                        
                        expanded = ""
                        for element, count in elements:
                            count = int(count) if count else 1
                            new_count = count * multiplier
                            if new_count > 1:
                                expanded += f"{element}{new_count}"
                            else:
                                expanded += element
                        
                        return expanded
                    
                    # 替换所有匹配的括号
                    formula_str = re.sub(pattern, expand_match, formula_str)
                    
                    # 如果没有变化，强制移除括号
                    if formula_str == old_formula:
                        formula_str = re.sub(r'[()]', '', formula_str)
                        break
                
                # 最终清理
                formula_str = re.sub(r'[()]', '', formula_str)
                return formula_str
            
            # 强制展开括号
            expanded_formula = force_expand_parentheses(working_formula)
            
            # 处理小数系数
            decimal_pattern = r'([A-Z][a-z]?)(\d*\.\d+)'
            def round_decimal(match):
                element = match.group(1)
                decimal_val = float(match.group(2))
                rounded_val = max(1, round(decimal_val))
                return f"{element}{rounded_val}"
            
            expanded_formula = re.sub(decimal_pattern, round_decimal, expanded_formula)
            
            # 检查展开后的化学式是否有效
            try:
                # 测试pymatgen是否能解析展开后的化学式
                comp = Composition(expanded_formula)
                
                # 如果能解析，直接返回展开后的化学式（不使用pymatgen的reduced_formula，因为它可能重新引入括号）
                message = f"🔄 简化化学式: '{formula}' → '{expanded_formula}' (强制展开，跳过pymatgen标准化)"
                print(message)
                self._log_message(message)
                
                return expanded_formula
                
            except Exception as e:
                # 如果展开后的化学式无效，尝试其他方法
                message = f"⚠️ 展开后化学式无效: '{expanded_formula}' → {e}"
                print(message)
                self._log_message(message)
                
                # 作为最后的手段，返回原始化学式（至少比None好）
                return formula
            
        except Exception as e:
            message = f"⚠️ 无法简化化学式 {formula}: {e}"
            print(message)
            self._log_message(message)
            return None
    
    def _parse_composition_from_formula(self, formula: str) -> str:
        """
        从复杂化学式中解析出可用于matminer的组分
        对于异质结（/）和负载结构（@），尝试合并或保留主要组分
        
        Args:
            formula: 化学式字符串
            
        Returns:
            清理后的化学式
        """
        message = f"🔄 解析化学式: {formula}"
        print(message)
        self._log_message(message)
        
        # 特殊处理：单个C应该直接返回
        if formula.strip() == 'C':
            message = f"✅ 特殊处理单个C元素: {formula} → C"
            print(message)
            self._log_message(message)
            return 'C'
        
        # 移除常见的非标准符号和修饰词
        cleaned_formula = formula.replace('Tx', '').replace('multi-layered structure', '')
        cleaned_formula = cleaned_formula.replace('β-', '').replace('g-', '')
        cleaned_formula = cleaned_formula.replace('α-', '').replace('γ-', '').replace('δ-', '')
        cleaned_formula = cleaned_formula.replace('N-doped', '').replace('-doped', '')
        
        # 处理Unicode下标符号，转换为普通数字
        unicode_subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            'ₓ': 'x'  # 添加下标x的转换
        }
        for unicode_sub, normal_num in unicode_subscripts.items():
            cleaned_formula = cleaned_formula.replace(unicode_sub, normal_num)
        
        # 特殊处理分数化学式（如Mo4/3CTx）
        fraction_pattern = r'^([A-Z][a-z]?\d+)/(\d+)([A-Z][a-z]?(?:Tx|x)?)$'
        fraction_match = re.match(fraction_pattern, cleaned_formula)
        if fraction_match:
            element_part = fraction_match.group(1)  # Mo4
            denominator = fraction_match.group(2)   # 3
            second_part = fraction_match.group(3).replace('Tx', '').replace('x', '')  # C
            
            # 构造正确的化学式
            try:
                # 尝试构造 Mo4C3 这样的化学式
                formula_attempt = f"{element_part}{second_part}{denominator}"
                comp = Composition(formula_attempt)
                result = comp.reduced_formula
                message = f"🔄 分数化学式: '{original_component}' → '{result}'"
                print(message)
                self._log_message(message)
                return result
            except:
                pass
        
        # 特殊处理：如果化学式包含括号但不是异质结/负载结构，直接返回原始化学式
        if '(' in cleaned_formula and '/' not in cleaned_formula and '@' not in cleaned_formula:
            message = f"🔄 提取化学式: '{cleaned_formula}' → '{cleaned_formula}'"
            print(message)
            self._log_message(message)
            message = f"   ✅ 简单化学式: {cleaned_formula}"
            print(message)
            self._log_message(message)
            return cleaned_formula
        
        # 检查是否包含异质结或负载结构
        has_heterostructure = '/' in cleaned_formula
        has_loading = '@' in cleaned_formula
        
        if has_heterostructure or has_loading:
            message = f"   结构类型: 异质结={has_heterostructure}, 负载={has_loading}"
            print(message)
            self._log_message(message)
            
            # 优先级处理: @ > / > - > _ > · > &
            separators = ['@', '/', '-', '_', '·', '&']
            
            for sep in separators:
                if sep in cleaned_formula:
                    # 特殊检查：如果是分数化学式（如Mo4/3CTx），跳过/分割
                    if sep == '/' and re.match(r'^([A-Z][a-z]?\d+)/(\d+)([A-Z][a-z]?(?:Tx|x)?)$', cleaned_formula):
                        continue
                    parts = cleaned_formula.split(sep)
                    valid_compositions = []
                    
                    message = f"   分割符 '{sep}', 组分: {parts}"
                    print(message)
                    self._log_message(message)
                    
                    for part in parts:
                        part = part.strip()
                        if part:
                            comp_cleaned = self._clean_component(part)
                            if comp_cleaned:
                                try:
                                    # 测试是否可以被pymatgen解析
                                    comp = Composition(comp_cleaned)
                                    valid_compositions.append((comp_cleaned, comp))
                                    message = f"   ✅ 有效组分: {part} → {comp_cleaned}"
                                    print(message)
                                    self._log_message(message)
                                except Exception as e:
                                    message = f"   ❌ 无效组分: {part} → {comp_cleaned} ({e})"
                                    print(message)
                                    self._log_message(message)
                                    continue
                    
                    # 处理有效组分
                    if len(valid_compositions) >= 2:
                        # 检查是否都是真正的化学式（不是简写）
                        real_chemical_comps = []
                        for comp_str, comp_obj in valid_compositions:
                            # 化学式判断条件：
                            # 1. 包含数字的通常是化学式 (如 Ti3C2, Fe2O3)
                            # 2. 单个元素符号也是有效的 (如 C, Ni, Fe)
                            # 3. 长度>2且包含大小写混合的通常是化学式 (如 TiO2, MoS2)
                            is_chemical = (
                                any(c.isdigit() for c in comp_str) or  # 包含数字
                                (len(comp_str) <= 2 and comp_str.istitle()) or  # 元素符号 (C, Ti, Fe)
                                (len(comp_str) > 2 and any(c.islower() for c in comp_str))  # 大小写混合
                            )
                            
                            if is_chemical:
                                real_chemical_comps.append((comp_str, comp_obj))
                        
                        if len(real_chemical_comps) >= 2:
                            try:
                                # 智能合并策略 - 优先合并最重要的组分
                                # 如果有3个或更多组分，选择最重要的2-3个
                                if len(real_chemical_comps) >= 3:
                                    # 按复杂度排序，优先选择复杂的化学式
                                    sorted_comps = sorted(real_chemical_comps, key=lambda x: (
                                        len(x[0]),  # 长度
                                        sum(1 for c in x[0] if c.isdigit()),  # 数字个数
                                        x[0] != 'C'  # 不是简单碳
                                    ), reverse=True)
                                    
                                    # 选择前3个最重要的组分
                                    selected_comps = sorted_comps[:3]
                                    message = f"   📋 多组分选择: {[comp[0] for comp in selected_comps]} (从{len(real_chemical_comps)}个中选择)"
                                    print(message)
                                    self._log_message(message)
                                else:
                                    selected_comps = real_chemical_comps
                                
                                comp1_str, comp1 = selected_comps[0]
                                comp2_str, comp2 = selected_comps[1]
                                
                                # 特殊情况：如果其中一个是简单碳（C），优先保留更复杂的化学式
                                if comp1_str == 'C' and len(comp2_str) > 2:
                                    message = f"   ✅ 优先保留复杂组分: {comp2_str}"
                                    print(message)
                                    self._log_message(message)
                                    return comp2_str
                                elif comp2_str == 'C' and len(comp1_str) > 2:
                                    message = f"   ✅ 优先保留复杂组分: {comp1_str}"
                                    print(message)
                                    self._log_message(message)
                                    return comp1_str
                                
                                # 尝试合并前两个组分
                                merged_comp = comp1 + comp2
                                
                                # 如果有第三个重要组分，也尝试加入
                                if len(selected_comps) >= 3:
                                    comp3_str, comp3 = selected_comps[2]
                                    try:
                                        merged_comp = merged_comp + comp3
                                        merged_formula = merged_comp.reduced_formula
                                        message = f"   ✅ 三元合并 {comp1_str} + {comp2_str} + {comp3_str} → {merged_formula}"
                                        print(message)
                                        self._log_message(message)
                                        return merged_formula
                                    except Exception as e:
                                        message = f"   ⚠️ 三元合并失败，使用二元合并: {e}"
                                        print(message)
                                        self._log_message(message)
                                        # 继续使用二元合并
                                
                                merged_formula = merged_comp.reduced_formula
                                message = f"   ✅ 合并化学式 {comp1_str} + {comp2_str} → {merged_formula}"
                                print(message)
                                self._log_message(message)
                                return merged_formula
                                
                            except Exception as e:
                                message = f"   ❌ 合并失败: {e}"
                                print(message)
                                self._log_message(message)
                                # 退回到更复杂的化学式
                                sorted_comps = sorted(real_chemical_comps, key=lambda x: len(x[0]), reverse=True)
                                result = sorted_comps[0][0]
                                message = f"   ✅ 使用最复杂化学式: {result}"
                                print(message)
                                self._log_message(message)
                                return result
                        else:
                            # 如果没有足够的真正化学式，使用第一个有效组分
                            result = valid_compositions[0][0]
                            message = f"   ✅ 使用主要组分: {result}"
                            print(message)
                            self._log_message(message)
                            return result
                    
                    elif len(valid_compositions) == 1:
                        result = valid_compositions[0][0]
                        message = f"   ✅ 单一有效组分: {result}"
                        print(message)
                        self._log_message(message)
                        return result
                    
                    break  # 找到分隔符就停止
        
        # 如果没有特殊结构，直接清理
        final_cleaned = self._clean_component(cleaned_formula)
        if final_cleaned:
            try:
                comp = Composition(final_cleaned)
                message = f"   ✅ 简单化学式: {final_cleaned}"
                print(message)
                self._log_message(message)
                return final_cleaned
            except Exception as e:
                message = f"   ❌ 简单化学式解析失败: {e}"
                print(message)
                self._log_message(message)
        
        message = f"   ❌ 无法解析化学式: {formula}"
        print(message)
        self._log_message(message)
        return ""
    
    def _clean_component(self, component: str) -> str:
        """
        清理单个组分字符串
        
        Args:
            component: 组分字符串
            
        Returns:
            清理后的组分
        """
        if not component:
            return ""
        
        # 特殊处理：单个C应该直接返回，不被跳过
        if component.strip() == 'C':
            return 'C'
        
        # 在方法开头定义descriptive_words，确保所有代码路径都能访问
        descriptive_words = [
            'based', 'pattern', 'substrate', 'resistive', 'cotton', 
            'structure', 'stacked', 'accordion', 'kirigami', 'origami',
            'on', 'with', 'derived', 'blended', 'composite', 'layered',
            'multi', 'nanofiber', 'aerogel', 'hydrogel', 'matrix', 'interconnected',
            'doped', 'modified', 'treated', 'coated', 'supported', 'loaded',
            'enhanced', 'activated', 'functionalized', 'decorated'
        ]
        
        # 处理Unicode下标符号，转换为普通数字
        unicode_subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            'ₓ': 'x'  # 添加下标x的转换
        }
        for unicode_sub, normal_num in unicode_subscripts.items():
            component = component.replace(unicode_sub, normal_num)
        
        # 移除常见的非化学符号
        component = component.strip()
        original_component = component  # 保存原始值用于调试
        
        # 特殊材料名称到化学式的映射（扩展版）
        special_mappings = {
            'RGO': 'C',
            'rGO': 'C', 
            'MXene': 'Ti3C2',
            'Graphene': 'C',
            'CNF': 'C',
            'CNTs': 'C',
            'Carbon': 'C',
            'Polypyrrole': 'C4H4N',
            'PPy': 'C4H4N',
            'PANI': 'C6H4N',
            'Polyacrylonitrile': 'C3H3N',
            'PAN': 'C3H3N',
            'Polydopamine': 'C8H11NO2',
            'PDA': 'C8H11NO2',
            'Polyethylene terephthalate': 'C10H8O4',
            'PET': 'C10H8O4',
            'Melamine': 'C3H6N6',
            'Chitosan': 'C6H11NO4',
            'Cellulose': 'C6H10O5', # 纤维素
            'Carboxymethyl cellulose': 'C6H10O5',
            'TMOs': 'TiO2',
            'CNZF': 'CoNiZnFe2O4',
            'MMMs': 'Fe3O4',
            'MMFs': 'Fe3O4',
            'Natural ferrites': 'Fe2O3',
            'TiCx': 'TiC',
            'W-type nanoferrite': 'BaFe18O27',
            'nanoferrite': 'Fe2O3',
            'ferrite': 'Fe2O3',
            # 新增更多常见材料
            'GO': 'C8O2H2',  # 氧化石墨烯
            'graphene oxide': 'C8O2H2',
            'CNT': 'C',
            'SWCNT': 'C',
            'MWCNT': 'C',
            'graphite': 'C',
            'carbon black': 'C',
            'carbon fiber': 'C',
            'activated carbon': 'C',
            'carbon aerogel': 'C',
            # 常见氧化物
            'ITO': 'In2Sn3O8',
            'YSZ': 'Y2Zr2O7',
            'LTO': 'Li4Ti5O12',
            # 常见合金
            'steel': 'Fe',
            'stainless steel': 'Fe',
            'brass': 'CuZn',
            # 聚合物简化
            'PVDF': 'C2H2F2',
            'PTFE': 'C2F4',
            'PE': 'C2H4',
            'PP': 'C3H6',
            'PS': 'C8H8',
            'PMMA': 'C5H8O2',
            'PVA': 'C2H4O',  # 聚乙烯醇
            'PVB': 'C8H14O2', # 聚乙烯醇缩丁醛
            'PI': 'C22H10N2O5', # 聚酰亚胺
            'TPU': 'C3H6NO2',  # 热塑性聚氨酯
            'PAA': 'C3H4O2',   # 聚丙烯酸
            # 纤维材料
            'CF': 'C',  # 碳纤维
            'CNF': 'C', # 碳纳米纤维
            'CNTs': 'C', # 碳纳米管
            'BCNF': 'C', # 细菌纤维素纳米纤维
            # 其他常见材料简写
            'EP': 'C21H25ClO5P',  # 环氧树脂的一种
            'EGaIn': 'Ga',  # 液态金属合金，简化为Ga
            'SCA': 'SiO2',  # 硅胶
            'AS': '',   # 气凝胶支架，太复杂，跳过
            'FCI': '',  # 柔性导电互连，跳过
            'PW': '',   # 石蜡，跳过
            'Polyurethane foam': '',  # 聚氨酯泡沫，太复杂，跳过
            'polyurethane foam': '',  # 聚氨酯泡沫，太复杂，跳过
            'paraffin wax': '',  # 石蜡，太复杂，跳过
            'Silica': 'SiO2',  # 二氧化硅
            'silica': 'SiO2',  # 二氧化硅
            'Gelatin': '',  # 明胶，太复杂，跳过
            'gelatin': '',  # 明胶，太复杂，跳过
            'Aramid nanofibers': '',  # 芳纶纳米纤维，太复杂，跳过
            'aramid nanofibers': '',  # 芳纶纳米纤维，太复杂，跳过
            'Carbon': 'C',  # 碳
            'carbon': 'C',  # 碳
            'G': 'C',   # 石墨烯的简写
            'LDH': '',          # 层状双氢氧化物，太复杂，跳过
            # 常见简写需要跳过的材料
            'BNC': '',  # 硼氮碳，太复杂，跳过
            'NFC': '',  # 纳米纤维素，太复杂，跳过  
            'PDMS': '',  # 聚二甲基硅氧烷，太复杂，跳过
            'Graphene': 'C',  # 石墨烯
            'graphene': 'C',   # 石墨烯
            'PPyNFs': 'C4H4N',  # 聚吡咯纳米纤维
            'CTFE': 'C2ClF3',   # 氯三氟乙烯
            'CNTs': 'C',        # 碳纳米管
            'SWCNTs': 'C',      # 单壁碳纳米管
            'MWCNTs': 'C',      # 多壁碳纳米管
            'RGO': 'C',         # 还原氧化石墨烯
            'GO': 'C',          # 氧化石墨烯
            'MXe': '',          # 错误的MXene缩写，应该跳过
            'Mxe': '',          # 错误的MXene缩写，应该跳过
            'CNF': 'C',         # 碳纳米纤维
            'NCS': 'C',         # 氮掺杂碳球
            'SiCNWs': 'SiC',    # 碳化硅纳米线
            'CNWs': 'C',        # 碳纳米线
            'CFA': 'C',         # 碳纤维气凝胶
            'PINF': '',         # 聚合物，太复杂，跳过
            'Carbon': 'C',      # 碳
            'carbon': 'C',      # 碳
            'SWCNH': 'C',       # 单壁碳纳米角
            'NC': 'C',          # 氮掺杂碳
            'BNNB': '',         # 硼氮材料，太复杂，跳过
            'PVB': '',          # 聚乙烯醇缩丁醛，太复杂，跳过
            'wax': '',          # 石蜡，跳过
            'Gr': 'C',          # 石墨烯的错误缩写
            'Graphene': 'C',    # 石墨烯
            'Polyimide': '',    # 聚酰亚胺，太复杂，跳过
            'Polypyrrole': 'C4H4N',  # 聚吡咯
            'Aramid': '',       # 芳纶，太复杂，跳过
            'Epoxy': '',        # 环氧树脂，太复杂，跳过
            'SFMO': '',         # 复杂铁氧体，跳过
            'PMA': '',          # 聚甲基丙烯酸，太复杂，跳过
            'Nanofiber': '',    # 纳米纤维，跳过
            'Honeycomb': '',    # 蜂窝结构，跳过
            'Composite': '',    # 复合材料，跳过
            'Chitosan': 'C6H11NO4',  # 壳聚糖
            'BBCN': '',         # 硼碳氮材料，太复杂，跳过
            'MOF': '',          # 金属有机框架，太复杂，跳过
            'TiCx': 'TiC',      # 碳化钛
            'GQD': 'C',         # 石墨烯量子点
            'PAM': '',          # 聚丙烯酰胺，太复杂，跳过
            'Polyacrylamide': '', # 聚丙烯酰胺，太复杂，跳过
            'PUA': '',          # 聚氨酯丙烯酸酯，太复杂，跳过
            'IBOA': '',         # 异冰片基丙烯酸酯，太复杂，跳过
            'DEGDA': '',        # 二乙二醇二丙烯酸酯，太复杂，跳过
            'TPO': '',          # 光引发剂，太复杂，跳过
            'FR': '',           # 阻燃剂，太复杂，跳过
            'hydrogel': '',     # 水凝胶，跳过
            'WPU': '',          # 水性聚氨酯，太复杂，跳过
            'paraffin': '',     # 石蜡，跳过
            'FA': '',           # 糠醇或其他聚合物，太复杂，跳过
            'N-CNF': 'C',       # 氮掺杂碳纳米纤维
            'Glass': '',        # 玻璃，太复杂，跳过
            'Fiber': '',        # 纤维，跳过
            'NPC': 'C',         # 氮掺杂多孔碳
            'SA': '',           # 海藻酸钠或其他，太复杂，跳过
            'WPC': '',          # 木塑复合材料，太复杂，跳过
            'particles': '',    # 颗粒，描述词，跳过
            'aerogel': '',      # 气凝胶，描述词，跳过
            'derivative': '',   # 衍生物，描述词，跳过
            'hollow': '',       # 中空，描述词，跳过
            'PBA': '',          # 普鲁士蓝类似物，太复杂，跳过
            'Aramid': '',       # 芳纶，太复杂，跳过
            'fabric': '',       # 织物，描述词，跳过
            'Polyimide': '',    # 聚酰亚胺，太复杂，跳过
            'MPC': 'C',         # 介孔碳
            'CoNiMPC': '',      # 钴镍介孔碳，太复杂，跳过
            'ZIF': '',          # 沸石咪唑框架，太复杂，跳过
            'Calcined': '',     # 煅烧的，描述词，跳过
            'nanofibers': '',   # 纳米纤维，描述词，跳过
            'pz': '',           # 吡嗪或其他配体，太复杂，跳过
            'CN': '',           # 氰基或碳氮材料，太复杂，跳过
            'PEO': '',          # 聚环氧乙烷，太复杂，跳过
            'MF': '',           # 三聚氰胺泡沫，太复杂，跳过
            'PF': '',           # 酚醛泡沫，太复杂，跳过
            'SR': '',           # 硅橡胶，太复杂，跳过
            'polysiloxane': '', # 聚硅氧烷，太复杂，跳过
            'Polymer': '',      # 聚合物，太复杂，跳过
            'multilayer': '',   # 多层，描述词，跳过
            'Kevlar': '',       # 凯夫拉纤维，太复杂，跳过
            'nanofiber': '',    # 纳米纤维，描述词，跳过
            'Epoxy': '',        # 环氧树脂，太复杂，跳过
            'epoxy': '',        # 环氧树脂，太复杂，跳过
            'NCF': 'C',         # 氮掺杂碳纤维
            'Cnp': 'C',         # 碳纳米颗粒
            'SiCnw': 'SiC',     # 碳化硅纳米线
            'FCM': '',          # 材料编号，跳过
            'MQDs': 'C',        # MXene量子点
            'NCNTs': 'C',       # 氮掺杂碳纳米管
            'CuMnHS': '',       # 复杂硫化物，太复杂，跳过
            'TCNFs': 'C',       # 碳纳米纤维
            'SiCNWs': 'SiC',    # 碳化硅纳米线
            'Polyacrylamide': '', # 聚丙烯酰胺，太复杂，跳过
            'Glycerol': '',     # 甘油，太复杂，跳过
            'Water': '',        # 水，跳过
            'Gel': '',          # 凝胶，描述词，跳过
            'terephthalamide': '', # 对苯二甲酰胺，太复杂，跳过
            'phenylene': '',    # 苯撑，太复杂，跳过
            'PyC': 'C',         # 热解碳
            'EP': '',           # 环氧树脂，跳过
            'alloy': '',        # 合金，描述词，跳过
            'Melamine': '',     # 三聚氰胺，太复杂，跳过
            'Foam': '',         # 泡沫，描述词，跳过
            'Carbonized': '',   # 碳化的，描述词，跳过
            'CNF': 'C',         # 碳纳米纤维
            'FSF': '',          # 材料编号，跳过
            'CSMXene': 'Ti3C2', # 冷喷涂MXene
            'deficient': '',    # 缺陷的，描述词，跳过
            'NFC': 'C',         # 纳米纤维素
            'c-': '',           # 前缀c-，跳过（如c-NFC）
            '3C': 'C',          # 3C碳
            'PINF': '',         # 聚酰亚胺纳米纤维，太复杂，跳过
            'CFA': '',          # 煤粉灰，太复杂，跳过
            'N-doped': '',      # 氮掺杂，描述词，跳过
            'doped': '',        # 掺杂，描述词，跳过
            'Carbon': 'C',      # 碳
            'fabric': '',       # 织物，描述词，跳过
            'Graphene': 'C',    # 石墨烯
            'Aerogel': '',      # 气凝胶，描述词，跳过
            'Aramid': '',       # 芳纶，太复杂，跳过
            'Nanofiber': '',    # 纳米纤维，描述词，跳过
            'Honeycomb': '',    # 蜂窝，描述词，跳过
            'Composite': '',    # 复合材料，描述词，跳过
            'Epoxy': '',        # 环氧树脂，太复杂，跳过
            'acrylate': '',     # 丙烯酸酯，太复杂，跳过
            'SFMO': '',         # 复杂材料，跳过
            'PMA': '',          # 聚甲基丙烯酸，太复杂，跳过
            'Gelatin': '',      # 明胶，太复杂，跳过
            'wax': '',          # 蜡，跳过
            'Polyacrylamide': '', # 聚丙烯酰胺，太复杂，跳过
            'PAM': '',          # 聚丙烯酰胺，太复杂，跳过
            'hydrogel': '',     # 水凝胶，描述词，跳过
            'TPO': '',          # 光引发剂，太复杂，跳过
            'carbon': 'C',      # 碳
            'C': 'C',           # 碳元素
            'Glass': '',        # 玻璃，跳过
            'Fiber': '',        # 纤维，跳过
            'g': '',            # 前缀g，跳过
            '3D': '',           # 三维，描述词，跳过
            'Aramid': '',       # 芳纶，太复杂，跳过
            'fabric': '',       # 织物，描述词，跳过
            'particles': '',    # 颗粒，描述词，跳过
            'aerogel': '',      # 气凝胶，描述词，跳过
            'derivative': '',   # 衍生物，描述词，跳过
            'with': '',         # 与，介词，跳过
            'nanoribbons': '',  # 纳米带，描述词，跳过
            'PBA': '',          # 普鲁士蓝类似物，太复杂，跳过
            'PDA': 'C8H11NO2',  # 聚多巴胺
            'derived': '',      # 衍生的，描述词，跳过
            'TX': '',           # 终端基团变体，跳过
            'Et': '',           # 乙基，太复杂，跳过
            'DI': '',           # 去离子，描述词，跳过
            'oxidized': '',     # 氧化的，描述词，跳过
            'micro': '',        # 微观，描述词，跳过
            'antennas': '',     # 天线，描述词，跳过
            'decorated': '',    # 装饰的，描述词，跳过
            'turbostratic': '', # 乱层的，描述词，跳过
            'graphitized': '',  # 石墨化的，描述词，跳过
            'porous': '',       # 多孔的，描述词，跳过
            'MFPC': '',         # 材料编号，跳过
            'GN': 'C',          # 石墨烯纳米片
            'FSF': '',          # 材料编号，跳过
            'chain': '',        # 链，描述词，跳过
            'CMC': 'C6H10O5',   # 羧甲基纤维素
            'PVB': '',          # 聚乙烯醇缩丁醛，太复杂，跳过
            'doped': '',        # 掺杂的，描述词，跳过
            'Melamine': 'C3H6N6', # 三聚氰胺
            'Foam': '',         # 泡沫，描述词，跳过
            'Carbonized': 'C',  # 碳化的，映射为碳
            'CNF': 'C',         # 碳纳米纤维
            'Carbonized CNF': 'C', # 碳化的碳纳米纤维，映射为碳
            'HCF': '',          # 材料编号，跳过
            'oxides': '',       # 氧化物，描述词，跳过
            'deficient': '',    # 缺陷的，描述词，跳过
            'ND': 'C',          # 纳米金刚石
            'AC': 'C',          # 活性炭
            'ANF': '',          # 芳纶纳米纤维，太复杂，跳过
            'gelatine': '',     # 明胶，太复杂，跳过
            'ecoflex': '',      # 生态柔性材料，太复杂，跳过
            'composite': '',    # 复合材料，描述词，跳过
            'Double': '',       # 双层，描述词，跳过
            'layer': '',        # 层，描述词，跳过
            'Double-layer Ti3C2': 'Ti3C2', # 双层Ti3C2，映射为Ti3C2
            'mesoporous': '',   # 介孔的，描述词，跳过
            'polypyrrole': 'C4H4N', # 聚吡咯
            'SiCnw': 'SiC',     # 碳化硅纳米线
            'FCM': '',          # 材料编号，跳过
            'CuMnHS': '',       # 复杂材料，跳过
            'MQDs': 'C',        # 量子点，映射为碳
            'NCNTs': 'C',       # 氮掺杂碳纳米管
            'HFP': '',          # 六氟丙烯，太复杂，跳过
            'PVDF-HFP': '',     # 聚偏氟乙烯-六氟丙烯，太复杂，跳过
            'microsphere': '',  # 微球，描述词，跳过
            'microspheres': '', # 微球，描述词，跳过
            'nanocomposite': '', # 纳米复合材料，描述词，跳过
            'nanocomposites': '', # 纳米复合材料，描述词，跳过
            'PU': '',           # 聚氨酯，太复杂，跳过
            'multi': '',        # 多层的，描述词，跳过
            'layered': '',      # 层状的，描述词，跳过
            'structure': '',    # 结构，描述词，跳过
            'Elastomer': '',    # 弹性体，太复杂，跳过
            'Array': '',        # 阵列，描述词，跳过
            'MIL': '',          # MOF材料，太复杂，跳过
            'phases': '',       # 相，描述词，跳过
            'with': '',         # 与，介词，跳过
            'and': '',          # 和，介词，跳过
            'TX': '',           # 终端基团变体，跳过
            'Gel': '',          # 凝胶，描述词，跳过
            'Water': '',        # 水，跳过
            'Glycerol': '',     # 甘油，太复杂，跳过
            'FCI': '',          # 材料编号，跳过
            'AS': '',           # 材料编号，跳过
            'Polyurethane': '', # 聚氨酯，太复杂，跳过
            'foam': '',         # 泡沫，描述词，跳过
            'nanofibers': '',   # 纳米纤维，描述词，跳过
            'Carbon fabric': 'C', # 碳织物
            'carbon fabric': 'C', # 碳织物
            'Graphene Aerogel': 'C', # 石墨烯气凝胶
            'graphene aerogel': 'C', # 石墨烯气凝胶
            'Aramid Nanofiber': '', # 芳纶纳米纤维，太复杂，跳过
            'aramid nanofiber': '', # 芳纶纳米纤维，太复杂，跳过
            'Glass Fiber': '',  # 玻璃纤维，太复杂，跳过
            'glass fiber': '',  # 玻璃纤维，太复杂，跳过
            'Polymer multilayer': '', # 聚合物多层，太复杂，跳过
            'polymer multilayer': '', # 聚合物多层，太复杂，跳过
            'multilayer': '',   # 多层，描述词，跳过
            'PDA-derived': '',  # PDA衍生的，描述词，跳过
            'oxidized': '',     # 氧化的，描述词，跳过
            'derived': ''       # 衍生的，描述词，跳过
         }
        
        # 首先检查完全匹配的特殊材料映射
        for name, formula in special_mappings.items():
            if component.lower() == name.lower():
                if formula:  # 只返回非空的映射
                    message = f"🔄 特殊材料映射: '{original_component}' → '{formula}'"
                    print(message)
                    self._log_message(message)
                    return formula
                else:
                    message = f"🔍 跳过复杂材料: '{original_component}'"
                    print(message)
                    self._log_message(message)
                    return ""
        
        # 然后尝试提取化学式部分（如从Ti3CNTx-CoNi-Gelatin中提取Ti3CN）
        # 使用正则表达式匹配开头的化学式模式
        
        # 首先处理包含连字符的复合化学式（如Ti3C2Tx-NiCo2S4, PI-PDA-Ti3C2Tx-ZnO）
        if '-' in component and not any(word in component.lower() for word in ['based', 'doped', 'type', 'derived']):
            # 尝试分割并提取所有化学式部分
            parts = component.split('-')
            valid_parts = []
            for part in parts:
                part_cleaned = part.replace('Tx', '').replace('x', '').strip()
                if part_cleaned and len(part_cleaned) > 1:
                    # 先检查是否是已知材料
                    mapped_formula = None
                    should_skip = False
                    for name, formula in special_mappings.items():
                        if part_cleaned.lower() == name.lower():
                            if formula == '':  # 空字符串表示跳过
                                should_skip = True
                                break
                            elif formula:  # 非空映射
                                mapped_formula = formula
                                break
                    
                    if should_skip:
                        continue  # 跳过这个部分
                    
                    if mapped_formula:
                        try:
                            comp = Composition(mapped_formula)
                            valid_parts.append(mapped_formula)
                            continue
                        except:
                            pass
                    
                    # 尝试直接解析化学式
                    try:
                        comp = Composition(part_cleaned)
                        valid_parts.append(part_cleaned)
                    except:
                        continue
            
            if len(valid_parts) >= 3:
                try:
                    # 三元或多元合并：选择最重要的3个组分
                    sorted_parts = sorted(valid_parts, key=lambda x: (
                        len(x),  # 长度
                        sum(1 for c in x if c.isdigit()),  # 数字个数
                        x != 'C'  # 不是简单碳
                    ), reverse=True)[:3]
                    
                    comp1 = Composition(sorted_parts[0])
                    comp2 = Composition(sorted_parts[1])
                    comp3 = Composition(sorted_parts[2])
                    merged = comp1 + comp2 + comp3
                    result = merged.reduced_formula
                    message = f"🔄 连字符多元合并: '{original_component}' → '{result}' (从{sorted_parts})"
                    print(message)
                    self._log_message(message)
                    return result
                except Exception as e:
                    message = f"   ❌ 连字符多元合并失败: {e}"
                    print(message)
                    self._log_message(message)
            
            if len(valid_parts) >= 2:
                try:
                    # 二元合并
                    sorted_parts = sorted(valid_parts, key=lambda x: (len(x), sum(1 for c in x if c.isdigit())), reverse=True)
                    comp1 = Composition(sorted_parts[0])
                    comp2 = Composition(sorted_parts[1])
                    merged = comp1 + comp2
                    result = merged.reduced_formula
                    message = f"🔄 连字符二元合并: '{original_component}' → '{result}' (从{sorted_parts[:2]})"
                    print(message)
                    self._log_message(message)
                    return result
                except Exception as e:
                    message = f"   ❌ 连字符二元合并失败: {e}"
                    print(message)
                    self._log_message(message)
            
            if len(valid_parts) == 1:
                message = f"🔄 连字符化学式提取: '{original_component}' → '{valid_parts[0]}'"
                print(message)
                self._log_message(message)
                return valid_parts[0]
        
        # 处理包含花括号的复杂化学式（如{V V 10}）
        if '{' in component and '}' in component:
            # 移除花括号和罗马数字，只保留化学元素
            cleaned = component.replace('{', '').replace('}', '')
            # 移除罗马数字（IV, V等）
            cleaned = re.sub(r'\b(IV|V|VI|VII|VIII|IX|X|I{1,3})\b', '', cleaned)
            # 提取化学元素和数字
            elements = re.findall(r'[A-Z][a-z]?\d*', cleaned)
            if elements:
                try:
                    # 重新构造化学式
                    reconstructed = ''.join(elements)
                    comp = Composition(reconstructed)
                    result = comp.reduced_formula
                    message = f"🔄 花括号化学式: '{original_component}' → '{result}'"
                    print(message)
                    self._log_message(message)
                    return result
                except:
                    pass

        # 处理包含方括号的复杂化学式（如Fe4[Fe(CN)6]3）
        if '[' in component and ']' in component:
            try:
                # 尝试直接解析整个复杂化学式
                cleaned_bracket = component.replace('Tx', '').replace('x', '')
                comp = Composition(cleaned_bracket)
                result = comp.reduced_formula
                message = f"🔄 复杂化学式: '{original_component}' → '{result}'"
                print(message)
                self._log_message(message)
                return result
            except:
                # 如果失败，尝试提取方括号外的主要部分
                bracket_pattern = r'^([A-Za-z0-9]+)\['
                match = re.match(bracket_pattern, component)
                if match:
                    main_part = match.group(1)
                    try:
                        comp = Composition(main_part)
                        result = comp.reduced_formula
                        message = f"🔄 复杂化学式主体: '{original_component}' → '{result}'"
                        print(message)
                        self._log_message(message)
                        return result
                    except:
                        pass

        # 先检查是否包含MXene但被错误提取
        if 'MXene' in component or 'mxene' in component:
            # 直接映射MXene相关的材料
            for name, formula in special_mappings.items():
                if 'mxene' in name.lower() and formula:
                    message = f"🔄 MXene材料映射: '{original_component}' → '{formula}'"
                    print(message)
                    self._log_message(message)
                    return formula
            # 默认MXene映射
            message = f"🔄 默认MXene映射: '{original_component}' → 'Ti3C2'"
            print(message)
            self._log_message(message)
            return 'Ti3C2'

        # 尝试多种化学式模式
        patterns = [
            r'^([A-Z][a-z]?[\d\.]*)+(?:Tx|x)?$',  # 完整化学式，包含小数点
            r'^([A-Z][a-z]?\d*)+(?:Tx|x)?',  # 标准化学式如Ti3C2Tx
            r'^([A-Z][a-z]?\d*)+',  # 简单化学式如Gd2O3
        ]
        
        for pattern in patterns:
            match = re.match(pattern, component)
            if match:
                potential_formula = match.group().replace('Tx', '').replace('x', '')
                if len(potential_formula) > 1:  # 至少2个字符
                    try:
                        # 对于含有小数点的复杂化学式，先尝试直接解析
                        if '.' in potential_formula:
                            # 对于Ba1.8Sr0.2Co2Fe11.9Pr0.1O22这样的复杂化学式
                            # 直接使用原始化学式，让pymatgen处理
                            try:
                                comp = Composition(component.replace('Tx', '').replace('x', ''))
                                result = comp.reduced_formula
                                message = f"🔄 复杂小数化学式: '{original_component}' → '{result}'"
                                print(message)
                                self._log_message(message)
                                return result
                            except:
                                # 如果直接解析失败，尝试保留小数部分
                                try:
                                    # 移除括号并保留小数
                                    cleaned_decimal = component.replace('(', '').replace(')', '').replace('Tx', '').replace('x', '')
                                    comp = Composition(cleaned_decimal)
                                    result = comp.reduced_formula
                                    message = f"🔄 小数化学式(清理括号): '{original_component}' → '{result}'"
                                    print(message)
                                    self._log_message(message)
                                    return result
                                except:
                                    pass
                        else:
                            comp = Composition(potential_formula)
                            message = f"🔄 提取化学式: '{original_component}' → '{potential_formula}'"
                            print(message)
                            self._log_message(message)
                            return potential_formula
                    except Exception as e:
                        # 如果复杂化学式解析失败，尝试简化
                        if '.' in potential_formula:
                            try:
                                # 移除小数点后的数字，保留整数部分
                                simplified = re.sub(r'\d*\.\d+', '1', potential_formula)
                                comp = Composition(simplified)
                                result = comp.reduced_formula
                                message = f"🔄 简化复杂化学式: '{original_component}' → '{result}'"
                                print(message)
                                self._log_message(message)
                                return result
                            except:
                                continue
                        continue
        
        # 检查是否包含特殊材料名称（部分匹配）
        for name, formula in special_mappings.items():
            if name.lower() in component.lower() and len(name) > 2:  # 避免短名称误匹配
                if formula:  # 只返回非空的映射
                    message = f"🔄 包含特殊材料: '{original_component}' → '{formula}'"
                    print(message)
                    self._log_message(message)
                    return formula
                else:
                    message = f"🔍 跳过复杂材料: '{original_component}'"
                    print(message)
                    self._log_message(message)
                    return ""
        
        # 特殊处理复杂描述性名称
        if 'accordion-origami' in component.lower() or 'kirigami' in component.lower():
            # 这类复杂结构描述通常是基于某种基础材料
            if 'ma' in component.lower():  # MA可能指微波吸收材料
                return 'C'  # 通常是碳基材料
            return 'C'  # 默认为碳材料
        
        # 特殊处理包含括号的化学式（如(HfO2-Ti3C2Tx)-NiFe2O4）
        if '(' in component and ')' in component:
            # 提取括号内外的内容
            bracket_pattern = r'\(([^)]+)\)(.*)$'
            match = re.match(bracket_pattern, component)
            if match:
                inside_bracket = match.group(1)  # HfO2-Ti3C2Tx
                outside_bracket = match.group(2).lstrip('-')  # NiFe2O4
                
                # 处理括号内的复合化学式
                if '-' in inside_bracket:
                    parts = inside_bracket.split('-')
                    valid_parts = []
                    for part in parts:
                        part_clean = part.replace('Tx', '').replace('x', '').strip()
                        if part_clean:
                            try:
                                comp = Composition(part_clean)
                                valid_parts.append(part_clean)
                            except:
                                continue
                    
                    if len(valid_parts) >= 2 and outside_bracket:
                        try:
                            # 合并所有有效组分
                            comp1 = Composition(valid_parts[0])
                            comp2 = Composition(valid_parts[1])
                            if outside_bracket:
                                comp3 = Composition(outside_bracket)
                                merged = comp1 + comp2 + comp3
                            else:
                                merged = comp1 + comp2
                            result = merged.reduced_formula
                            message = f"🔄 括号化学式合并: '{original_component}' → '{result}'"
                            print(message)
                            self._log_message(message)
                            return result
                        except:
                            pass

        # 移除常见描述词和修饰词（更智能的清理）
        remove_words = [
            'Aerogel', 'aerogel', 'derived', 'composite', 'structure', 'Foam', 'foam',
            'fabric', 'Fabric', 'blended', 'doped', '-doped', 'N-doped',
            'Stacked', 'accordion-origami', 'kirigami', 'MA', 'PRS', 'type', 'nano'
        ]
        
        # 更智能的清理：保留化学式开头部分
        original_comp = component
        for word in remove_words:
            if word in component:
                # 尝试移除这个词
                temp_comp = component.replace(word, '').strip()
                if temp_comp and len(temp_comp) >= 2:
                    # 检查剩余部分是否包含化学元素特征
                    if any(c.isupper() for c in temp_comp) and not temp_comp.isalpha():
                        component = temp_comp
                    elif len(temp_comp) == 2 and temp_comp.istitle():  # 可能是元素符号
                        component = temp_comp
        
        # 移除空格和某些特殊字符，但保留化学式中的连字符
        component = component.replace(' ', '')
        
        # 特殊处理含有连字符的化学式（如Co-Ti3C2）
        
        if '-' in component and not any(word in component.lower() for word in descriptive_words):
            # 检查是否是化学式-化学式的组合
            parts = component.split('-')
            if len(parts) == 2:
                part1, part2 = parts
                # 检查两部分是否都可能是化学式
                if (re.match(r'^[A-Z][a-z]?\d*$', part1) and 
                    re.match(r'^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$', part2)):
                    try:
                        # 尝试合并两个化学式
                        comp1 = Composition(part1)
                        comp2 = Composition(part2.replace('Tx', '').replace('x', ''))
                        merged = comp1 + comp2
                        result = merged.reduced_formula
                        message = f"🔄 连字符化学式合并: '{original_component}' → '{result}'"
                        print(message)
                        self._log_message(message)
                        return result
                    except:
                        # 合并失败，选择更复杂的部分
                        if len(part2) > len(part1):
                            component = part2.replace('Tx', '').replace('x', '')
                            message = f"🔄 连字符化学式提取: '{original_component}' → '{component}'"
                            print(message)
                            self._log_message(message)
                            return component
                        else:
                            component = part1
                            message = f"🔄 连字符化学式提取: '{original_component}' → '{component}'"
                            print(message)
                            self._log_message(message)
                            return component
                else:
                    # 如果不是两个化学式，移除连字符
                    component = component.replace('-', '')
        
        component = component.replace('_', '')
        
        # 处理单个元素（如 Co, Ni, Fe, C）
        single_elements = ['Ti', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Al', 'Mo', 'Cr', 'Mn', 'V', 'C', 'N', 'O', 'S', 'Si', 'Nb']
        if component in single_elements:
            return component
        
        # 避免将复杂材料名称错误提取为单个元素
        # 如果组分包含明显的非化学词汇，跳过单字母提取
        non_chemical_indicators = [
            'aramid', 'epoxy', 'polymer', 'composite', 'nanofiber', 
            'honeycomb', 'acrylate', 'polyimide', 'polypyrrole'
        ]
        if any(indicator in component.lower() for indicator in non_chemical_indicators):
            message = f"🔍 跳过复杂材料: '{original_component}'"
            print(message)
            self._log_message(message)
            return ""
        
        # 如果组分太短但是单个元素，允许通过
        if len(component) < 2:
            if component in single_elements:
                return component
            return ""
        
        # 提取可能的化学式部分
        # 匹配化学式模式：元素符号+数字
        chem_pattern = r'[A-Z][a-z]?\d*'
        matches = re.findall(chem_pattern, component)
        
        if matches:
            # 重组化学式
            reconstructed = ''.join(matches)
            # 验证重组的化学式是否合理
            if len(reconstructed) >= 2 or reconstructed in single_elements:
                return reconstructed
        
        # 最后尝试：如果包含已知化学式片段
        known_compounds = [
            'TiO2', 'Fe2O3', 'Co3O4', 'NiO', 'CuS', 'SnS', 'CoO', 'SiC', 'TiC', 'MoS2',
            'Al2O3', 'SiO2', 'ZnO', 'CuO', 'MnO2', 'Cr2O3', 'V2O5', 'WO3', 'MoO3',
            'BaTiO3', 'SrTiO3', 'LaFeO3', 'BiFeO3', 'PbZrTiO3',
            'LiFePO4', 'LiCoO2', 'LiMn2O4', 'LiNiO2',
            'ZnS', 'CdS', 'PbS', 'Ag2S', 'Cu2S', 'FeS2',
            'Si3N4', 'AlN', 'BN', 'TiN', 'VN', 'CrN',
            'WC', 'TaC', 'HfC', 'ZrC', 'NbC', 'VC'
        ]
        
        for compound in known_compounds:
            if compound.lower() in component.lower():
                return compound
        
        # 特殊处理包含"fabric"的材料
        if 'fabric' in component.lower():
            if 'carbon' in component.lower():
                message = f"🔄 碳纤维材料: '{original_component}' → 'C'"
                print(message)
                self._log_message(message)
                return 'C'
            else:
                # 其他fabric材料跳过
                message = f"🔍 跳过纤维材料: '{original_component}'"
                print(message)
                self._log_message(message)
                return ""

        # 检查是否包含描述性词汇，如果是则跳过
        
        if any(word in component.lower() for word in descriptive_words):
            message = f"🔍 跳过描述性名称: '{original_component}'"
            print(message)
            self._log_message(message)
            return ""
        
        # 检查是否为无法识别的简写/缩写
        if len(component) <= 4 and component.isalpha() and component.isupper():
            # 检查是否在已知映射中
            found_in_mapping = False
            for name in special_mappings.keys():
                if name.lower() == component.lower():
                    found_in_mapping = True
                    break
            
            # 如果不在已知映射中，且不是常见元素符号，就跳过
            common_elements = ['C', 'N', 'O', 'S', 'H', 'Ti', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Al', 'Si', 'Mo', 'W', 'V', 'Cr', 'Mn']
            if not found_in_mapping and component not in common_elements:
                message = f"🔍 跳过未知简写: '{original_component}'"
                print(message)
                self._log_message(message)
                return ""
        
        # 验证最终结果
        final_result = component if len(component) >= 2 else ""
        
        # 调试信息
        if not final_result and original_component:
            print(f"🔍 调试: '{original_component}' → 清理后无效")
        elif final_result != original_component:
            print(f"🔄 转换: '{original_component}' → '{final_result}'")
        
        return final_result
    
    def _analyze_formula_issues(self, failed_formulas: List[str]):
        """
        分析化学式解析失败的原因
        
        Args:
            failed_formulas: 失败的化学式列表
        """
        print(f"\n🔍 分析解析失败的原因:")
        
        issue_categories = {
            'complex_descriptive': [],  # 复杂描述性名称
            'polymer_names': [],        # 聚合物名称
            'incomplete_formulas': [],  # 不完整的化学式
            'special_characters': [],   # 特殊字符
            'unknown_materials': []     # 未知材料
        }
        
        for formula in failed_formulas[:20]:  # 分析前20个
            formula_lower = formula.lower()
            
            if any(word in formula_lower for word in ['structure', 'stacked', 'layered', 'composite', 'derived']):
                issue_categories['complex_descriptive'].append(formula)
            elif any(word in formula_lower for word in ['poly', 'polymer', 'plastic', 'resin']):
                issue_categories['polymer_names'].append(formula)
            elif len(formula) < 3 or formula.isalpha():
                issue_categories['incomplete_formulas'].append(formula)
            elif any(char in formula for char in ['(', ')', '[', ']', '{', '}', ':', ';']):
                issue_categories['special_characters'].append(formula)
            else:
                issue_categories['unknown_materials'].append(formula)
        
        for category, formulas in issue_categories.items():
            if formulas:
                print(f"📋 {category}: {len(formulas)} 个")
                for formula in formulas[:3]:
                    print(f"     例: {formula}")
    
    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为数据增加matminer特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            增强特征后的DataFrame
        """
        if not MATMINER_AVAILABLE or self.featurizer is None:
            print("⚠️ matminer不可用，跳过特征增强")
            return df
        
        df_enhanced = df.copy()
        
        # 解析化学式
        print("🔄 解析化学式...")
        compositions = []
        valid_indices = []
        failed_formulas = []
        
        for idx, row in df_enhanced.iterrows():
            formula = row['formula']
            try:
                # 尝试直接解析
                comp_str = self._parse_composition_from_formula(formula)
                if comp_str:
                    # 进一步简化以确保matminer兼容性
                    simplified_str = self._simplify_formula_for_matminer(comp_str)
                    if simplified_str:
                        comp = Composition(simplified_str)
                        compositions.append(comp)  # 添加Composition对象，不是字符串
                        valid_indices.append(idx)
                        message = f"✅ 成功: {formula} → {comp_str} → {simplified_str}"
                        print(message)
                        self._log_message(message)
                    else:
                        failed_formulas.append(formula)
                        message = f"⚠️ 无法简化化学式: {formula} → {comp_str}"
                        print(message)
                        self._log_message(message)
                else:
                    failed_formulas.append(formula)
                    message = f"⚠️ 无法解析化学式: {formula}"
                    print(message)
                    self._log_message(message)
            except Exception as e:
                failed_formulas.append(formula)
                message = f"⚠️ 解析化学式失败 {formula}: {e}"
                print(message)
                self._log_message(message)
        
        print(f"\n📊 化学式解析统计:")
        print(f"✅ 成功解析: {len(compositions)} 个")
        print(f"❌ 解析失败: {len(failed_formulas)} 个")
        print(f"📈 成功率: {len(compositions)/(len(compositions)+len(failed_formulas))*100:.1f}%")
        
        # 显示成功解析的示例
        if compositions:
            print(f"\n✅ 成功解析示例:")
            for i, comp in enumerate(compositions[:5]):
                print(f"   {i+1}. {comp}")
            if len(compositions) > 5:
                print(f"   ... 还有 {len(compositions)-5} 个")
        
        if failed_formulas:
            print(f"\n❌ 失败的化学式示例:")
            for i, formula in enumerate(failed_formulas[:10]):  # 显示前10个失败案例
                print(f"   {i+1}. {formula}")
            if len(failed_formulas) > 10:
                print(f"   ... 还有 {len(failed_formulas)-10} 个")
            
            # 分析失败原因
            self._analyze_formula_issues(failed_formulas)
        
        if len(compositions) == 0:
            print("❌ 没有有效的化学式可以提取特征")
            return df_enhanced
        
        # 创建临时DataFrame用于特征提取
        temp_df = pd.DataFrame({
            'composition': compositions,
            'index': valid_indices
        })
        
        try:
            print("🔄 提取材料特征...")
            # 特征提取（跳过出错行）
            temp_df_features = self.featurizer.featurize_dataframe(
                temp_df.copy(), col_id="composition", ignore_errors=True
            )
            
            print(f"📊 特征提取结果:")
            print(f"   输入样本数: {len(temp_df)}")
            print(f"   输出样本数: {len(temp_df_features)}")
            print(f"   输出列数: {len(temp_df_features.columns)}")
            
            # 检查是否有完全空的行
            empty_rows = temp_df_features[temp_df_features.columns.difference(['composition', 'index'])].isna().all(axis=1).sum()
            print(f"   完全空的行数: {empty_rows}")
            
            # 移除composition列（已经不需要了）
            feature_columns = [col for col in temp_df_features.columns if col not in ['composition', 'index']]
            
            print(f"✅ 成功提取 {len(feature_columns)} 个材料特征")
            
            # 检查特征的有效性
            nan_counts = temp_df_features[feature_columns].isna().sum()
            total_samples = len(temp_df_features)
            
            print(f"\n📊 特征有效性分析:")
            print(f"   总样本数: {total_samples}")
            print(f"   总特征数: {len(feature_columns)}")
            
            # 计算有效率分布
            valid_rates = ((total_samples - nan_counts) / total_samples * 100).round(1)
            print(f"   特征有效率分布:")
            print(f"     100%有效: {(valid_rates == 100).sum()} 个特征")
            print(f"     90-99%有效: {((valid_rates >= 90) & (valid_rates < 100)).sum()} 个特征")
            print(f"     50-89%有效: {((valid_rates >= 50) & (valid_rates < 90)).sum()} 个特征")
            print(f"     10-49%有效: {((valid_rates >= 10) & (valid_rates < 50)).sum()} 个特征")
            print(f"     <10%有效: {(valid_rates < 10).sum()} 个特征")
            
            # 显示最差的几个特征
            worst_features = valid_rates.nsmallest(5)
            print(f"   最差的5个特征:")
            for feat, rate in worst_features.items():
                print(f"     {feat}: {rate}%有效")
            
            # 调整阈值：从90%降低到50%，如果还是没有就降到10%
            thresholds = [0.5, 0.1, 0.01]  # 50%, 10%, 1%
            valid_features = []
            
            for threshold in thresholds:
                valid_features = nan_counts[nan_counts < total_samples * (1 - threshold)].index.tolist()
                if len(valid_features) > 0:
                    print(f"   使用阈值 {threshold*100}%: 保留 {len(valid_features)} 个特征")
                    break
            
            if len(valid_features) == 0:
                print("   ⚠️ 即使使用1%阈值也没有有效特征，保留所有特征进行分析")
                valid_features = feature_columns.copy()
                
            print(f"💡 有效特征数量: {len(valid_features)} / {len(feature_columns)}")
            if len(valid_features) < len(feature_columns):
                print(f"⚠️ 丢弃了 {len(feature_columns) - len(valid_features)} 个无效特征")
            
            # 将特征添加到原始数据中
            for i, (_, row) in enumerate(temp_df_features.iterrows()):
                original_idx = valid_indices[i]
                for col in valid_features:  # 只使用有效特征
                    df_enhanced.loc[original_idx, col] = row[col]
            
            # 为没有特征的行填充NaN（只针对有效特征）
            for col in valid_features:
                if col not in df_enhanced.columns:
                    df_enhanced[col] = np.nan
            
            print(f"✅ 特征增强完成，数据形状: {df_enhanced.shape}")
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return df_enhanced
        
        return df_enhanced
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_labels if self.feature_labels else []
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建基础特征（当matminer不可用时）
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加基础特征的DataFrame
        """
        df_basic = df.copy()
        
        # 基于化学式创建一些简单特征
        df_basic['formula_length'] = df_basic['formula'].str.len()
        
        # 元素检测特征
        df_basic['has_metal'] = df_basic['formula'].str.contains('Ti|Fe|Co|Ni|Cu|Zn|Al|Mo|Cr|Mn|V', na=False).astype(int)
        df_basic['has_carbon'] = df_basic['formula'].str.contains('C', na=False).astype(int)
        df_basic['has_oxygen'] = df_basic['formula'].str.contains('O', na=False).astype(int)
        df_basic['has_sulfur'] = df_basic['formula'].str.contains('S', na=False).astype(int)
        df_basic['has_nitrogen'] = df_basic['formula'].str.contains('N', na=False).astype(int)
        
        # MXene相关特征
        df_basic['is_mxene'] = df_basic['formula'].str.contains('Ti3C2|Ti2C|V2C|Nb2C|Ti4N3|MXene', na=False).astype(int)
        df_basic['has_titanium'] = df_basic['formula'].str.contains('Ti', na=False).astype(int)
        
        # 材料类型特征
        df_basic['has_oxide'] = df_basic['formula'].str.contains('O2|O3|TiO2|Fe2O3|Al2O3|SiO2', na=False).astype(int)
        df_basic['has_sulfide'] = df_basic['formula'].str.contains('S2|S4|MoS2|WS2', na=False).astype(int)
        df_basic['has_carbide'] = df_basic['formula'].str.contains('TiC|SiC|Mo2C|WC', na=False).astype(int)
        df_basic['has_nitride'] = df_basic['formula'].str.contains('TiN|BN|Si3N4|AlN', na=False).astype(int)
        
        # 聚合物/有机物特征
        df_basic['has_polymer'] = df_basic['formula'].str.contains('PANI|PVDF|TPU|PAA|PTFE', na=False).astype(int)
        df_basic['has_carbon_material'] = df_basic['formula'].str.contains('CNT|graphene|carbon|PyC', na=False).astype(int)
        
        # 复合材料特征
        df_basic['is_composite'] = df_basic['formula'].str.contains('/', na=False).astype(int)
        df_basic['is_core_shell'] = df_basic['formula'].str.contains('@', na=False).astype(int)
        df_basic['has_multiple_phases'] = df_basic['formula'].str.contains('&|-', na=False).astype(int)
        
        # 计算分隔符数量（复杂程度指标）
        df_basic['separator_count'] = df_basic['formula'].str.count('[/@&-]')
        
        # 基于元素组成创建特征
        if 'elemental_composition' in df_basic.columns:
            df_basic['element_count'] = df_basic['elemental_composition'].str.split(',').str.len()
            df_basic['element_count'] = df_basic['element_count'].fillna(0)
            
            # 检查特定元素的存在
            for element in ['Ti', 'Fe', 'Co', 'Ni', 'Mo', 'C', 'O', 'S', 'N']:
                df_basic[f'has_element_{element}'] = df_basic['elemental_composition'].str.contains(element, na=False).astype(int)
        else:
            df_basic['element_count'] = 0
        
        # 数值化特征：提取数字信息
        def extract_numbers(formula):
            if pd.isna(formula):
                return []
            numbers = re.findall(r'\d+', str(formula))
            return [int(n) for n in numbers] if numbers else [0]
        
        # 统计化学式中的数字特征
        df_basic['number_count'] = df_basic['formula'].apply(lambda x: len(extract_numbers(x)))
        df_basic['max_number'] = df_basic['formula'].apply(lambda x: max(extract_numbers(x)) if extract_numbers(x) else 0)
        df_basic['sum_numbers'] = df_basic['formula'].apply(lambda x: sum(extract_numbers(x)))
        
        # 添加更多基础特征来补充matminer特征不足
        # 基于化学式字符的统计特征
        df_basic['uppercase_count'] = df_basic['formula'].str.count('[A-Z]')
        df_basic['lowercase_count'] = df_basic['formula'].str.count('[a-z]')
        df_basic['digit_count'] = df_basic['formula'].str.count('\d')
        df_basic['special_char_count'] = df_basic['formula'].str.count('[^A-Za-z0-9]')
        
        # 特定元素组合特征
        df_basic['has_transition_metal'] = df_basic['formula'].str.contains('Ti|Fe|Co|Ni|Cu|Cr|Mn|V|Mo|W', na=False).astype(int)
        df_basic['has_noble_metal'] = df_basic['formula'].str.contains('Au|Ag|Pt|Pd', na=False).astype(int)
        df_basic['has_rare_earth'] = df_basic['formula'].str.contains('La|Ce|Nd|Gd|Y', na=False).astype(int)
        
        # 化学键类型推断
        df_basic['likely_ionic'] = ((df_basic['has_metal'] == 1) & (df_basic['has_oxygen'] == 1)).astype(int)
        df_basic['likely_covalent'] = ((df_basic['has_carbon'] == 1) & (df_basic['has_nitrogen'] == 1)).astype(int)
        df_basic['likely_metallic'] = ((df_basic['has_metal'] == 1) & (df_basic['has_carbon'] == 0) & (df_basic['has_oxygen'] == 0)).astype(int)
        
        # 材料复杂度特征
        df_basic['complexity_score'] = (df_basic['element_count'] * 2 + 
                                       df_basic['separator_count'] * 3 + 
                                       df_basic['number_count'])
        
        # 基于已知材料性能的经验特征
        df_basic['high_performance_indicators'] = (
            df_basic['is_mxene'] * 3 +
            df_basic['has_sulfide'] * 2 +
            df_basic['has_carbon_material'] * 2 +
            df_basic['is_composite'] * 1
        )
        
        print(f"✅ 创建了增强基础特征，数据形状: {df_basic.shape}")
        print(f"📊 基础特征数量: {df_basic.shape[1] - df.shape[1]}")
        return df_basic


def enhance_dataset_features(df: pd.DataFrame, use_matminer: bool = True) -> pd.DataFrame:
    """
    增强数据集特征的便捷函数
    
    Args:
        df: 输入DataFrame
        use_matminer: 是否使用matminer（如果可用）
        
    Returns:
        特征增强后的DataFrame
    """
    enhancer = FeatureEnhancer()
    
    if use_matminer and MATMINER_AVAILABLE:
        return enhancer.enhance_features(df)
    else:
        return enhancer.create_basic_features(df)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('.')
    
    from data_processor import DataProcessor
    
    # 加载数据
    processor = DataProcessor("../json/all.json")
    df = processor.process_data()
    
    # 增强特征
    enhancer = FeatureEnhancer()
    df_enhanced = enhancer.enhance_features(df)
    
    print("\n特征增强测试完成！")
    print(f"原始数据形状: {df.shape}")
    print(f"增强后数据形状: {df_enhanced.shape}")
    print(f"新增特征数量: {df_enhanced.shape[1] - df.shape[1]}") 