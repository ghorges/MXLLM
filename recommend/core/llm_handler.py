"""
LLM处理器模块
负责与OpenAI API的交互和推荐生成
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMHandler:
    """LLM处理器，负责与OpenAI API交互"""
    
    def __init__(self, api_key: str = None):
        """
        初始化LLM处理器
        
        Args:
            api_key: OpenAI API密钥
        """
        self.api_key = api_key
        self.client = None
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str) -> bool:
        """
        设置API密钥（保持兼容性）
        
        Args:
            api_key: OpenAI API密钥
            
        Returns:
            是否设置成功
        """
        return self.set_api_config({
            'api_key': api_key,
            'api_base': 'https://api.openai.com/v1',
            'model': 'gpt-3.5-turbo'
        })
    
    def set_api_config(self, config: dict) -> bool:
        """
        设置API配置
        
        Args:
            config: 包含api_key, api_base, model的配置字典
            
        Returns:
            是否设置成功
        """
        try:
            self.api_key = config['api_key']
            self.api_base = config['api_base']
            self.model = config['model']
            
            # 处理API地址
            if self.api_base.endswith('/v1'):
                base_url = self.api_base
            else:
                base_url = self.api_base.rstrip('/') + '/v1'
            
            # 创建客户端
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
            
            # 测试API连接
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10,
                timeout=10
            )
            
            logger.info(f"AI服务连接成功 - 模型: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"设置AI配置失败: {e}")
            self.client = None
            return False
    
    def is_api_ready(self) -> bool:
        """
        检查API是否就绪
        
        Returns:
            API是否就绪
        """
        return self.client is not None
    
    def chat_with_recommendations(self, 
                                user_message: str,
                                recommendations: Dict[str, Any] = None,
                                conversation_history: List[Dict[str, str]] = None) -> str:
        """
        基于推荐结果与用户聊天
        
        Args:
            user_message: 用户消息
            recommendations: RAG推荐结果
            conversation_history: 对话历史
            
        Returns:
            AI回复
        """
        if not self.is_api_ready():
            return "❌ AI service not configured, please configure AI service first."
        
        try:
            # 构建系统提示
            system_prompt = self._build_system_prompt()
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加对话历史
            if conversation_history:
                messages.extend(conversation_history[-10:])  # 保留最近10条对话
            
            # 添加推荐信息到用户消息中
            enhanced_message = self._enhance_user_message(user_message, recommendations)
            messages.append({"role": "user", "content": enhanced_message})
            
            # 调用AI API
            model_name = getattr(self, 'model', 'gpt-3.5-turbo')
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return f"❌ Error occurred while generating response: {str(e)}"
    
    def generate_recommendation_summary(self, recommendations: Dict[str, Any]) -> str:
        """
        生成推荐摘要
        
        Args:
            recommendations: 推荐结果
            
        Returns:
            推荐摘要文本
        """
        if not self.is_api_ready():
            return "API not configured, unable to generate recommendation summary."
        
        try:
            # 构建推荐数据字符串
            rec_text = self._format_recommendations_for_prompt(recommendations)
            
            prompt = f"""
            Based on the following MXLLM material recommendation data, generate a concise English summary:
            
            {rec_text}
            
            Please provide:
            1. Recommended chemical formulas and their characteristics
            2. Recommended synthesis processes and their advantages
            3. Recommended testing methods and their applications
            4. Overall suggestions and precautions
            
            Requirements: Use concise professional language, emphasize practicality. Do not use markdown format, use plain text format.
            """
            
            model_name = getattr(self, 'model', 'gpt-3.5-turbo')
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"生成推荐摘要失败: {e}")
            return f"Error occurred while generating recommendation summary: {str(e)}"
    
    def analyze_user_query(self, query: str) -> Dict[str, Any]:
        """
        分析用户查询，提取关键信息
        
        Args:
            query: 用户查询
            
        Returns:
            分析结果
        """
        if not self.is_api_ready():
            return {"intent": "unknown", "keywords": [], "requirements": []}
        
        try:
            prompt = f"""
            分析以下用户查询，提取关键信息：
            
            查询: "{query}"
            
            请以JSON格式返回：
            {{
                "intent": "查询意图（如：寻找化学式、合成方法、测试流程、综合咨询等）",
                "keywords": ["关键词1", "关键词2", ...],
                "requirements": ["需求1", "需求2", ...],
                "application": "应用场景（如：吸波材料、电磁屏蔽等）"
            }}
            """
            
            model_name = getattr(self, 'model', 'gpt-3.5-turbo')
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # 尝试解析JSON响应
            import json
            try:
                result = json.loads(response.choices[0].message.content.strip())
                return result
            except json.JSONDecodeError:
                return {
                    "intent": "general_inquiry",
                    "keywords": [query],
                    "requirements": [],
                    "application": "未指定"
                }
                
        except Exception as e:
            logger.error(f"分析用户查询失败: {e}")
            return {
                "intent": "unknown",
                "keywords": [query],
                "requirements": [],
                "application": "未知"
            }
    
    def translate_to_professional_english(self, query_text: str) -> str:
        """
        将查询转换或润色为专业英文
        
        Args:
            query_text: 查询文本（可以是中文或英文）
            
        Returns:
            专业英文查询
        """
        if not self.is_api_ready():
            return query_text
        
        try:
            prompt = f"""
Please translate and polish the following query into a professional English material science term:

Query: "{query_text}"

Requirements:
1. If it's Chinese, please translate it to English
2. If it's already English, please polish it into a more professional term
3. Use standard material science professional terms
4. Professional term equivalents:
   - "Heterostructure" → "heterostructure"
   - "Composite" → "composite"
   - "Electromagnetic wave absorption materials" → "electromagnetic wave absorption materials"
   - "Nanomaterials" → "nanomaterials"
5. Ensure academic accuracy of terms
6. Only return the final professional English query, do not add any other content

Professional English query:"""

            # 打印翻译并润色prompt信息
            logger.info("=" * 60)
            logger.info("🌐✨ Translation and Polishing PROMPT Details:")
            logger.info("=" * 60)
            logger.info(f"🔍 Original Query: {query_text}")
            logger.info("📋 Full Prompt:")
            logger.info("-" * 30)
            logger.info(prompt)
            logger.info("-" * 30)

            model_name = getattr(self, 'model', 'gpt-3.5-turbo')
            logger.info(f"🤖 Using model: {model_name}")
            logger.info(f"🔧 Parameters: max_tokens=100, temperature=0.1")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # 获取和处理响应
            raw_response = response.choices[0].message.content.strip()
            logger.info("🤖 Translation and Polishing Response:")
            logger.info(f"📄 Raw Response: '{raw_response}'")
            
            english_query = raw_response
            # 清理可能的格式问题
            english_query = english_query.replace('"', '').replace('Professional English query:', '').replace('English query:', '').strip()
            
            if english_query != raw_response:
                logger.info(f"🧹 Cleaned response: '{english_query}'")
            
            logger.info(f"✅ Translation and Polishing completed: {query_text} → {english_query}")
            logger.info("=" * 60)
            
            return english_query
            
        except Exception as e:
            logger.error("❌ Translation and Polishing failed:")
            logger.error(f"Error details: {e}")
            logger.error("=" * 60)
            return query_text

    def analyze_and_extract_recommendations(self, query: str, context_text: str) -> Dict[str, Any]:
        """
        分析搜索结果并提取结构化推荐
        
        Args:
            query: 用户查询
            context_text: 搜索到的文档上下文
            
        Returns:
            包含化学式、合成工艺、测试流程的结构化字典
        """
        if not self.is_api_ready():
            return {
                "chemical_formulas": [],
                "synthesis_methods": [],
                "testing_procedures": [],
                "error": "LLM service not configured"
            }
        
        try:
            prompt = f"""
You are a rigorous materials science research assistant. Based on the "Search Results" below and user query "{query}", select and recommend **only 1 best material**, and output **verified, verifiable** multi-step synthesis and testing protocols.

【Strict Rules | Must Follow】
1) No speculation or padding: Only write **content you are certain is real**; do not output placeholders (like "N/A", "unknown", empty strings, 0).
2) No units means no entry: If the original text doesn't provide **values or units** for an item, **don't write that key**; don't add or estimate on your own.
3) Multi-step: Both synthesis and testing should be written step-by-step in **steps arrays** (each object = one step). If literature has multiple steps, increase accordingly; if you think there are **important steps not mentioned in original text** (like necessary washing/drying/mounting/calibration), **allow supplementing as "inferred steps"**:
   - "Inferred steps" must not contain specific values/units not given;
   - Must add `"inferred": true` and `"justification"` (brief rationale, like "standard process essential step/industry practice").
4) Precursors: If precursors exist, must include an independent step with `role: "precursor"`.
5) Recommend only 1 material: Only 1 material entry; briefly explain selection rationale in `rationale` (based on original text evidence).
6) For heterostructure queries, only select heterostructure materials.
7) Output **only a valid JSON object**, no additional text; all strings use **double quotes**.

【Search Results】
{context_text}

【Output JSON Structure】(Keys may be omitted; except `material.chemical_formula` is required, other fields omit if no reliable information)
{{
  "source": "Source literature title",
  "doi": "DOI number",
  "query": "{query}",
  "content": [
    {{
      "record_designation": "Sample/formula identifier (if given in original text)",
      "material": {{
        "chemical_formula": "Specific chemical formula (required, e.g. Ti3C2Tx or explicit composite/heterostructure)",
        "name": "Material name/common name (if any)",
        "composition_type": "Single/doped/composite/heterostructure (if any)",
        "structure_type": "Crystal/layered/porous/core-shell/interface type (if any)",
        "morphology": "Morphology (if any)",
        "heterostructure_type": "Heterostructure type (if any)"
      }},
      "performance": {{
        "rl_min": "e.g. \"-50.06 dB\" (if available, retain original units with necessary conditions inline)",
        "matching_thickness": "e.g. \"1.5 mm\" (if available)",
        "eab": "e.g. \"4.0 GHz\" (if available)",
        "other": "Other key performance points mentioned in original text (if any, brief)"
      }},
      "synthesis_steps": [
        {{
          "step": 1,
          "role": "precursor",
          "step_name": "Step name (prioritize original text terminology)",
          "method": "Method/process name (if any)",
          "inputs": ["Raw materials/solvents/additives/precursors (if any)"],
          "conditions": ["Write key conditions as **string entries** one by one; omit this key if none"],
          "outputs": "Intermediates/morphology/structural features obtained in this step (if any)",
          "notes": "Safety/filtering/washing/drying etc. (if any)"
        }}
        /* Can continue adding steps 2, 3...; for "inferred steps", need to add:
           "inferred": true,
           "justification": "Why this step is standard and necessary (no values/units)"
        */
      ],
      "testing_steps": [
        {{
          "step": 1,
          "step_name": "Sample pretreatment/mounting/measurement etc. (prioritize original text terminology)",
          "method": "Testing method/instrument paradigm (if any)",
          "parameters": ["Write key parameters as **string entries**; omit this key if none"],
          "notes": "Such as baseline/calibration/repetition times and other original text points (if any)"
        }}
        /* Same as above, list step by step; can include "inferred steps" (need to mark inferred and justification, no values/units) */
      ],
      "confidence": "high/medium/low"
    }}
  ]
}}
"""

            # 打印完整的prompt信息
            logger.info("=" * 80)
            logger.info("�� AI PROMPT Details:")
            logger.info("=" * 80)
            logger.info(f"🔍 Query: {query}")
            logger.info(f"📏 Context Length: {len(context_text)} characters")
            logger.info("📋 Full Prompt:")
            logger.info("-" * 40)
            logger.info(prompt)
            logger.info("-" * 40)

            model_name = getattr(self, 'model', 'gpt-3.5-turbo')
            logger.info(f"🤖 Using model: {model_name}")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # 获取原始响应
            raw_response = response.choices[0].message.content.strip()
            logger.info("=" * 80)
            logger.info("🤖 AI Response Details:")
            logger.info("=" * 80)
            logger.info(f"📄 Raw Response Length: {len(raw_response)} characters")
            logger.info("📋 Full Response:")
            logger.info("-" * 40)
            logger.info(raw_response)
            logger.info("-" * 40)
            
            # 清理可能的格式问题
            clean_response = self._clean_json_response(raw_response)
            
            if clean_response != raw_response:
                logger.info("🧹 Cleaned response:")
                logger.info("-" * 40)
                logger.info(clean_response)
                logger.info("-" * 40)
            
            # 解析JSON响应
            import json
            try:
                result = json.loads(clean_response)
                logger.info("✅ JSON parsed successfully")
                logger.info(f"📊 Returned results: {len(result.get('chemical_formulas', []))} chemical formulas, "
                           f"{len(result.get('synthesis_methods', []))} synthesis methods, "
                           f"{len(result.get('testing_procedures', []))} testing procedures")
                logger.info("=" * 80)
                return result
            except json.JSONDecodeError as e:
                logger.error("❌ JSON parsing failed details:")
                logger.error(f"Error: {e}")
                logger.error(f"Error position: Line {e.lineno}, Column {e.colno}")
                logger.error(f"Cleaned response first 500 characters: {clean_response[:500]}...")
                
                # 尝试手动修复常见JSON问题
                fixed_response = self._try_fix_json(clean_response)
                if fixed_response:
                    try:
                        result = json.loads(fixed_response)
                        logger.info("✅ JSON fixed successfully")
                        logger.info("=" * 80)
                        return result
                    except:
                        logger.error("❌ JSON fixing also failed")
                
                logger.info("=" * 80)
                return {
                    "chemical_formulas": [],
                    "synthesis_methods": [],
                    "testing_procedures": [],
                    "error": "LLM returned incorrect format",
                    "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
                }
                
        except Exception as e:
            logger.error("❌ LLM call failed:")
            logger.error(f"Error details: {e}")
            logger.error("=" * 80)
            return {
                "chemical_formulas": [],
                "synthesis_methods": [],
                "testing_procedures": [],
                "error": f"LLM analysis error: {str(e)}"
            }
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示
        
        Returns:
            系统提示文本
        """
        return """
        You are a professional MXLLM material research assistant, specializing in helping users with MXLLM electromagnetic wave absorption material research and development.

        Your responsibilities:
        1. Recommend appropriate chemical formulas, synthesis processes, and testing procedures based on provided literature data
        2. Answer professional questions about MXLLM materials
        3. Provide practical research suggestions and technical guidance
        4. Explain the relationship between material properties and application scenarios

        Response requirements:
        - Answer in English
        - Use professional but understandable language
        - Provide specific actionable suggestions
        - **When citing recommendation data, must include corresponding DOI numbers to ensure literature traceability**
        - Format DOI information when recommending chemical formulas, synthesis processes, or testing procedures
        - Emphasize practicality and feasibility
        - Do not use markdown formatting like **

        CRITICAL: When users inquire, you MUST prioritize answering based on the specific recommendation data provided in the user message. 
        If recommendation data is provided, base your entire response on that data and cite the specific sources, DOI numbers, and experimental details mentioned.
        Only use general knowledge if explicitly noted that no database recommendations are available.
        If recommendation data includes DOI numbers, please clearly display them in your answer, format as:
        - Chemical Formula: Ti3C2Tx (DOI: 10.xxxx/xxxx)
        - Synthesis Method: HF Etching (DOI: 10.xxxx/xxxx)
        
        Always check if the user message contains "Relevant recommendations retrieved from database" - if it does, focus your response on analyzing and explaining those specific recommendations rather than providing general information.
        """
    
    def _enhance_user_message(self, user_message: str, recommendations: Dict[str, Any] = None) -> str:
        """
        增强用户消息，添加推荐信息
        
        Args:
            user_message: 用户原始消息
            recommendations: 推荐结果
            
        Returns:
            增强后的消息
        """
        if not recommendations:
            # 没有推荐数据时，直接返回用户消息
            return f"User query: {user_message}\n\nNote: Currently in quick mode, unable to provide database-based recommendations, please answer based on general knowledge."
        
        # 检查新格式的数据结构 (包含content键)
        if recommendations.get('content') and isinstance(recommendations['content'], list) and len(recommendations['content']) > 0:
            enhanced = f"User query: {user_message}\n\n"
            enhanced += "Relevant recommendations retrieved from database:\n"
            enhanced += self._format_recommendations_for_prompt(recommendations)
            return enhanced
        
        # 检查旧格式的数据结构
        if any(recommendations.get(key, []) for key in ['chemical_formulas', 'synthesis_methods', 'testing_procedures']):
            enhanced = f"User query: {user_message}\n\n"
            enhanced += "Relevant recommendations retrieved from database:\n"
            enhanced += self._format_recommendations_for_prompt(recommendations)
        return enhanced
        
        # 如果推荐数据为空，使用快速模式提示
        return f"User query: {user_message}\n\nNote: Currently in quick mode, unable to provide database-based recommendations, please answer based on general knowledge."
    
    def _format_recommendations_for_prompt(self, recommendations: Dict[str, Any]) -> str:
        """
        格式化推荐结果用于提示
        
        Args:
            recommendations: 推荐结果
            
        Returns:
            格式化后的文本
        """
        formatted = ""
        
        def format_value(value):
            """通用值格式化函数，处理各种数据类型"""
            if isinstance(value, dict):
                # 字典类型：尝试提取关键信息
                if 'name' in value:
                    # 如果有name字段，优先使用name
                    result = value['name']
                    if 'amount' in value:
                        result += f" ({value['amount']})"
                    elif 'volume' in value:
                        result += f" ({value['volume']})"
                    return result
                else:
                    # 否则转换为key: value格式
                    pairs = [f"{k}: {v}" for k, v in value.items()]
                    return "; ".join(pairs)
            elif isinstance(value, list):
                # 列表类型：递归处理每个元素
                return ", ".join([format_value(item) for item in value])
            else:
                # 其他类型直接转字符串
                return str(value)
        
        # 检查新格式的数据结构 (包含content键)
        if recommendations.get('content') and isinstance(recommendations['content'], list):
            content = recommendations['content']
            source = recommendations.get('source', 'Unknown source')
            doi = recommendations.get('doi', '')
            
            formatted += f"Data source: {source}\n"
            if doi:
                formatted += f"DOI: {doi}\n"
            formatted += "\n"
            
            for i, item in enumerate(content, 1):
                material = item.get('material', {})
                performance = item.get('performance', {})
                synthesis_steps = item.get('synthesis_steps', [])
                testing_steps = item.get('testing_steps', [])
                confidence = item.get('confidence', 'medium')
                
                formula = material.get('chemical_formula', 'Unknown')
                formatted += f"Material {i}: {formula} (Confidence: {confidence})\n"
                
                # 材料信息
                if material.get('composition_type'):
                    formatted += f"  Composition type: {material['composition_type']}\n"
                if material.get('structure_type'):
                    formatted += f"  Structure type: {material['structure_type']}\n"
                if material.get('morphology'):
                    formatted += f"  Morphology: {material['morphology']}\n"
                
                # 性能数据
                if performance:
                    formatted += "  Performance data:\n"
                    for key, value in performance.items():
                        if value and value != "":
                            display_name = {
                                'rl_min': 'RL Minimum',
                                'matching_thickness': 'Matching Thickness',
                                'eab': 'Effective Absorption Bandwidth',
                                'other': 'Other Performance'
                            }.get(key, key)
                            formatted += f"    {display_name}: {format_value(value)}\n"
                
                # 合成步骤
                if synthesis_steps:
                    formatted += "  Synthesis steps:\n"
                    for step in synthesis_steps:
                        step_name = step.get('step_name', 'Unknown step')
                        method = step.get('method', '')
                        formatted += f"    {step.get('step', 0)}. {step_name}"
                        if method:
                            formatted += f" ({method})"
                        formatted += "\n"
                        
                        if step.get('inputs'):
                            inputs = step['inputs']
                            formatted += f"       Inputs: {format_value(inputs)}\n"
                        
                        if step.get('conditions'):
                            conditions = step['conditions']
                            formatted += f"       Conditions: {format_value(conditions)}\n"
                        
                        if step.get('outputs'):
                            outputs = step['outputs']
                            formatted += f"       Outputs: {format_value(outputs)}\n"
                        
                        if step.get('notes'):
                            notes = step['notes']
                            formatted += f"       Notes: {format_value(notes)}\n"
                
                # 测试步骤
                if testing_steps:
                    formatted += "  Testing steps:\n"
                    for step in testing_steps:
                        step_name = step.get('step_name', 'Unknown step')
                        method = step.get('method', '')
                        formatted += f"    {step.get('step', 0)}. {step_name}"
                        if method:
                            formatted += f" ({method})"
                        formatted += "\n"
                        
                        if step.get('parameters'):
                            parameters = step['parameters']
                            formatted += f"       Parameters: {format_value(parameters)}\n"
                        
                        if step.get('notes'):
                            notes = step['notes']
                            formatted += f"       Notes: {format_value(notes)}\n"
                
                formatted += "\n"
            
            return formatted
        
        # 处理旧格式的数据结构（保持兼容性）
        # 化学式推荐
        if recommendations.get('chemical_formulas'):
            formatted += "Recommended chemical formulas:\n"
            for i, formula in enumerate(recommendations['chemical_formulas'], 1):
                formatted += f"{i}. {formula.get('formula', 'Unknown')}\n"
                formatted += f"   Source: {formula.get('source', 'Unknown')}\n"
                if formula.get('doi'):
                    formatted += f"   DOI: {formula['doi']}\n"
                formatted += f"   Relevance: {formula.get('score', 0):.2f}\n"
                
                # 添加验证和性能信息
                if formula.get('validation'):
                    validation = formula['validation']
                    formatted += f"   Performance validation: {formula.get('performance_summary', 'Unknown')}\n"
                    
                    if validation['found_in_db']:
                        # 实验数据
                        exp_data = validation['experimental_data']
                        if exp_data and len(exp_data) > 0:
                            first_data = exp_data[0]
                            formatted += f"   Experimental performance: "
                            if first_data.get('properties'):
                                props = first_data['properties']
                                prop_strs = [f"{k}={v}" for k, v in props.items()]
                                formatted += ", ".join(prop_strs)
                            formatted += "\n"
                            if first_data.get('doi'):
                                formatted += f"   Performance data DOI: {first_data['doi']}\n"
                    
                    elif validation.get('prediction'):
                        # 预测数据
                        pred = validation['prediction']
                        formatted += f"   AI predicted performance: EAB={pred['eab_prediction']}({pred['eab_meaning']}), RL={pred['rl_prediction']}({pred['rl_meaning']})\n"
                        formatted += f"   Prediction confidence: {pred['confidence']:.2f}\n"
                
                if formula.get('description'):
                    formatted += f"   Description: {formula['description']}\n"
                formatted += "\n"
        
        # 合成工艺推荐
        if recommendations.get('synthesis_methods'):
            formatted += "Recommended synthesis processes:\n"
            for i, method in enumerate(recommendations['synthesis_methods'], 1):
                formatted += f"{i}. {method.get('method', 'Unknown')}\n"
                formatted += f"   Source: {method.get('source', 'Unknown')}\n"
                if method.get('doi'):
                    formatted += f"   DOI: {method['doi']}\n"
                formatted += f"   Relevance: {method.get('score', 0):.2f}\n"
                if method.get('description'):
                    formatted += f"   Description: {method['description']}\n"
                formatted += "\n"
        
        # 测试流程推荐
        if recommendations.get('testing_procedures'):
            formatted += "Recommended testing procedures:\n"
            for i, procedure in enumerate(recommendations['testing_procedures'], 1):
                formatted += f"{i}. {procedure.get('procedure', 'Unknown')}\n"
                formatted += f"   Source: {procedure.get('source', 'Unknown')}\n"
                if procedure.get('doi'):
                    formatted += f"   DOI: {procedure['doi']}\n"
                formatted += f"   Relevance: {procedure.get('score', 0):.2f}\n"
                if procedure.get('description'):
                    formatted += f"   Description: {procedure['description']}\n"
                formatted += "\n"
        
        return formatted
    
    def format_chat_response(self, 
                           ai_response: str, 
                           recommendations: Dict[str, Any] = None) -> str:
        """
        格式化聊天回复，包含推荐信息
        
        Args:
            ai_response: AI回复
            recommendations: 推荐结果
            
        Returns:
            格式化后的回复
        """
        formatted_response = ai_response
        
        if recommendations:
            formatted_response += "\n\n" + "="*50 + "\n"
            formatted_response += "📋 **Database-based recommendation results**\n\n"
            
            # 检查新格式的数据结构
            if recommendations.get('content') and isinstance(recommendations['content'], list):
                # 新的JSON格式
                content = recommendations['content']
                source = recommendations.get('source', 'Unknown source')
                doi = recommendations.get('doi', '')
                
                formatted_response += f"�� **Data source:** {source}\n"
                if doi:
                    formatted_response += f"�� **DOI:** {doi}\n"
                formatted_response += "\n"
                
                for i, item in enumerate(content, 1):
                    material = item.get('material', {})
                    performance = item.get('performance', {})
                    confidence = item.get('confidence', 'medium')
                    
                    formula = material.get('chemical_formula', 'Unknown')
                    confidence_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}.get(confidence, confidence)
                    
                    formatted_response += f"🧪 **Material {i}: {formula}** (Confidence: {confidence_text})\n"
                    
                    # 材料特性
                    if material.get('composition_type'):
                        formatted_response += f"   • Composition type: {material['composition_type']}\n"
                    if material.get('structure_type'):
                        formatted_response += f"   • Structure type: {material['structure_type']}\n"
                    if material.get('morphology'):
                        formatted_response += f"   • Morphology: {material['morphology']}\n"
                    
                    # 性能数据
                    if performance:
                        formatted_response += "   �� **Performance data:**\n"
                        for key, value in performance.items():
                            if value and value != "":
                                display_name = {
                                    'rl_min': 'RL Minimum',
                                    'matching_thickness': 'Matching Thickness',
                                    'eab': 'Effective Absorption Bandwidth',
                                    'other': 'Other Performance'
                                }.get(key, key)
                                formatted_response += f"     - {display_name}: {value}\n"
                    
                    # 合成步骤
                    synthesis_steps = item.get('synthesis_steps', [])
                    if synthesis_steps:
                        formatted_response += "   ⚗️ **Synthesis steps:**\n"
                        for step in synthesis_steps:
                            step_name = step.get('step_name', 'Unknown step')
                            method = step.get('method', '')
                            formatted_response += f"     {step.get('step', 0)}. {step_name}"
                            if method:
                                formatted_response += f" ({method})"
                            formatted_response += "\n"
                    
                    # 测试步骤
                    testing_steps = item.get('testing_steps', [])
                    if testing_steps:
                        formatted_response += "   �� **Testing steps:**\n"
                        for step in testing_steps:
                            step_name = step.get('step_name', 'Unknown step')
                            method = step.get('method', '')
                            formatted_response += f"     {step.get('step', 0)}. {step_name}"
                            if method:
                                formatted_response += f" ({method})"
                            formatted_response += "\n"
                    
                    formatted_response += "\n"
            else:
                # 兼容旧格式
                # 添加化学式推荐
                if recommendations.get('chemical_formulas'):
                    formatted_response += "🧪 **Recommended chemical formulas:**\n"
                    for i, formula in enumerate(recommendations['chemical_formulas'], 1):
                        formatted_response += f"{i}. **{formula.get('formula', 'Unknown')}**\n"
                        formatted_response += f"   📖 Source: {formula.get('source', 'Unknown')}\n"
                        formatted_response += f"   🎯 Relevance: {formula.get('score', 0):.1%}\n\n"
                
                # 添加合成工艺推荐
                if recommendations.get('synthesis_methods'):
                    formatted_response += "⚗️ **Recommended synthesis processes:**\n"
                    for i, method in enumerate(recommendations['synthesis_methods'], 1):
                        formatted_response += f"{i}. **{method.get('method', 'Unknown')}**\n"
                        formatted_response += f"   📖 Source: {method.get('source', 'Unknown')}\n"
                        formatted_response += f"   �� Relevance: {method.get('score', 0):.1%}\n\n"
                
                # 添加测试流程推荐
                if recommendations.get('testing_procedures'):
                    formatted_response += "🔬 **Recommended testing procedures:**\n"
                    for i, procedure in enumerate(recommendations['testing_procedures'], 1):
                        formatted_response += f"{i}. **{procedure.get('procedure', 'Unknown')}**\n"
                        formatted_response += f"   �� Source: {procedure.get('source', 'Unknown')}\n"
                        formatted_response += f"   🎯 Relevance: {procedure.get('score', 0):.1%}\n\n"
        
        return formatted_response 

    def _clean_json_response(self, response: str) -> str:
        """清理JSON响应中的常见问题"""
        # 移除可能的前后缀
        response = response.strip()
        
        # 找到JSON的开始和结束
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx+1]
        
        # 移除可能的markdown代码块标记
        response = response.replace('```json', '').replace('```', '')
        
        return response.strip()
    
    def _try_fix_json(self, json_str: str) -> str:
        """尝试修复常见的JSON格式问题"""
        try:
            # 修复常见的引号问题
            import re
            
            # 确保属性名都有双引号
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # 修复尾随逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            return json_str
        except:
            return None 