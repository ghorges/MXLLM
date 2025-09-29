"""
RAG检索增强生成系统
实现基于向量数据库的文档检索和推荐
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle

from .data_loader import DataLoader
from .material_validator import MaterialValidator

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG检索增强生成系统 - 使用FAISS替代ChromaDB"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = None):
        """
        初始化RAG系统
        
        Args:
            embedding_model: 嵌入模型名称
            persist_directory: 数据库持久化目录
        """
        if persist_directory is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(current_dir, "..", "data", "vectordb")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 延迟初始化嵌入模型和FAISS，避免启动时阻塞
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []  # 存储文档内容
        self.metadatas = []  # 存储元数据
        self.doc_ids = []    # 存储文档ID
        self.collection_name = "mxene_materials"
        
        # FAISS相关文件路径
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.documents_file = os.path.join(persist_directory, "documents.pkl")
        self.metadatas_file = os.path.join(persist_directory, "metadatas.pkl")
        self.doc_ids_file = os.path.join(persist_directory, "doc_ids.pkl")
        
        self.data_loader = DataLoader()
        
        # 延迟初始化MaterialValidator，避免启动时异常
        self.material_validator = None
        self._init_material_validator()
        
        self.is_initialized = False
    
    def _init_material_validator(self):
        """安全初始化MaterialValidator"""
        try:
            from .material_validator import MaterialValidator
            self.material_validator = MaterialValidator(self.data_loader)
            logger.info("✅ MaterialValidator初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ MaterialValidator初始化失败: {e}")
            logger.warning("   预测功能将不可用，但推荐功能正常")
            self.material_validator = None
    
    def _init_embedding_and_faiss(self):
        """初始化嵌入模型和FAISS"""
        if self.embedding_model is None:
            logger.info("正在初始化嵌入模型...")
            try:
                # 设置离线模式，避免联网下载
                import os
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                # 尝试加载本地缓存的模型
                logger.info(f"尝试加载本地模型: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, 
                                                         local_files_only=True)
                logger.info(f"✅ 成功加载本地嵌入模型: {self.embedding_model_name}")
                
            except Exception as local_error:
                logger.warning(f"⚠️ 本地模型加载失败: {local_error}")
                logger.info("尝试在线下载模型...")
                
                try:
                    # 如果本地没有，尝试下载（仅首次）
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    logger.info(f"✅ 成功下载并加载嵌入模型: {self.embedding_model_name}")
                    logger.info("💡 下次启动将使用本地缓存，无需联网")
                    
                except Exception as download_error:
                    logger.error(f"❌ 模型下载也失败: {download_error}")
                    logger.error("🔧 解决方案:")
                    logger.error("   1. 检查网络连接")
                    logger.error("   2. 或手动下载模型到 ~/.cache/huggingface/transformers/")
                    logger.error("   3. 或使用其他可用的本地模型")
                    raise Exception("嵌入模型初始化失败，请检查网络或模型缓存")
        
        if self.faiss_index is None:
            logger.info("正在初始化FAISS索引...")
            try:
                # 检查是否存在已保存的索引
                if os.path.exists(self.index_file):
                    logger.info("加载已存在的FAISS索引...")
                    self.faiss_index = faiss.read_index(self.index_file)
                    
                    # 加载对应的文档数据
                    with open(self.documents_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    with open(self.metadatas_file, 'rb') as f:
                        self.metadatas = pickle.load(f)
                    with open(self.doc_ids_file, 'rb') as f:
                        self.doc_ids = pickle.load(f)
                    
                    logger.info(f"✅ 成功加载FAISS索引，包含 {len(self.documents)} 个文档")
                else:
                    # 创建新的空索引
                    dimension = 384  # all-MiniLM-L6-v2的向量维度
                    self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积作为相似度
                    logger.info("✅ 创建新的FAISS索引")
                
            except Exception as e:
                logger.error(f"❌ FAISS初始化失败: {e}")
                raise

    def _has_any_docs(self) -> bool:
        """检查是否已有文档"""
        return len(self.documents) > 0

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """统一的文本编码 + 清洗"""
        vecs = self.embedding_model.encode(texts, convert_to_numpy=True)
        vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        # 归一化向量，用于内积计算相似度
        faiss.normalize_L2(vecs)
        return vecs

    def initialize_database(self, force_rebuild: bool = False, quick_mode: bool = False) -> bool:
        """
        初始化向量数据库
        
        Args:
            force_rebuild: 是否强制重建数据库
            quick_mode: 快速模式，跳过耗时的向量化过程
            
        Returns:
            是否初始化成功
        """
        try:
            logger.info(f"开始初始化数据库... (快速模式: {quick_mode})")
            
            # 快速模式：跳过向量数据库，直接返回成功
            if quick_mode:
                logger.info("快速模式：跳过向量数据库初始化")
                self.is_initialized = True
                return True
            
            # 初始化嵌入模型和FAISS（可能耗时）
            logger.info("🔄 正在初始化核心组件...")
            self._init_embedding_and_faiss()
            
            # 检查是否需要加载数据
            if not force_rebuild:
                if self._has_any_docs():
                    logger.info("数据库中已有数据，跳过数据加载")
                    self.is_initialized = True
                    return True
                else:
                    logger.info("数据库为空，需要加载数据")
            
            # 加载数据
            logger.info("开始加载MXene数据...")
            mxene_data = self.data_loader.get_mxene_data()
            if not mxene_data:
                logger.warning("未找到MXene数据，但标记为已初始化")
                self.is_initialized = True
                return True
            logger.info(f"加载了 {len(mxene_data)} 条MXene数据")
            
            # 如需重建，先清空现有数据
            if force_rebuild and self._has_any_docs():
                logger.info("强制重建，清空现有数据...")
                self._clear_index()
            
            # 为每个类别创建专门的文档
            all_documents: List[str] = []
            all_metadatas: List[Dict[str, Any]] = []
            all_ids: List[str] = []
            
            categories = ['chemical_formula', 'synthesis_method', 'testing_procedure']
            for category in categories:
                docs, metas, ids = self._create_category_documents(mxene_data, category)
                all_documents.extend(docs)
                all_metadatas.extend(metas)
                all_ids.extend(ids)
                logger.info(f"为类别 {category} 创建了 {len(docs)} 个文档")
            
            # 批量添加文档到FAISS索引
            if all_documents:
                batch_size = 50  # 减小批次大小，避免超时
                total_batches = (len(all_documents) - 1) // batch_size + 1
                logger.info(f"开始分批添加 {len(all_documents)} 个文档，共 {total_batches} 批")
                
                try:
                    for i in range(0, len(all_documents), batch_size):
                        batch_docs = all_documents[i:i+batch_size]
                        batch_metas = all_metadatas[i:i+batch_size]
                        batch_ids = all_ids[i:i+batch_size]
                        
                        batch_num = i // batch_size + 1
                        logger.info(f"正在添加批次 {batch_num}/{total_batches} ({len(batch_docs)} 个文档)")
                        
                        try:
                            # 过滤空文本
                            clean = [(d, m, _id) for (d, m, _id) in zip(batch_docs, batch_metas, batch_ids) if d and d.strip()]
                            if not clean:
                                logger.warning(f"批次 {batch_num} 全是空文档，跳过")
                                continue
                            docs2, metas2, ids2 = zip(*clean)

                            vecs = self._encode_texts(list(docs2))
                            self._add_vectors_to_index(vecs, list(docs2), list(metas2), list(ids2))
                            logger.info(f"✅ 批次 {batch_num} 添加成功")
                        except Exception as batch_error:
                            logger.error(f"❌ 批次 {batch_num} 添加失败: {batch_error}")
                            # 尝试逐条添加
                            if len(batch_docs) > 1:
                                logger.info("尝试单个文档添加...")
                                for j, (doc, meta, doc_id) in enumerate(zip(batch_docs, batch_metas, batch_ids)):
                                    try:
                                        if not doc or not doc.strip():
                                            continue
                                        _v = self._encode_texts([doc])
                                        self._add_vectors_to_index(_v, [doc], [meta], [doc_id])
                                    except Exception as single_error:
                                        logger.warning(f"跳过文档 {doc_id}: {single_error}")
                            else:
                                logger.warning(f"跳过批次 {batch_num}")
                                continue
                    logger.info("✅ 文档添加流程完成")
                    
                    # 保存索引和数据
                    self._save_index()
                    
                except Exception as add_error:
                    logger.error(f"❌ 添加文档过程中发生严重错误: {add_error}")
                    logger.warning("⚠️ 将标记为已初始化以保持系统可用")
            else:
                logger.warning("没有文档可添加到数据库")
            
            self.is_initialized = True
            logger.info("数据库初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            return False
    
    def _clear_index(self):
        """清空FAISS索引和相关数据"""
        dimension = 384  # all-MiniLM-L6-v2的向量维度
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.metadatas = []
        self.doc_ids = []
    
    def _add_vectors_to_index(self, vectors: np.ndarray, documents: List[str], 
                             metadatas: List[Dict[str, Any]], ids: List[str]):
        """添加向量到FAISS索引"""
        # 添加向量到FAISS索引
        self.faiss_index.add(vectors)
        
        # 同步更新文档数据
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.doc_ids.extend(ids)
    
    def _save_index(self):
        """保存FAISS索引和相关数据"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.faiss_index, self.index_file)
            
            # 保存文档数据
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.metadatas_file, 'wb') as f:
                pickle.dump(self.metadatas, f)
            with open(self.doc_ids_file, 'wb') as f:
                pickle.dump(self.doc_ids, f)
                
            logger.info("✅ FAISS索引和数据已保存")
        except Exception as e:
            logger.error(f"❌ 保存FAISS索引失败: {e}")
    
    def _build_document_text(self, item: Dict[str, Any]) -> str:
        """
        构建文档文本
        
        Args:
            item: 数据项
            
        Returns:
            组合后的文档文本
        """
        parts = []
        
        if item.get('title'):
            parts.append(f"标题: {item['title']}")
        
        if item.get('abstract'):
            parts.append(f"摘要: {item['abstract']}")
        
        if item.get('chemical_formula'):
            parts.append(f"化学式: {item['chemical_formula']}")
        
        if item.get('synthesis_method'):
            parts.append(f"合成方法: {item['synthesis_method']}")
        
        if item.get('testing_procedure'):
            parts.append(f"测试流程: {item['testing_procedure']}")
        
        # 添加内容文本（截取前1000字符避免过长）
        content = item.get('content', [])
        if isinstance(content, list):
            content_text = ' '.join(content)[:1000]
        else:
            content_text = str(content)[:1000]
        
        if content_text.strip():
            parts.append(f"内容: {content_text}")
        
        return '\n'.join(parts)
    
    def _create_category_documents(self, data: List[Dict[str, Any]], 
                                 category: str) -> Tuple[List[str], List[Dict], List[str]]:
        """
        为特定类别创建专门的文档
        
        Args:
            data: 数据列表
            category: 类别名称
            
        Returns:
            文档列表、元数据列表、ID列表
        """
        documents = []
        metadatas = []
        ids = []
        
        category_map = {
            'chemical_formula': '化学式',
            'synthesis_method': '合成方法',
            'testing_procedure': '测试流程'
        }
        
        # 关键词用于从内容中识别相关信息
        category_keywords = {
            'chemical_formula': ['Ti3C2', 'MXene', 'formula', 'composition', 'chemical', 'Ti2C', 'Nb2C', 'V2C', 'Ta2C', 'Mo2C'],
            'synthesis_method': ['synthesis', 'preparation', 'fabrication', 'etching', 'HF', 'method', 'procedure', 'process'],
            'testing_procedure': ['characterization', 'measurement', 'analysis', 'test', 'XRD', 'SEM', 'TEM', 'VNA', 'electromagnetic']
        }
        
        keywords = category_keywords.get(category, [])
        
        for i, item in enumerate(data):
            # 首先检查原始字段
            value = item.get(category, '')
            if not value or not value.strip():
                # 如果原始字段为空，从title、abstract、content中查找相关内容
                content_text = ""
                if item.get('title'):
                    content_text += item['title'] + " "
                if item.get('abstract'):
                    content_text += item['abstract'] + " "
                if item.get('content'):
                    if isinstance(item['content'], list):
                        content_text += " ".join(item['content'])
                    else:
                        content_text += str(item['content'])
                
                # 检查是否包含相关关键词
                content_lower = content_text.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    # 提取相关片段作为值
                    sentences = content_text.split('.')
                    relevant_sentences = []
                    for sentence in sentences:
                        if any(keyword.lower() in sentence.lower() for keyword in keywords):
                            relevant_sentences.append(sentence.strip())
                    
                    if relevant_sentences:
                        value = '. '.join(relevant_sentences[:3])  # 取前3个相关句子
            
            if value and value.strip():
                # 创建专门针对该类别的文档
                doc_text = f"{category_map[category]}: {value}\n"
                
                # 添加相关上下文
                if item.get('title'):
                    doc_text += f"研究标题: {item['title']}\n"
                if item.get('abstract'):
                    doc_text += f"研究摘要: {item['abstract'][:300]}...\n"
                
                documents.append(doc_text)
                
                metadata = {
                    'doi': item.get('doi', ''),
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'source': item.get('source', ''),
                    'type': category,
                    'value': value
                }
                metadatas.append(metadata)
                ids.append(f"{category}_{i}")
        
        return documents, metadatas, ids
    
    def search_similar(self, query: str, n_results: int = 5, 
                      category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档（使用手动 query_embeddings）
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            category: 指定搜索类别
            
        Returns:
            相似文档列表
        """
        # 确保组件已初始化
        if self.faiss_index is None:
            logger.warning("数据库未初始化，尝试初始化...")
            if not self.initialize_database():
                return []
        
        if not self.is_initialized:
            logger.warning("数据库未初始化，尝试初始化...")
            if not self.initialize_database():
                return []
        
        try:
            # 手动编码查询向量
            qvec = self._encode_texts([query])
            
            # 执行搜索
            D, I = self.faiss_index.search(qvec, min(n_results * 3, len(self.documents)))  # 获取更多结果用于过滤
            
            search_results = []
            for distance, idx in zip(D[0], I[0]):
                if idx == -1:  # FAISS返回-1表示无效索引
                    continue
                    
                doc_idx = int(idx)
                if doc_idx >= len(self.metadatas):
                    continue
                    
                doc_meta = self.metadatas[doc_idx]
                doc_text = self.documents[doc_idx]
                
                # 应用类别过滤
                if category and doc_meta.get('type') != category:
                    continue
                
                result = {
                    'document': doc_text,
                    'metadata': doc_meta,
                    'similarity_score': float(distance),  # FAISS使用内积，值越大越相似
                    'distance': float(distance)
                }
                search_results.append(result)
                
                # 达到所需结果数量即停止
                if len(search_results) >= n_results:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def recommend_chemical_formula(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        推荐化学式
        """
        enhanced_query = f"MXene 化学式 材料组成 {query}"
        results = self.search_similar(
            enhanced_query, 
            n_results=n_results * 2,
            category='chemical_formula'
        )
        
        formulas: Dict[str, Dict[str, Any]] = {}
        for result in results:
            meta = result['metadata']
            formula = meta.get('value', '')
            if formula and formula not in formulas:
                formulas[formula] = {
                    'formula': formula,
                    'source': meta.get('title', '未知来源'),
                    'doi': meta.get('doi', ''),
                    'score': result['similarity_score'],
                    'description': (result.get('document') or '')[:200] + '...'
                }
        
        sorted_formulas = sorted(
            formulas.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:n_results]
        
        return sorted_formulas
    
    def recommend_synthesis_method(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        推荐合成工艺
        """
        enhanced_query = f"MXene 合成方法 制备工艺 {query}"
        results = self.search_similar(
            enhanced_query,
            n_results=n_results * 2,
            category='synthesis_method'
        )
        
        methods: Dict[str, Dict[str, Any]] = {}
        for result in results:
            meta = result['metadata']
            method = meta.get('value', '')
            if method and method not in methods:
                methods[method] = {
                    'method': method,
                    'source': meta.get('title', '未知来源'),
                    'doi': meta.get('doi', ''),
                    'score': result['similarity_score'],
                    'description': (result.get('document') or '')[:200] + '...'
                }
        
        sorted_methods = sorted(
            methods.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:n_results]
        
        return sorted_methods
    
    def recommend_testing_procedure(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        推荐测试流程
        """
        enhanced_query = f"MXene 测试方法 表征手段 性能测试 {query}"
        results = self.search_similar(
            enhanced_query,
            n_results=n_results * 2,
            category='testing_procedure'
        )
        
        procedures: Dict[str, Dict[str, Any]] = {}
        for result in results:
            meta = result['metadata']
            procedure = meta.get('value', '')
            if procedure and procedure not in procedures:
                procedures[procedure] = {
                    'procedure': procedure,
                    'source': meta.get('title', '未知来源'),
                    'doi': meta.get('doi', ''),
                    'score': result['similarity_score'],
                    'description': (result.get('document') or '')[:200] + '...'
                }
        
        sorted_procedures = sorted(
            procedures.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:n_results]
        
        return sorted_procedures
    
    def get_comprehensive_recommendations(self, query: str) -> Dict[str, Any]:
        """
        获取综合推荐（化学式、合成工艺、测试流程）
        """
        recommendations = {
            'chemical_formulas': self.recommend_chemical_formula(query),
            'synthesis_methods': self.recommend_synthesis_method(query),
            'testing_procedures': self.recommend_testing_procedure(query)
        }
        return recommendations
    
    def get_enhanced_recommendations(self, query: str, enable_prediction: bool = True) -> Dict[str, Any]:
        """
        获取增强版推荐（包含材料验证和性能预测）
        """
        logger.info(f"🚀 获取增强版推荐: {query}")
        
        # 1. 获取基础推荐
        base_recommendations = self.get_comprehensive_recommendations(query)
        
        # 2. 验证推荐的化学式
        enhanced_formulas = []
        for formula_item in base_recommendations['chemical_formulas']:
            formula = formula_item['formula']
            
            # 验证材料性能（根据参数决定是否预测）
            if self.material_validator:
                validation_result = self.material_validator.validate_material(formula, enable_prediction=enable_prediction)
            else:
                validation_result = {
                    'formula': formula,
                    'source': 'none',
                    'found_in_db': False,
                    'experimental_data': None,
                    'prediction': None,
                    'confidence': 'low',
                    'summary': "材料验证器不可用"
                }
            
            enhanced_item = {
                **formula_item,
                'validation': validation_result,
                'has_experimental_data': validation_result['found_in_db'],
                'performance_summary': self._generate_performance_summary(validation_result)
            }
            enhanced_formulas.append(enhanced_item)
        
        enhanced_recommendations = {
            'chemical_formulas': enhanced_formulas,
            'synthesis_methods': base_recommendations['synthesis_methods'],
            'testing_procedures': base_recommendations['testing_procedures'],
            'validation_summary': self._generate_validation_summary(enhanced_formulas)
        }
        
        logger.info(f"✅ 完成增强推荐，验证了{len(enhanced_formulas)}个化学式")
        return enhanced_recommendations
    
    def get_ai_predictions_for_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """
        为推荐的化学式获取AI预测
        逻辑：如果有数据库数据就返回，如果没有就进行AI预测
        """
        logger.info(f"🤖 获取AI预测: {query}")
        
        base_recommendations = self.get_comprehensive_recommendations(query)
        
        prediction_results = []
        for formula_item in base_recommendations['chemical_formulas']:
            formula = formula_item['formula']
            
            if self.material_validator:
                validation_result = self.material_validator.validate_material(formula, enable_prediction=True)
            else:
                validation_result = {
                    'formula': formula,
                    'source': 'none',
                    'found_in_db': False,
                    'experimental_data': None,
                    'prediction': None,
                    'confidence': 'low',
                    'summary': "材料验证器不可用"
                }
            
            prediction_item = {
                **formula_item,
                'validation': validation_result,
                'has_experimental_data': validation_result['found_in_db'],
            }
            prediction_results.append(prediction_item)
        
        logger.info(f"✅ 完成AI预测，处理了{len(prediction_results)}个化学式")
        return prediction_results
    
    def _generate_performance_summary(self, validation_result: Dict[str, Any]) -> str:
        """生成性能摘要"""
        if validation_result['found_in_db']:
            data_count = len(validation_result['experimental_data'])
            return f"实验数据 ({data_count}条研究)"
        elif validation_result['source'] == 'prediction' and validation_result['prediction']:
            pred = validation_result['prediction']
            return f"AI预测 (置信度: {pred['confidence']:.2f})"
        elif validation_result['source'] == 'none':
            return "仅推荐数据 (无性能验证)"
        else:
            return "无性能数据"
    
    def _generate_validation_summary(self, enhanced_formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证摘要"""
        total = len(enhanced_formulas)
        experimental_count = sum(1 for item in enhanced_formulas if item['has_experimental_data'])
        predicted_count = sum(1 for item in enhanced_formulas if not item['has_experimental_data'] and item['validation']['prediction'])
        
        return {
            'total_formulas': total,
            'experimental_data_count': experimental_count,
            'predicted_count': predicted_count,
            'coverage_rate': experimental_count / total if total > 0 else 0
        }
    
    def clear_database(self):
        """清空数据库"""
        try:
            # 清空FAISS索引和数据
            self._clear_index()
            
            # 删除持久化文件
            for file_path in [self.index_file, self.documents_file, self.metadatas_file, self.doc_ids_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            self.is_initialized = False
            logger.info("数据库已清空并重建")
            
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息（避免使用 count/query_texts，全部用 get(limit=...)）
        """
        try:
            # 估算总量：分批 get，只拿 id（避免拉 embeddings/documents）
            total = len(self.documents)

            # 按类型统计（仅判断是否存在）
            type_counts = {}
            for doc_type in ['general', 'chemical_formula', 'synthesis_method', 'testing_procedure']:
                try:
                    got = sum(1 for meta in self.metadatas if meta.get('type') == doc_type)
                    type_counts[doc_type] = got
                except Exception:
                    type_counts[doc_type] = 0
            
            return {
                'total_documents': total,
                'type_distribution': type_counts,
                'is_initialized': self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                'total_documents': 0,
                'type_distribution': {},
                'is_initialized': False
            }

    def get_direct_recommendations(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        直接推荐流程：中文转英文 -> 英文搜索文档 -> 获取DOI文献 -> LLM分析 -> 返回结构化结果
        
        Args:
            query: 用户查询（如"异质结"）
            n_results: 搜索结果数量
            
        Returns:
            包含化学式、合成工艺、测试流程的字典，以及英文查询
        """
        logger.info(f"🔍 直接推荐: {query}")
        
        # 步骤1: 中文转专业英文
        from .llm_handler import LLMHandler
        from .config_cache import config_cache
        
        # 创建LLM处理器并加载配置
        llm_handler = LLMHandler()
        config = config_cache.load_config()
        if config:
            llm_handler.set_api_config(config)
        
        english_query = query  # 默认使用原查询
        if llm_handler.is_api_ready():
            english_query = llm_handler.translate_to_professional_english(query)
            logger.info(f"🌐 英文查询: {english_query}")
        else:
            logger.warning("LLM未配置，跳过翻译步骤")
        
        # 步骤2: 使用英文查询搜索相关文档，只获取前2个最匹配的DOI
        search_results = self.search_similar(english_query, n_results=n_results)
        
        if not search_results:
            logger.warning("未找到相关文档")
            return {
                "original_query": query,
                "english_query": english_query,
                "chemical_formulas": [],
                "synthesis_methods": [],
                "testing_procedures": [],
                "error": "未找到相关文档"
            }
        
        # 步骤3: 对搜索结果进行去重，只保留前2个最匹配的结果，并提取DOI
        unique_results, top_dois = self._get_unique_top_results(search_results, max_results=2)
        logger.info("=" * 60)
        logger.info("📄 去重和DOI提取详情:")
        logger.info("=" * 60)
        logger.info(f"🔍 从 {len(search_results)} 个搜索结果中去重后保留 {len(unique_results)} 个")
        logger.info(f"📋 提取到的DOI: {top_dois}")
        
        # 显示去重后的结果信息
        for i, result in enumerate(unique_results, 1):
            meta = result.get('metadata', {})
            doi = meta.get('doi', 'N/A')
            score = result.get('similarity_score', 0)
            title = meta.get('title', '未知')
            logger.info(f"  去重结果{i}: DOI={doi}, 相似度={score:.3f}, 标题={title[:50]}...")
        logger.info("=" * 60)
        
        # 步骤4: 根据DOI从all_mxene.json中获取完整文献信息
        full_literature_data = self._get_literature_by_dois(top_dois)
        logger.info("📚 文献检索详情:")
        logger.info(f"📄 检索到 {len(full_literature_data)} 篇完整文献")
        
        for i, paper in enumerate(full_literature_data, 1):
            title = paper.get('title', '未知')
            authors = paper.get('authors', '未知')
            year = paper.get('year', '未知')
            doi = paper.get('doi', '未知')
            logger.info(f"  文献{i}: {title[:60]}... (作者: {authors[:30]}..., 年份: {year}, DOI: {doi})")
        logger.info("=" * 60)
        
        # 步骤5: 组合搜索结果和文献数据
        # 组合RAG搜索结果（文本格式）和all.json数据（JSON格式）
        
        # 第一部分：RAG向量搜索结果（文本格式）
        context_text = self._format_search_results_for_llm(unique_results)
        literature_text = self._format_literature_for_llm(full_literature_data)
        
        # 第二部分：尝试获取all.json格式的结构化数据
        json_context = self._format_literature_as_json(full_literature_data, unique_results)
        
        # 组合两部分数据
        if json_context:
            combined_context = f"""
【第一部分：RAG向量搜索结果】
{context_text}

【第二部分：完整文献信息】
{literature_text}

【第三部分：结构化材料数据（all.json格式）】
{json_context}
"""
            logger.info("📊 使用组合格式：RAG搜索结果 + 结构化JSON数据")
        else:
            combined_context = f"""
【RAG搜索结果】
{context_text}

【完整文献信息】
{literature_text}
"""
            logger.info("📝 使用文本格式：仅RAG搜索结果")
        
        if not llm_handler.is_api_ready():
            logger.warning("LLM未配置，返回基础搜索结果")
            result = self._fallback_to_basic_search(search_results)
            result["original_query"] = query
            result["english_query"] = english_query
            result["dois_found"] = top_dois
            result["literature_count"] = len(full_literature_data)
            return result
        
        # 步骤6: 让LLM分析搜索结果和文献信息并返回JSON格式
        structured_result = llm_handler.analyze_and_extract_recommendations(english_query, combined_context)
        
        # 添加查询信息
        structured_result["original_query"] = query
        structured_result["english_query"] = english_query
        structured_result["dois_found"] = top_dois
        structured_result["literature_count"] = len(full_literature_data)
        
        # 步骤7: 处理新格式的JSON结构
        if structured_result.get("content") and isinstance(structured_result["content"], list):
            # 新的JSON格式：提取化学式进行数据库增强
            content_list = structured_result["content"]
            enhanced_content = []
            
            for item in content_list:
                if "material" in item and "chemical_formula" in item["material"]:
                    formula = item["material"]["chemical_formula"]
                    # 为这个化学式创建一个临时的旧格式对象来使用现有的增强逻辑
                    temp_formula = {"formula": formula}
                    enhanced_formula = self._enhance_formulas_with_data([temp_formula])
                    
                    # 将增强的验证信息添加到材料信息中
                    if enhanced_formula and len(enhanced_formula) > 0:
                        validation = enhanced_formula[0].get('validation', {})
                        if validation:
                            item["material"]["validation"] = validation
                            item["material"]["has_experimental_data"] = validation.get('found_in_db', False)
                
                enhanced_content.append(item)
            
            structured_result["content"] = enhanced_content
            logger.info(f"✅ 完成直接推荐（新格式），返回 {len(enhanced_content)} 个材料")
        elif structured_result.get("chemical_formulas"):
            # 兼容旧格式
            enhanced_formulas = self._enhance_formulas_with_data(structured_result["chemical_formulas"])
            structured_result["chemical_formulas"] = enhanced_formulas
            logger.info(f"✅ 完成直接推荐（旧格式），返回 {len(structured_result.get('chemical_formulas', []))} 个化学式")
        else:
            logger.info("✅ 完成推荐，但未找到有效的材料信息")
        
        return structured_result
    
    def _format_search_results_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为LLM可理解的文本"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            doc = result.get('document', '')
            meta = result.get('metadata', {})
            score = result.get('similarity_score', 0)
            
            context_part = f"文档 {i} (相似度: {score:.3f}):\n"
            context_part += f"标题: {meta.get('title', '未知')}\n"
            context_part += f"内容: {doc[:500]}...\n"  # 限制长度避免超出token限制
            if meta.get('doi'):
                context_part += f"DOI: {meta['doi']}\n"
            context_part += "\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _fallback_to_basic_search(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """当LLM不可用时的备用方案"""
        # 简单提取一些基本信息
        formulas = []
        methods = []
        procedures = []
        
        for result in search_results[:3]:  # 只取前3个结果
            meta = result.get('metadata', {})
            doc = result.get('document', '')
            
            # 简单的关键词匹配来分类
            if meta.get('type') == 'chemical_formula' or 'formula' in meta.get('value', '').lower():
                formulas.append({
                    "formula": meta.get('value', '未知'),
                    "source": meta.get('title', '未知'),
                    "doi": meta.get('doi', ''),
                    "confidence": "medium"
                })
            elif 'synthesis' in doc.lower() or 'preparation' in doc.lower():
                methods.append({
                    "method": doc[:100] + "...",
                    "source": meta.get('title', '未知'),
                    "doi": meta.get('doi', ''),
                    "confidence": "medium"
                })
            elif 'test' in doc.lower() or 'measurement' in doc.lower():
                procedures.append({
                    "procedure": doc[:100] + "...",
                    "source": meta.get('title', '未知'),
                    "doi": meta.get('doi', ''),
                    "confidence": "medium"
                })
        
        return {
            "chemical_formulas": formulas,
            "synthesis_methods": methods,
            "testing_procedures": procedures,
            "fallback": True
        }
    
    def _enhance_formulas_with_data(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用数据库查询或AI预测增强化学式信息"""
        enhanced_formulas = []
        
        for formula_item in formulas:
            formula = formula_item.get('formula', '')
            if not formula:
                continue
            
            # 查询数据库或进行预测
            if self.material_validator:
                validation_result = self.material_validator.validate_material(formula, enable_prediction=True)
                enhanced_item = {
                    **formula_item,
                    'validation': validation_result,
                    'has_experimental_data': validation_result.get('found_in_db', False),
                    'performance_data': validation_result.get('experimental_data') or validation_result.get('prediction')
                }
            else:
                enhanced_item = formula_item
            
            enhanced_formulas.append(enhanced_item)
        
        return enhanced_formulas

    def _get_unique_top_results(self, search_results: List[Dict[str, Any]], max_results: int = 2) -> Tuple[List[Dict[str, Any]], List[str]]:
        """对搜索结果进行去重，返回前N个唯一结果和对应的DOI列表"""
        unique_results = []
        dois = []
        seen_identifiers = set()
        seen_dois = set()
        
        for result in search_results:
            meta = result.get('metadata', {})
            
            # 创建唯一标识符：优先使用DOI，其次使用标题
            identifier = None
            if meta.get('doi'):
                identifier = meta['doi'].strip().lower()
            elif meta.get('title'):
                identifier = meta['title'].strip().lower()
            else:
                # 如果都没有，使用内容的前50个字符作为标识符
                doc = result.get('document', '')
                identifier = doc[:50].strip().lower()
            
            # 如果是唯一的结果，则添加
            if identifier and identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_results.append(result)
                
                # 同时提取DOI
                doi = meta.get('doi', '')
                if doi and doi not in seen_dois:
                    # 清理和验证DOI
                    if doi.startswith('http'):
                        # 提取DOI部分，如 https://doi.org/10.1016/xxx -> 10.1016/xxx
                        if '/10.' in doi:
                            doi = doi.split('/10.', 1)[1]
                            doi = '10.' + doi
                    elif not doi.startswith('10.'):
                        # 跳过不是标准DOI格式的
                        doi = None
                    
                    if doi:
                        dois.append(doi)
                        seen_dois.add(doi)
                
                # 只保留前N个去重后的结果
                if len(unique_results) >= max_results:
                    break
        
        return unique_results, dois

    def _extract_top_dois(self, search_results: List[Dict[str, Any]], max_dois: int = 2) -> List[str]:
        """从搜索结果中提取前N个最匹配的DOI"""
        dois = []
        seen_dois = set()
        
        for result in search_results:
            meta = result.get('metadata', {})
            doi = meta.get('doi', '')
            
            # 清理和验证DOI
            if doi and doi not in seen_dois:
                # 简单清理DOI格式
                if doi.startswith('http'):
                    # 提取DOI部分，如 https://doi.org/10.1016/xxx -> 10.1016/xxx
                    if '/10.' in doi:
                        doi = doi.split('/10.', 1)[1]
                        doi = '10.' + doi
                elif not doi.startswith('10.'):
                    # 跳过不是标准DOI格式的
                    continue
                
                dois.append(doi)
                seen_dois.add(doi)
                
                if len(dois) >= max_dois:
                    break
        
        return dois
    
    def _get_literature_by_dois(self, dois: List[str]) -> List[Dict[str, Any]]:
        """根据DOI列表从data/all_mxene.json中获取完整文献信息"""
        if not dois:
            return []
        
        # 加载all_mxene.json数据
        all_data = self.data_loader.load_all_mxene_json()
        if not all_data:
            logger.warning("无法加载all_mxene.json数据")
            return []
        
        found_literature = []
        seen_dois = set()  # 用于去重
        
        for doi in dois:
            # 跳过已经处理过的DOI
            if doi in seen_dois:
                logger.info(f"🔄 DOI {doi} 已处理过，跳过重复")
                continue
            
            # 在all_mxene.json中搜索匹配的DOI
            matching_papers = []
            for paper in all_data:
                paper_doi = paper.get('doi', '')
                
                # 多种DOI匹配方式
                if self._is_doi_match(paper_doi, doi):
                    # 检查是否已经添加过这篇文献（基于DOI和标题）
                    paper_id = self._get_paper_id(paper)
                    if not any(self._get_paper_id(existing) == paper_id for existing in found_literature):
                        matching_papers.append(paper)
            
            if matching_papers:
                logger.info(f"📖 DOI {doi} 找到 {len(matching_papers)} 条文献")
                found_literature.extend(matching_papers)
                seen_dois.add(doi)
            else:
                logger.warning(f"⚠️ DOI {doi} 未在all_mxene.json中找到匹配文献")
        
        # 最终去重检查
        unique_literature = self._deduplicate_literature(found_literature)
        if len(unique_literature) != len(found_literature):
            logger.info(f"🧹 去重完成：{len(found_literature)} → {len(unique_literature)} 篇文献")
        
        return unique_literature
    
    def _is_doi_match(self, paper_doi: str, target_doi: str) -> bool:
        """判断两个DOI是否匹配"""
        if not paper_doi or not target_doi:
            return False
        
        # 清理和标准化DOI
        def clean_doi(doi):
            doi = doi.strip().lower()
            if 'doi.org/' in doi:
                doi = doi.split('doi.org/')[-1]
            elif doi.startswith('http'):
                if '/10.' in doi:
                    doi = doi.split('/10.', 1)[1]
                    doi = '10.' + doi
            return doi
        
        clean_paper_doi = clean_doi(paper_doi)
        clean_target_doi = clean_doi(target_doi)
        
        return clean_paper_doi == clean_target_doi
    
    def _get_paper_id(self, paper: Dict[str, Any]) -> str:
        """生成文献的唯一标识符"""
        doi = paper.get('doi', '')
        title = paper.get('title', '')
        authors = paper.get('authors', '')
        
        # 使用DOI、标题和作者的组合作为唯一标识
        if doi:
            # 清理DOI格式用作标识
            clean_doi = doi.lower().strip()
            if 'doi.org/' in clean_doi:
                clean_doi = clean_doi.split('doi.org/')[-1]
            return f"doi:{clean_doi}"
        elif title:
            return f"title:{title.lower().strip()[:100]}"  # 使用标题前100字符
        else:
            return f"authors:{authors.lower().strip()[:50]}"  # 使用作者前50字符
    
    def _deduplicate_literature(self, literature: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复的文献"""
        seen_ids = set()
        unique_literature = []
        
        for paper in literature:
            paper_id = self._get_paper_id(paper)
            if paper_id not in seen_ids:
                unique_literature.append(paper)
                seen_ids.add(paper_id)
            else:
                title = paper.get('title', '未知')[:50]
                logger.debug(f"🗑️ 跳过重复文献: {title}...")
        
        return unique_literature
    
    def _format_literature_for_llm(self, literature_data: List[Dict[str, Any]]) -> str:
        """将完整文献数据格式化为LLM可理解的文本"""
        if not literature_data:
            return "未找到完整文献信息。"
        
        literature_parts = []
        
        for i, paper in enumerate(literature_data, 1):
            literature_part = f"完整文献 {i}:\n"
            literature_part += f"标题: {paper.get('title', '未知')}\n"
            literature_part += f"作者: {paper.get('authors', '未知')}\n"
            literature_part += f"期刊: {paper.get('journal', '未知')}\n"
            literature_part += f"年份: {paper.get('year', '未知')}\n"
            literature_part += f"DOI: {paper.get('doi', '未知')}\n"
            
            # 摘要
            if paper.get('abstract'):
                literature_part += f"摘要: {paper['abstract'][:500]}...\n"
            
            # 关键词
            if paper.get('keywords'):
                literature_part += f"关键词: {', '.join(paper['keywords'])}\n"
            
            # 化学式信息
            if paper.get('chemical_formula'):
                literature_part += f"化学式: {paper['chemical_formula']}\n"
            
            # 合成方法信息  
            if paper.get('synthesis_method'):
                literature_part += f"合成方法: {paper['synthesis_method']}\n"
            
            # 测试方法信息
            if paper.get('testing_procedure'):
                literature_part += f"测试流程: {paper['testing_procedure']}\n"
            
            # 性能数据
            if paper.get('performance_data'):
                literature_part += f"性能数据: {paper['performance_data']}\n"
            
            literature_part += "\n"
            literature_parts.append(literature_part)
        
        return "\n".join(literature_parts)
    
    def _format_literature_as_json(self, literature_data: List[Dict[str, Any]], search_results: List[Dict[str, Any]]) -> str:
        """将文献数据格式化为JSON格式供新prompt使用"""
        if not literature_data:
            return None
            
        # 加载all.json数据
        all_json_data = self.data_loader.load_all_json()
        if not all_json_data:
            logger.warning("无法加载all.json")
            return None
        
        # 收集所有匹配DOI的JSON记录
        matching_json_records = []
        
        for paper in literature_data:
            paper_doi = paper.get('doi', '')
            if not paper_doi:
                continue
                
            # 在all.json中找到所有匹配这个DOI的记录
            for json_record in all_json_data:
                record_doi = json_record.get('doi', '')
                
                # DOI匹配检查
                if self._is_doi_match(record_doi, paper_doi):
                    # 直接添加完整的JSON记录，不做任何修改
                    matching_json_records.append(json_record)
        
        if matching_json_records:
            # 将匹配的JSON记录完整输出
            import json
            try:
                formatted_json = json.dumps(matching_json_records, ensure_ascii=False, indent=2)
                logger.info(f"📊 找到 {len(matching_json_records)} 个匹配的JSON记录")
                return f"all.json中匹配的完整记录：\n{formatted_json}"
            except Exception as e:
                logger.warning(f"JSON序列化失败: {e}")
                return None
        
        logger.info("未找到匹配的JSON记录")
        return None
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取JSON格式的材料数据"""
        import re
        import json
        
        # 查找JSON对象的模式
        json_patterns = [
            r'\{[^{}]*"chemical_formula"[^{}]*\}',  # 简单的单层JSON
            r'\{(?:[^{}]|{[^{}]*})*"chemical_formula"(?:[^{}]|{[^{}]*})*\}',  # 包含嵌套的JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # 尝试解析JSON
                    data = json.loads(match)
                    if isinstance(data, dict) and 'chemical_formula' in data:
                        return data
                except:
                    continue
        
        # 如果没有找到完整的JSON，尝试提取关键信息构建基本JSON
        return self._construct_basic_json_from_text(text)
    
    def _construct_basic_json_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中构建基本的JSON结构"""
        import re
        
        # 提取化学式
        formula_patterns = [
            r'\b[A-Z][a-z]?\d*[A-Z][a-z]?\d*T?x?\b',  # 基本MXene模式如Ti3C2Tx
            r'\b[A-Z][a-z]?\d*C\d*T?x?\b',  # 碳化物模式
            r'\b[A-Z][a-z]?-MXene\b',  # 明确的MXene标记
        ]
        
        formulas = []
        for pattern in formula_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            formulas.extend(matches)
        
        if not formulas:
            return None
            
        # 构建基本JSON结构
        basic_json = {
            "material": {
                "chemical_formula": formulas[0] if formulas else "未知"
            }
        }
        
        # 尝试提取性能数据
        performance = {}
        
        # 查找RL值
        rl_patterns = [
            r'RL.*?(-?\d+\.?\d*)\s*dB',
            r'reflection\s*loss.*?(-?\d+\.?\d*)\s*dB',
            r'反射损耗.*?(-?\d+\.?\d*)\s*dB'
        ]
        
        for pattern in rl_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                performance['rl_min'] = f"{match.group(1)} dB"
                break
        
        # 查找EAB值
        eab_patterns = [
            r'EAB.*?(\d+\.?\d*)\s*GHz',
            r'effective\s*absorption\s*bandwidth.*?(\d+\.?\d*)\s*GHz',
            r'有效吸收带宽.*?(\d+\.?\d*)\s*GHz'
        ]
        
        for pattern in eab_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                performance['eab'] = f"{match.group(1)} GHz"
                break
        
        if performance:
            basic_json['performance'] = performance
        
        return basic_json
