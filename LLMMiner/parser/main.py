import os
import sys
import json
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 添加当前目录到系统路径，以便能够导入同目录下的模块
sys.path.append(str(current_dir))

from typing import Dict, Any, List, Annotated, Tuple
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END  # 导入END常量

# 使用绝对导入，但不指定LLMMiner前缀
from abstract import AbstractAnalyzer
from field_extractor import FieldExtractor
from content_extractor import ContentExtractor
from content_evaluator import ContentEvaluator
from content_optimizer import ContentOptimizer
from utils import load_config, get_temperature, get_max_retries, setup_logging

# 初始化日志系统
log = setup_logging()

# 确保日志对象不为None，如果为None则创建一个回退日志记录器
if log is None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log = logging.getLogger("LLMMiner.parser.fallback")
    log.warning("使用回退日志记录器，原始setup_logging()返回None")

import time

class ProcessState(TypedDict):
    text: str
    abstract_analysis: Dict[str, bool]
    fields: List[str]
    content: List[Dict[str, Any]]
    evaluation_result: Dict[str, Any]
    optimization_attempts: int

class MaterialParser:
    def __init__(self, prompt_file: str = "prompt.yaml"):  # 使用相对路径
        # 加载配置
        self.config = load_config()
        
        # 获取默认模型
        # self.default_model = self.config.get("models", {}).get("moonshot_version")
        # self.default_model = "kimi"
        self.default_model = "gemini"
        
        # 获取各阶段的温度参数
        self.abstract_temp = get_temperature("abstract_analysis")
        self.field_temp = get_temperature("field_extraction")
        self.content_temp = get_temperature("content_extraction")
        self.eval_temp = get_temperature("content_evaluation")
        self.opt_temp = get_temperature("content_optimization")
        
        # 初始化各组件
        self.abstract_analyzer = AbstractAnalyzer(prompt_file)
        self.field_extractor = FieldExtractor(prompt_file)
        self.content_extractor = ContentExtractor(prompt_file)
        self.content_evaluator = ContentEvaluator(prompt_file)
        self.content_optimizer = ContentOptimizer(prompt_file)
        
        # Create state graph
        self.graph = StateGraph(ProcessState)
        # Build the processing pipeline
        self.workflow = self.build_pipeline()
        
    def visualize_pipeline(self, output_path: str = "parser_graph.png"):
        """
        可视化解析器的图结构
        
        Args:
            output_path: 输出图像的路径
        """
        try:
            # 获取当前脚本所在目录
            current_dir = Path(__file__).parent.absolute()
            
            # 如果提供的是相对路径，转换为绝对路径
            if not os.path.isabs(output_path):
                output_path = os.path.join(current_dir, output_path)
                
            # 尝试方法1: 按照demo.py中的方法 - graph.get_graph().draw_mermaid_png()
            try:
                # 获取内部图形结构
                graph_obj = self.graph
                if hasattr(graph_obj, 'get_graph'):
                    inner_graph = graph_obj.get_graph()
                    if hasattr(inner_graph, 'draw_mermaid_png'):
                        graph_image = inner_graph.draw_mermaid_png()
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # 保存到文件
                        with open(output_path, "wb") as f:
                            f.write(graph_image)
                            
                        log.info(f"图结构已保存为 {output_path}")
                        return True
                    else:
                        log.warning("内部图形对象没有draw_mermaid_png方法")
                else:
                    log.warning("StateGraph对象没有get_graph方法")
                    
                # 尝试方法2: 直接使用graph.draw_mermaid_png()
                if hasattr(graph_obj, 'draw_mermaid_png'):
                    graph_image = graph_obj.draw_mermaid_png()
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 保存到文件
                    with open(output_path, "wb") as f:
                        f.write(graph_image)
                        
                    log.info(f"图结构已保存为 {output_path}")
                    return True
                else:
                    log.warning("StateGraph对象没有draw_mermaid_png方法")
                    
                # 尝试方法3: 使用workflow.get_graph().draw_mermaid_png()
                if hasattr(self.workflow, 'get_graph'):
                    inner_graph = self.workflow.get_graph()
                    if hasattr(inner_graph, 'draw_mermaid_png'):
                        graph_image = inner_graph.draw_mermaid_png()
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # 保存到文件
                        with open(output_path, "wb") as f:
                            f.write(graph_image)
                            
                        log.info(f"图结构已保存为 {output_path}")
                        return True
                    else:
                        log.warning("workflow的内部图形对象没有draw_mermaid_png方法")
                else:
                    log.warning("workflow对象没有get_graph方法")
                
                # 如果以上方法都失败，再使用其他备选方案
                raise AttributeError("无法找到可用的图形绘制方法")
                
            except AttributeError as e:
                log.error(f"尝试使用demo.py中的方法失败: {e}")
                
                # 最后的备选方案：使用PIL库手动绘制简单的图形
                try:
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # 创建一个简单的图像
                    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                    d = ImageDraw.Draw(img)
                    
                    # 尝试加载字体
                    try:
                        font = ImageFont.truetype("arial.ttf", 15)
                    except:
                        font = ImageFont.load_default()
                    
                    # 绘制节点和边
                    nodes = {
                        "abstract_analysis": (100, 100),
                        "field_extraction": (300, 100),
                        "content_extraction": (500, 100),
                        "evaluate_content": (400, 200),
                        "optimize_content": (200, 200),
                        "finished": (600, 200),
                        "END": (600, 300)
                    }
                    
                    # 绘制节点
                    for name, pos in nodes.items():
                        d.rectangle([pos[0]-50, pos[1]-20, pos[0]+50, pos[1]+20], outline=(0, 0, 0), fill=(220, 220, 220))
                        d.text((pos[0], pos[1]), name, fill=(0, 0, 0), font=font, anchor="mm")
                    
                    # 绘制边
                    edges = [
                        ("abstract_analysis", "field_extraction"),
                        ("field_extraction", "content_extraction"),
                        ("content_extraction", "evaluate_content"),
                        ("evaluate_content", "optimize_content"),
                        ("evaluate_content", "finished"),
                        ("optimize_content", "evaluate_content"),
                        ("finished", "END")
                    ]
                    
                    for start, end in edges:
                        start_pos = nodes[start]
                        end_pos = nodes[end]
                        d.line([start_pos[0], start_pos[1], end_pos[0], end_pos[1]], fill=(0, 0, 0), width=1)
                    
                    # 保存图像
                    img.save(output_path)
                    log.info(f"使用PIL生成的简单图结构已保存为 {output_path}")
                    return True
                except ImportError:
                    log.error("无法导入PIL库。请使用 'pip install pillow' 安装。")
                
                # 如果所有图像生成方法都失败，保存为文本文件
                mermaid_description = """
                graph TD
                    abstract_analysis[Abstract Analysis] --> field_extraction[Field Extraction]
                    field_extraction --> content_extraction[Content Extraction]
                    content_extraction --> evaluate_content[Evaluation]
                    evaluate_content -->|needs optimization| optimize_content[Optimization]
                    evaluate_content -->|perfect/max attempts| finished[Finished]
                    optimize_content --> evaluate_content
                    finished --> END[End]
                """
                
                text_path = output_path.replace('.png', '.txt')
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(mermaid_description)
                log.info(f"已将自定义Mermaid描述保存为文本文件：{text_path}")
                return False
                
        except Exception as e:
            log.error(f"无法生成图结构可视化: {e}")
            log.warning("注意: 这个功能需要在支持可视化的环境中运行，如Jupyter Notebook或支持IPython的环境。")
            return False
        
    def analyze_abstract(self, state: ProcessState) -> ProcessState:
        """Analyze the abstract text"""
        state["abstract_analysis"] = self.abstract_analyzer.analyze(
            state["text"],
            model_name=self.default_model,
            temperature=self.abstract_temp
        )
        log.info(f"Abstract analysis completed. Status: {state['abstract_analysis']['status']}")
        return state
        
    def extract_fields(self, state: ProcessState) -> ProcessState:
        """Extract fields from the text"""
        # Only continue if it's a relevant paper
        if not (state["abstract_analysis"]["is_mxene_material"] and 
                (state["abstract_analysis"]["is_absorption_study"] or 
                 state["abstract_analysis"]["is_emi_shielding"])):
            state["fields"] = []
            log.info("Paper does not meet criteria for field extraction. Skipping.")
            return state
            
        state["fields"] = self.field_extractor.extract(
            state["text"],
            model_name=self.default_model,
            temperature=self.field_temp
        )
        log.info(f"Field extraction completed. Found {len(state['fields'])} fields.")
        return state
        
    def extract_content(self, state: ProcessState) -> ProcessState:
        """Extract detailed content for each field"""
        if not state["fields"]:
            state["content"] = []
            log.info("No fields to extract content for. Skipping content extraction.")
            return state
            
        state["content"] = self.content_extractor.extract(
            state["text"], 
            state["fields"],
            model_name=self.default_model,
            temperature=self.content_temp
        )
        log.info(f"Content extraction completed. Extracted {len(state['content'])} items.")
        return state
        
    def evaluate_content(self, state: ProcessState) -> ProcessState:
        """Evaluate the extracted content"""
        state["evaluation_result"] = self.content_evaluator.evaluate(
            state["text"],
            state["content"],
            state["fields"],
            model_name=self.default_model,
            temperature=self.eval_temp
        )
        log.info(f"Content evaluation completed. Status: {state['evaluation_result']['status']}")
        return state
        
    def optimize_content(self, state: ProcessState) -> ProcessState:
        """Optimize the content based on evaluation results"""
        state["content"] = self.content_optimizer.optimize(
            state["content"],
            state["evaluation_result"],
            model_name=self.default_model,
            temperature=self.opt_temp
        )
        state["optimization_attempts"] += 1
        log.info(f"Content optimization completed. Attempts: {state['optimization_attempts']}")
        return state
        
    def _route_after_evaluation(self, state: ProcessState) -> str:
        """Determine next step after evaluation"""
        # If perfect or too many optimization attempts, move to end
        if (state["evaluation_result"]["status"] == "perfect" or 
            state["optimization_attempts"] >= 3):
            log.info(f"Evaluation status is perfect or optimization attempts reached max. Moving to finished.")
            return "finished"  # 使用一个正常节点名
            
        # Otherwise, try optimization
        log.info(f"Evaluation status is not perfect. Attempting optimization. Current attempts: {state['optimization_attempts']}")
        return "optimize_content"
        
    def finished(self, state: ProcessState) -> ProcessState:
        """最终处理，在结束前执行一些清理或最终操作"""
        # 这里可以添加任何需要在流程结束前执行的操作
        log.info("Finished node reached. Finalizing state.")
        return state
        
    def build_pipeline(self) -> StateGraph:
        """Build the processing pipeline using langgraph"""
        # Add nodes for each processing step
        self.graph.add_node("abstract_analysis", self.analyze_abstract)
        self.graph.add_node("field_extraction", self.extract_fields)
        self.graph.add_node("content_extraction", self.extract_content)
        self.graph.add_node("evaluate_content", self.evaluate_content)
        self.graph.add_node("optimize_content", self.optimize_content)
        self.graph.add_node("finished", self.finished)
        
        # Set entry point
        self.graph.set_entry_point("abstract_analysis")
        
        # Add sequential edges
        self.graph.add_edge("abstract_analysis", "field_extraction")
        self.graph.add_edge("field_extraction", "content_extraction")
        self.graph.add_edge("content_extraction", "evaluate_content")
        
        # Add conditional edge after evaluation
        self.graph.add_conditional_edges(
            "evaluate_content",
            self._route_after_evaluation,
            {
                "optimize_content": "optimize_content",
                "finished": "finished"
            }
        )
        
        # Add edge from optimization back to evaluation
        self.graph.add_edge("optimize_content", "evaluate_content")
        
        # 从finished节点连接到END
        self.graph.add_edge("finished", END)
        
        # Compile the graph
        return self.graph.compile()
        
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the input text through the complete pipeline
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: Complete analysis results including:
                - abstract_analysis: Classification results
                - fields: List of identified fields
                - content: Detailed content for each field
                - evaluation_result: Final evaluation results
        """
        # Initialize state
        initial_state: ProcessState = {
            "text": text,
            "abstract_analysis": {},
            "fields": [],
            "content": [],
            "evaluation_result": {},
            "optimization_attempts": 0
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "abstract_analysis": final_state["abstract_analysis"],
            "fields": final_state["fields"],
            "content": final_state["content"],
            "evaluation_result": final_state["evaluation_result"]
        }

def get_processed_dois(file_path="extract.json"):
    """
    从JSON文件中读取已处理过的DOI列表
    
    Args:
        file_path: JSON文件路径，默认为项目根目录下的extract.json
        
    Returns:
        set: 已处理过的DOI集合
    """
    processed_dois = set()
    try:
        # 如果文件路径不是绝对路径，则转换为相对于项目根目录的绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path)
        
        # 如果文件存在，读取DOI
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 确保existing_data是列表
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
                
                # 提取所有DOI
                for item in existing_data:
                    if isinstance(item, dict) and "doi" in item:
                        doi = clean_doi(item["doi"])
                        if doi and doi != "Unknown":
                            processed_dois.add(doi)
                
                log.info(f"从 {file_path} 读取到 {len(processed_dois)} 个已处理的DOI")
            except json.JSONDecodeError:
                log.warning(f"文件 {file_path} 格式不正确")
    except Exception as e:
        log.error(f"读取已处理DOI失败: {e}")
    
    return processed_dois

def clean_doi(doi_string):
    """
    清理DOI字符串，删除可能存在的http://或https://前缀
    
    Args:
        doi_string: 原始DOI字符串
        
    Returns:
        str: 清理后的DOI字符串
    """
    if doi_string and isinstance(doi_string, str):
        # 删除http://或https://前缀
        if doi_string.startswith("http://"):
            doi_string = doi_string[7:]
        elif doi_string.startswith("https://"):
            doi_string = doi_string[8:]
    return doi_string

def save_to_json(result, file_path="extract.json"):
    """
    将解析结果保存到JSON文件
    
    Args:
        result: 解析结果，可以是单个字典或字典列表
        file_path: JSON文件路径，默认为项目根目录下的extract.json
    """
    # 确保我们一定会执行保存操作
    save_attempted = False
    
    # 如果文件路径不是绝对路径，则转换为相对于项目根目录的绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(project_root, file_path)
    
    # 计算clean_extract.json文件路径
    clean_file_path = os.path.join(os.path.dirname(file_path), "clean_" + os.path.basename(file_path))
    
    # 如果文件已存在，先读取现有内容
    existing_data = []
    try:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                # 确保existing_data是列表
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                # 如果文件格式不正确，重新开始
                log.warning(f"文件 {file_path} 格式不正确，将被覆盖")
                existing_data = []
            except Exception as e:
                # 其他任何读取错误，记录但继续
                log.warning(f"读取文件时出错: {e}，将创建新文件")
                existing_data = []
    except Exception as e:
        log.warning(f"检查文件存在时出错: {e}，将创建新文件")
        existing_data = []
    
    # 读取clean文件现有内容
    existing_clean_data = []
    try:
        if os.path.exists(clean_file_path):
            try:
                with open(clean_file_path, 'r', encoding='utf-8') as f:
                    existing_clean_data = json.load(f)
                # 确保existing_clean_data是列表
                if not isinstance(existing_clean_data, list):
                    existing_clean_data = [existing_clean_data]
            except json.JSONDecodeError:
                # 如果文件格式不正确，重新开始
                log.warning(f"文件 {clean_file_path} 格式不正确，将被覆盖")
                existing_clean_data = []
            except Exception as e:
                # 其他任何读取错误，记录但继续
                log.warning(f"读取文件时出错: {e}，将创建新文件")
                existing_clean_data = []
    except Exception as e:
        log.warning(f"检查文件存在时出错: {e}，将创建新文件")
        existing_clean_data = []
    
    # 将新结果添加到现有数据中
    try:
        if isinstance(result, list):
            existing_data.extend(result)
            
            # 为每个结果创建clean版本并添加到clean数据中
            for item in result:
                if isinstance(item, dict):
                    clean_item = create_clean_item(item)
                    existing_clean_data.append(clean_item)
        else:
            existing_data.append(result)
            
            # 创建单个结果的clean版本
            if isinstance(result, dict):
                clean_item = create_clean_item(result)
                existing_clean_data.append(clean_item)
    except Exception as e:
        log.error(f"添加新结果到现有数据时出错: {e}")
        # 即使添加失败，也尝试保存原始结果
        try:
            if not isinstance(result, list):
                result = [result]
            existing_data = result
            
            # 创建clean版本
            existing_clean_data = []
            for item in result:
                if isinstance(item, dict):
                    clean_item = create_clean_item(item)
                    existing_clean_data.append(clean_item)
            
            log.warning("无法合并数据，将只保存当前结果")
        except:
            log.error("无法处理结果数据，将跳过保存")
            return False
    
    # 确保目录存在
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    except Exception as e:
        log.warning(f"创建目录时出错: {e}")
    
    # 写入extract.json文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"成功将结果保存到 {file_path}")
        save_attempted = True
    except Exception as e:
        log.error(f"保存结果到 {file_path} 失败: {e}")
        
        # 尝试使用备用方法保存
        try:
            # 使用临时文件名
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            log.info(f"已使用备用方法保存到 {backup_path}")
            save_attempted = True
        except Exception as e2:
            log.error(f"备用保存方法也失败: {e2}")
    
    # 写入clean_extract.json文件
    try:
        with open(clean_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_clean_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"成功将清洁版结果保存到 {clean_file_path}")
        save_attempted = True
    except Exception as e:
        log.error(f"保存清洁版结果到 {clean_file_path} 失败: {e}")
        
        # 尝试使用备用方法保存
        try:
            # 使用临时文件名
            backup_path = f"{clean_file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(existing_clean_data, f, ensure_ascii=False, indent=2)
            log.info(f"已使用备用方法保存清洁版结果到 {backup_path}")
        except Exception as e2:
            log.error(f"备用保存清洁版结果方法也失败: {e2}")
    
    return save_attempted

def create_clean_item(item):
    """
    创建一个按指定顺序排列的清洁版本的数据项
    
    Args:
        item: 原始数据项
        
    Returns:
        dict: 按指定顺序排列的数据项（只包含doi、content和fields）
    """
    clean_item = {}
    
    # 1. 首先添加doi (如果存在)
    if "doi" in item:
        clean_item["doi"] = item["doi"]
    
    # 2. 然后添加content (如果存在)
    if "content" in item:
        clean_item["content"] = item["content"]
    
    # 3. 最后添加fields字段 (如果存在)
    if "fields" in item:
        clean_item["fields"] = item["fields"]
    
    return clean_item

def format_duration(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes}min{sec}s"

# 在main()函数和process_paper()等主流程加时间记录
# 1. main()整体开始和结束
# 2. process_paper()每个主要步骤加耗时

def main():
    start_time = time.time()
    log.info("==== 解析流程开始 ====")
    # 创建解析器实例
    parser = MaterialParser()
    
    # 生成可视化
    parser.visualize_pipeline()
    
    # 获取已处理的DOI列表
    processed_dois = get_processed_dois()
    log.info(f"已有 {len(processed_dois)} 篇论文被处理过")
    
    # 读取JSON文件
    try:
        # 构建JSON文件的路径
        json_path = os.path.join(project_root, "Data", "springer_extracted.json")
        log.info(f"Attempting to read JSON file from: {json_path}")
        # 检查文件是否存在
        if not os.path.exists(json_path):
            log.warning(f"文件不存在: {json_path}")
            # 尝试列出Data目录中的文件
            data_dir = os.path.join(project_root, "Data")
            if os.path.exists(data_dir):
                log.info(f"Data目录中的文件:")
                for file in os.listdir(data_dir):
                    if file.endswith("_extracted.json"):
                        log.info(f" - {file}")
            
            exit(0)
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 检查是否为列表（多篇论文）或单个字典（单篇论文）
        if isinstance(data, list):
            log.info(f"Found {len(data)} papers.")
            # 使用for循环处理所有论文
            total_papers = len(data)
            mxene_papers = 0
            absorption_papers = 0
            review_papers = 0
            emi_shielding_papers = 0
            qualified_papers = 0  # 符合条件的论文（MXene材料相关 且 吸收研究相关 且 非综述 且 非EMI屏蔽）
            processed_papers = 0  # 成功完成全流程处理的论文
            
            for i, paper in enumerate(data):
                log.info(f"\nProcessing paper {i+1}/{total_papers}")
                
                # 检查DOI是否已经处理过
                paper_doi = clean_doi(paper.get("doi", "Unknown"))
                if paper_doi != "Unknown" and paper_doi in processed_dois:
                    log.info(f"论文DOI: {paper_doi} 已经处理过，跳过")
                    continue
                
                results = process_paper(parser, paper, processed_dois)
                #exit(0)
                if results:
                    # 统计各类论文数量
                    if results['abstract_analysis'].get('is_mxene_material', False):
                        mxene_papers += 1
                    if results['abstract_analysis'].get('is_absorption_study', False):
                        absorption_papers += 1
                    if results['abstract_analysis'].get('is_review_paper', False):
                        review_papers += 1
                    if results['abstract_analysis'].get('is_emi_shielding', False):
                        emi_shielding_papers += 1
                    
                    # 统计符合条件的论文
                    if (results['abstract_analysis'].get('is_mxene_material', False) and 
                        results['abstract_analysis'].get('is_absorption_study', False) and 
                        not results['abstract_analysis'].get('is_review_paper', False) and
                        not results['abstract_analysis'].get('is_emi_shielding', False)):
                        qualified_papers += 1
                    
                    # 统计成功完成全流程处理的论文
                    if 'fields' in results and results['fields'] and 'content' in results and results['content']:
                        processed_papers += 1
            
            # 打印统计信息
            log.info(f"\nStatistics:")
            log.info(f"Total papers: {total_papers}")
            log.info(f"MXene material related papers: {mxene_papers}")
            log.info(f"Absorption study related papers: {absorption_papers}")
            log.info(f"Review papers: {review_papers}")
            log.info(f"EMI shielding related papers: {emi_shielding_papers}")
            log.info(f"Qualified papers (MXene related and absorption study related and not review and not EMI shielding): {qualified_papers}")
            log.info(f"Processed papers: {processed_papers}")
            
        else:
            # 单篇论文
            # 检查DOI是否已经处理过
            paper_doi = clean_doi(data.get("doi", "Unknown"))
            if paper_doi != "Unknown" and paper_doi in processed_dois:
                log.info(f"论文DOI: {paper_doi} 已经处理过，跳过")
            else:
                process_paper(parser, data, processed_dois)
            
    except Exception as e:
        log.error(f"Error processing JSON file: {e}")
        # 使用示例文本作为备用
        text = """
        MXene materials, particularly Ti3C2Tx, have garnered significant attention due to their exceptional electromagnetic wave absorption properties. 
        In this study, we prepared Ti3C2Tx/polymer composites and investigated their microwave absorption performance in the frequency range of 2-18 GHz. 
        The results showed a maximum reflection loss of -45.6 dB at 12.4 GHz with a thickness of 1.8 mm, and an effective absorption bandwidth of 4.2 GHz. 
        The outstanding absorption properties can be attributed to the unique 2D layered structure of MXene, which facilitates multiple reflections and interfacial polarization.
        """
        
        # 直接调用abstract_analyzer
        abstract_analysis = parser.abstract_analyzer.analyze(
            text,
            model_name=parser.default_model,
            temperature=parser.abstract_temp
        )
        
        log.info("Parsing results (using example text):")
        log.info(abstract_analysis)
    # 在主流程最后加结束时间
    end_time = time.time()
    log.info(f"==== 解析流程结束，总耗时: {format_duration(end_time - start_time)} ====")

def process_paper(parser, paper, processed_dois=None):
    """处理单篇论文数据"""
    try:
        t0 = time.time()
        # 提取摘要和正文
        abstract = paper.get("abstract", "")
        text_parts = paper.get("text", [])

        if len(abstract) < 20:
                       # 整理返回结果
            result = {
                "doi": full_state["doi"]
            }
            return result
        
        # 将正文部分合并为单个字符串，跳过前5段引言部分
        text = "\n\n".join(text_parts[5:-2]) if text_parts and len(text_parts) > 5 else ("\n\n".join(text_parts) if text_parts else "")
        text = text + "\n\n".join(paper.get("table", []))
        # text = text[:15000]
        log.info(f"Paper DOI: {paper.get('doi', 'Unknown')}")
        log.info(f"Abstract length: {len(abstract)} characters")
        log.info(f"Text length: {len(text)} characters")
        
        # 首先只分析摘要
        # 直接调用abstract_analyzer
        t1 = time.time()
        print("\n" + "=" * 50)
        print("开始分析论文摘要 (使用流式输出)")
        print("=" * 50)
        abstract_analysis = parser.abstract_analyzer.analyze(
            abstract,
            model_name=parser.default_model,
            temperature=parser.abstract_temp,
            use_streaming=False  # 启用流式输出
        )
        t2 = time.time()
        log.info(f"[process_paper] 摘要分析耗时: {format_duration(t2-t1)}")
        
        log.info("\nAbstract analysis results:")
        log.info(f"MXene material related: {abstract_analysis.get('is_mxene_material', False)}")
        log.info(f"Absorption study related: {abstract_analysis.get('is_absorption_study', False)}")
        log.info(f"Is review paper: {abstract_analysis.get('is_review_paper', False)}")
        log.info(f"EMI shielding related: {abstract_analysis.get('is_emi_shielding', False)}")
        
        # 检查条件：前两个为True，后两个为False
        is_mxene = abstract_analysis.get('is_mxene_material', False)
        is_absorption = abstract_analysis.get('is_absorption_study', False)
        is_review = abstract_analysis.get('is_review_paper', False)
        is_emi = abstract_analysis.get('is_emi_shielding', False)
        
        # 只有当is_mxene和is_absorption为True，is_review和is_emi为False时才继续处理
        if is_mxene and is_absorption and not is_review and not is_emi:
            t3 = time.time()
            log.info("Meets processing criteria. Continuing with full text analysis...")
            
            # 创建新的状态，包含摘要分析结果和正文
            full_state = {
                "text": text,  # 只使用正文
                "abstract_analysis": abstract_analysis,  # 使用摘要的分析结果
                "fields": [],
                "content": [],
                "evaluation_result": {},
                "optimization_attempts": 0,
                "optimization_history": [],  # 新增字段，用于存储每次优化的历史记录
                "doi": clean_doi(paper.get("doi", "Unknown"))  # 使用clean_doi函数处理DOI
            }
            
            # 从field_extraction开始处理
            # 提取字段
            t4 = time.time()
            print("\n" + "=" * 50)
            print("开始提取文章字段 (使用流式输出)")
            print("=" * 50)
            fields = parser.field_extractor.extract(
                text,
                model_name=parser.default_model,
                temperature=parser.field_temp,
                use_streaming=False  # 启用流式输出
            )
            t5 = time.time()
            log.info(f"[process_paper] 字段提取耗时: {format_duration(t5-t4)}")
            full_state["fields"] = fields
            log.info(f"Extracted fields: {fields}")
            
            # 提取内容
            if fields:
                t6 = time.time()
                print("\n" + "=" * 50)
                print("开始提取文章详细内容 (使用流式输出)")
                print("=" * 50)
                content = parser.content_extractor.extract(
                    text, 
                    fields,
                    model_name=parser.default_model,
                    # model_name="kimi",
                    temperature=parser.content_temp,
                    use_streaming=False  # 启用流式输出
                )
                t7 = time.time()
                log.info(f"[process_paper] 内容提取耗时: {format_duration(t7-t6)}")
                full_state["content"] = content
                log.info(f"Extracted content: {content}")
                
                # 优化循环：最多尝试5次优化
                optimization_attempts = 0
                evaluation = {"status": "needs_optimization"}  # 初始状态
                
                while evaluation["status"] != "perfect" and optimization_attempts < 5:
                    t_loop_start = time.time()
                    # 先评估当前内容
                    t_eval = time.time()
                    print("\n" + "=" * 50)
                    print(f"开始第 {optimization_attempts+1} 次内容评估 (使用流式输出)")
                    print("=" * 50)
                    evaluation = parser.content_evaluator.evaluate(
                        text,
                        content,
                        fields,
                        model_name=parser.default_model,
                        #model_name="kimi",
                        temperature=parser.eval_temp,
                        use_streaming=False  # 启用流式输出
                    )
                    t_eval_end = time.time()
                    log.info(f"[process_paper] 内容评估耗时: {format_duration(t_eval_end-t_eval)}")
                    full_state["evaluation_result"] = evaluation
                    log.info(f"Content evaluation result: {evaluation}")
                    
                    # 记录本次评估结果
                    iteration_record = {
                        "iteration": optimization_attempts,
                        "evaluation": evaluation.copy(),
                        "content_before_optimization": [item.copy() for item in content] if content else []
                    }
                    
                    # 如果已经完美，跳出循环
                    if evaluation["status"] == "perfect":
                        log.info("Content is already perfect. No optimization needed.")
                        # 将最后一次评估记录添加到历史
                        full_state["optimization_history"].append(iteration_record)
                        break
                            
                    log.info(f"\nAttempting content optimization {optimization_attempts+1}...")
                    # 优化内容
                    t_opt = time.time()
                    print("\n" + "=" * 50)
                    print(f"开始第 {optimization_attempts+1} 次内容优化 (使用流式输出)")
                    print("=" * 50)
                    optimized_content = parser.content_optimizer.optimize(
                        content,
                        evaluation,
                        model_name=parser.default_model,
                        # model_name="kimi",
                        temperature=parser.opt_temp,
                        use_streaming=False  # 启用流式输出
                    )
                    t_opt_end = time.time()
                    log.info(f"[process_paper] 内容优化耗时: {format_duration(t_opt_end-t_opt)}")
                    full_state["content"] = optimized_content
                    
                    # 更新历史记录，添加优化后的内容
                    iteration_record["content_after_optimization"] = [item.copy() for item in optimized_content] if optimized_content else []
                    iteration_record["optimization_time"] = format_duration(t_opt_end-t_opt)
                    iteration_record["evaluation_time"] = format_duration(t_eval_end-t_eval)
                    full_state["optimization_history"].append(iteration_record)
                    
                    content = optimized_content  # 更新content变量，用于下次优化
                    
                    optimization_attempts += 1
                    log.info(f"Completed optimization attempt {optimization_attempts}")
                    log.info(f"Added optimization record to history. Total records: {len(full_state['optimization_history'])}")
                
                full_state["optimization_attempts"] = optimization_attempts
                
                # 打印结果
            log.info(f"\nExtracted fields: {full_state['fields']}")
            log.info(f"Content evaluation status: {full_state['evaluation_result'].get('status', 'Not evaluated')}")
            
            # 打印详细内容
            if full_state['content']:
                log.info("\nExtracted detailed content:")
                for item in full_state['content']:
                    log.info(f"- Field: {item.get('field', 'Unknown field')}, Value: {item.get('value', 'Not extracted')[:100]}...")
            
            t_end = time.time()
            log.info(f"[process_paper] 处理论文总耗时: {format_duration(t_end-t0)}")
            
            # 整理返回结果
            result = {
                "abstract_analysis": abstract_analysis,
                "fields": full_state["fields"],
                "content": full_state["content"],
                "evaluation_result": full_state["evaluation_result"],
                "optimization_history": full_state["optimization_history"],
                "optimization_attempts": full_state["optimization_attempts"],
                "doi": full_state["doi"]
            }
            
            # 将结果保存到extract.json文件
            save_to_json(result)
            
            # time.sleep(60)
            return result
        else:
            log.info("Does not meet processing criteria. Skipping this paper.")
            t_end = time.time()
            log.info(f"[process_paper] 处理论文总耗时: {format_duration(t_end-t0)}")
            
            # 整理返回结果，仅包含摘要分析和doi
            result = {
                "abstract_analysis": abstract_analysis,
                "doi": clean_doi(paper.get("doi", "Unknown"))
            }
            
            # 将结果保存到extract.json文件
            save_to_json(result)
            
            return result
    except Exception as e:
        log.error(f"Error processing paper: {e}")
        return None

if __name__ == "__main__":
    main() 