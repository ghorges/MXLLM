"""
主窗口界面
实现美观简洁的聊天界面
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter,
    QScrollArea, QFrame, QMessageBox, QProgressBar, QDialog,
    QDialogButtonBox, QFormLayout, QTextBrowser, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from PyQt5 import QtCore

# 添加父目录到路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.rag_system import RAGSystem
from core.llm_handler import LLMHandler
from core.config_cache import config_cache

logger = logging.getLogger(__name__)


class InitializationWorker(QThread):
    """初始化工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)
    
    def __init__(self, rag_system):
        super().__init__()
        self.rag_system = rag_system
    
    def run(self):
        try:
            logger.info("初始化线程开始...")
            self.progress.emit("正在初始化向量数据库...")
            
            # 如果在启动前就被请求中断，直接返回
            if self.isInterruptionRequested():
                self.finished.emit(False)
                return

            logger.info("调用rag_system.initialize_database()...")
            success = self.rag_system.initialize_database()
            # 初始化完成后再次检查是否被请求忽略
            if self.isInterruptionRequested():
                self.finished.emit(False)
                return
            
            logger.info(f"数据库初始化结果: {success}")
            self.finished.emit(success)
            
        except Exception as e:
            error_msg = f"初始化线程异常: {str(e)}"
            logger.error(error_msg)
            logger.exception("详细异常信息:")
            self.error.emit(error_msg)
            self.finished.emit(False)


class ChatWorker(QThread):
    """聊天处理工作线程"""
    response_ready = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)
    translation_ready = pyqtSignal(str)  # 翻译完成信号
    rag_search_ready = pyqtSignal(int)   # RAG搜索完成信号
    doi_search_ready = pyqtSignal(list, int)  # DOI检索完成信号 (dois, literature_count)
    recommendations_ready = pyqtSignal(dict)  # LLM推荐分析完成信号
    
    def __init__(self, llm_handler, rag_system, message, conversation_history):
        super().__init__()
        self.llm_handler = llm_handler
        self.rag_system = rag_system
        self.message = message
        self.conversation_history = conversation_history
    
    def run(self):
        try:
            # 检查RAG系统是否可用
            if self.rag_system is None:
                # 快速模式：仅使用LLM聊天，不使用推荐
                logger.info("快速模式：使用纯LLM聊天")
                response = self.llm_handler.chat_with_recommendations(
                    self.message, None, self.conversation_history
                )
                
                # 返回空的推荐结果
                empty_recommendations = {
                    'chemical_formulas': [],
                    'synthesis_methods': [],
                    'testing_procedures': [],
                    'ai_predictions': []
                }
                
                self.response_ready.emit(response, empty_recommendations)
                return
            
            # 新的分步骤推荐模式
            from core.config_cache import config_cache
            
            # 步骤1：翻译（如果需要）
            config = config_cache.load_config()
            english_query = self.message
            
            if config and self.llm_handler.is_api_ready():
                # 检测是否需要翻译（简单检测：包含中文字符）
                import re
                if re.search(r'[\u4e00-\u9fff]', self.message):
                    english_query = self.llm_handler.translate_to_professional_english(self.message)
                    # 发送翻译结果信号
                    self.translation_ready.emit(english_query)
            
            # 步骤2：RAG搜索
            search_results = self.rag_system.search_similar(english_query, n_results=10)
            # 发送RAG搜索结果信号
            self.rag_search_ready.emit(len(search_results))
            
            # 步骤3：DOI检索
            top_dois = self.rag_system._extract_top_dois(search_results, max_dois=2)
            full_literature_data = self.rag_system._get_literature_by_dois(top_dois)
            # 发送DOI检索结果信号
            self.doi_search_ready.emit(top_dois, len(full_literature_data))
            
            # 步骤4：LLM推荐分析
            recommendations = self.rag_system.get_direct_recommendations(self.message)
            
            # 立即发送推荐结果到界面显示
            self.recommendations_ready.emit(recommendations)
            
            # 将化学式数据也作为AI预测数据使用（保持UI兼容性）
            if 'content' not in recommendations:
                recommendations['ai_predictions'] = recommendations.get('chemical_formulas', [])
            
            # 步骤5：生成聊天回复
            response = self.llm_handler.chat_with_recommendations(
                self.message, recommendations, self.conversation_history
            )
            
            self.response_ready.emit(response, recommendations)
            
        except Exception as e:
            logger.error(f"聊天处理失败: {e}")
            self.error_occurred.emit(str(e))


class APIConfigDialog(QDialog):
    """API Configuration Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Service Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setup_ui()
        self.load_cached_values()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 说明标签
        info_label = QLabel(
            "Configure AI service parameters to start using chat functionality:\n\n"
            "• Supports OpenAI, Azure, or other compatible API services\n"
            "• Configuration will be automatically saved locally for next use\n"
            "• Click 'Test Connection' to verify if configuration is correct"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 配置表单
        form_layout = QFormLayout()
        
        # API密钥
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("sk-... or your API key")
        form_layout.addRow("API Key:", self.api_key_input)
        
        # API地址
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("https://api.openai.com/v1")
        self.api_base_input.setText("https://api.openai.com/v1")
        form_layout.addRow("API Address:", self.api_base_input)
        
        # 模型名称
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("gpt-3.5-turbo")
        self.model_input.setText("gpt-3.5-turbo")
        form_layout.addRow("Model Name:", self.model_input)
        
        layout.addLayout(form_layout)
        
        # 测试按钮
        test_layout = QHBoxLayout()
        self.test_button = QPushButton("🔍 Test Connection")
        self.test_button.clicked.connect(self.test_connection)
        test_layout.addWidget(self.test_button)
        
        self.test_result_label = QLabel("")
        test_layout.addWidget(self.test_result_label)
        test_layout.addStretch()
        
        layout.addLayout(test_layout)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 清除配置按钮
        self.clear_button = QPushButton("🗑️ Clear Cache")
        self.clear_button.clicked.connect(self.clear_config)
        self.clear_button.setStyleSheet("background-color: #FF5722; color: white;")
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        # 确定取消按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        # 设置按钮文本为英文
        buttons.button(QDialogButtonBox.Ok).setText("OK")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_cached_values(self):
        """Load cached configuration values as defaults"""
        try:
            from core.config_cache import config_cache
            cached_config = config_cache.load_config()
            if cached_config:
                self.api_key_input.setText(cached_config.get('api_key', ''))
                self.api_base_input.setText(cached_config.get('api_base', 'https://api.openai.com/v1'))
                self.model_input.setText(cached_config.get('model', 'gpt-3.5-turbo'))
                logger.info("Loaded cached configuration to dialog")
        except Exception as e:
            logger.error(f"Failed to load cached configuration to dialog: {e}")
    
    def test_connection(self):
        """Test API connection"""
        api_key = self.api_key_input.text().strip()
        api_base = self.api_base_input.text().strip()
        model = self.model_input.text().strip()
        
        if not api_key:
            self.test_result_label.setText("❌ Please enter API key")
            self.test_result_label.setStyleSheet("color: red;")
            return
        
        if not api_base:
            self.test_result_label.setText("❌ Please enter API address")
            self.test_result_label.setStyleSheet("color: red;")
            return
            
        if not model:
            self.test_result_label.setText("❌ Please enter model name")
            self.test_result_label.setStyleSheet("color: red;")
            return
        
        # 开始测试
        self.test_button.setEnabled(False)
        self.test_button.setText("🔄 Testing...")
        self.test_result_label.setText("Testing connection...")
        self.test_result_label.setStyleSheet("color: blue;")
        
        # 创建测试线程
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class TestWorker(QThread):
            result = pyqtSignal(bool, str)
            
            def __init__(self, api_key, api_base, model):
                super().__init__()
                self.api_key = api_key
                self.api_base = api_base
                self.model = model
            
            def run(self):
                try:
                    from openai import OpenAI
                    
                    # 创建客户端
                    if self.api_base.endswith('/v1'):
                        base_url = self.api_base
                    else:
                        base_url = self.api_base.rstrip('/') + '/v1'
                    
                    client = OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                    
                    # 发送测试消息
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=50,
                        timeout=10
                    )
                    
                    # 检查回复
                    if response.choices and response.choices[0].message.content:
                        self.result.emit(True, "Connection successful!")
                    else:
                        self.result.emit(False, "API response format exception")
                        
                except Exception as e:
                    self.result.emit(False, f"Connection failed: {str(e)}")
        
        self.test_worker = TestWorker(api_key, api_base, model)
        self.test_worker.result.connect(self.on_test_result)
        self.test_worker.start()
    
    def on_test_result(self, success: bool, message: str):
        """处理测试结果"""
        self.test_button.setEnabled(True)
        self.test_button.setText("�� Test Connection")
        
        if success:
            self.test_result_label.setText(f"✅ {message}")
            self.test_result_label.setStyleSheet("color: green;")
        else:
            self.test_result_label.setText(f"❌ {message}")
            self.test_result_label.setStyleSheet("color: red;")
    
    def clear_config(self):
        """Clear cached configuration"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            from core.config_cache import config_cache
            
            reply = QMessageBox.question(self, "Clear Cache", 
                "Are you sure you want to clear all cached configurations?\n\n"
                "This will delete saved API key, address and model configurations.",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if config_cache.clear_config():
                    # 清空输入框
                    self.api_key_input.clear()
                    self.api_base_input.setText("https://api.openai.com/v1")
                    self.model_input.setText("gpt-3.5-turbo")
                    self.test_result_label.setText("✅ Cache cleared")
                    self.test_result_label.setStyleSheet("color: green;")
                else:
                    self.test_result_label.setText("❌ Failed to clear cache")
                    self.test_result_label.setStyleSheet("color: red;")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_config(self):
        """获取配置"""
        return {
            'api_key': self.api_key_input.text().strip(),
            'api_base': self.api_base_input.text().strip(),
            'model': self.model_input.text().strip()
        }


class ChatMessage(QFrame):
    """聊天消息组件"""
    
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(text)
    
    def setup_ui(self, text: str):
        self.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 发送者标签
        sender_label = QLabel("You" if self.is_user else "AI Experimental Results Summary")
        sender_label.setFont(QFont("微软雅黑", 10, QFont.Bold))
        
        if self.is_user:
            sender_label.setStyleSheet("color: #007ACC;")
            self.setStyleSheet("""
                ChatMessage {
                    background-color: #E3F2FD;
                    border: 1px solid #BBDEFB;
                    border-radius: 10px;
                    margin: 2px;
                    padding: 5px;
                }
            """)
        else:
            sender_label.setStyleSheet("color: #2E7D32;")
            self.setStyleSheet("""
                ChatMessage {
                    background-color: #F1F8E9;
                    border: 1px solid #C8E6C9;
                    border-radius: 10px;
                    margin: 2px;
                    padding: 5px;
                }
            """)
        
        layout.addWidget(sender_label)
        
        # 消息内容
        message_text = QTextBrowser()
        message_text.setPlainText(text)
        # 移除最大高度限制，让内容完全展示
        message_text.setMaximumHeight(400)
        message_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        message_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        message_text.setFrameStyle(QFrame.NoFrame)
        message_text.setStyleSheet("background-color: transparent; border: none;")
        
        layout.addWidget(message_text)
        self.setLayout(layout)


class RecommendationPanel(QWidget):
    """推荐结果面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("📋 Recommendation Results")
        title_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        title_label.setStyleSheet("color: #1976D2; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 化学式标签页
        self.formulas_tab = QTextBrowser()
        self.tab_widget.addTab(self.formulas_tab, "🧪Formulas")
        
        # 合成工艺标签页
        self.synthesis_tab = QTextBrowser()
        self.tab_widget.addTab(self.synthesis_tab, "⚗️Synthesis")
        
        # 测试流程标签页
        self.testing_tab = QTextBrowser()
        self.tab_widget.addTab(self.testing_tab, "🔬Testing")
        
        # AI预测标签页
        self.prediction_tab = QTextBrowser()
        self.tab_widget.addTab(self.prediction_tab, "🤖AI Prediction")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
        # 设置样式
        self.setStyleSheet("""
            RecommendationPanel {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #F5F5F5;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-bottom: none;
            }
        """)
    
    def update_recommendations(self, recommendations: Dict[str, Any]):
        """更新推荐结果"""
        try:
            # 检查新格式的数据结构
            if 'content' in recommendations and isinstance(recommendations['content'], list):
                # 新的JSON格式
                content = recommendations['content']
                if content:
                    # 格式化各个标签页内容
                    formulas_html = self._format_new_materials(content)
                    self.formulas_tab.setHtml(formulas_html)
                    
                    synthesis_html = self._format_new_synthesis(content)
                    self.synthesis_tab.setHtml(synthesis_html)
                    
                    testing_html = self._format_new_testing(content)
                    self.testing_tab.setHtml(testing_html)
                    
                    prediction_html = self._format_new_performance(content, recommendations)
                    self.prediction_tab.setHtml(prediction_html)
                else:
                    # 空内容处理
                    empty_html = "<p>No recommendation content available</p>"
                    self.formulas_tab.setHtml(empty_html)
                    self.synthesis_tab.setHtml(empty_html)
                    self.testing_tab.setHtml(empty_html)
                    self.prediction_tab.setHtml(empty_html)
            else:
                # 旧格式的兼容处理
                formulas_html = self._format_formulas(recommendations.get('chemical_formulas', []))
                self.formulas_tab.setHtml(formulas_html)
                
                synthesis_html = self._format_synthesis(recommendations.get('synthesis_methods', []))
                self.synthesis_tab.setHtml(synthesis_html)
                
                testing_html = self._format_testing(recommendations.get('testing_procedures', []))
                self.testing_tab.setHtml(testing_html)
                
                prediction_html = self._format_predictions(recommendations.get('ai_predictions', []))
                self.prediction_tab.setHtml(prediction_html)
            
        except Exception as e:
            logger.error(f"更新推荐结果失败: {e}")
    
    def _format_formulas(self, formulas: List[Dict[str, Any]]) -> str:
        """格式化化学式推荐（简化版，不显示预测信息）"""
        if not formulas:
            return "<p>No chemical formula recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        for i, formula in enumerate(formulas, 1):
            # 确定验证状态
            has_experimental_data = formula.get('has_experimental_data', False)
            validation = formula.get('validation', {})
            
            # 选择显示样式
            if has_experimental_data:
                border_color = "#4CAF50"
                data_type_icon = "🧪"
                data_type_text = "Has experimental data"
            else:
                border_color = "#007ACC"
                data_type_icon = "📋"
                data_type_text = "Recommended material"
            
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #F8F9FA; border-left: 4px solid {border_color}; border-radius: 5px;'>
                <h4 style='margin: 0 0 8px 0; color: {border_color};'>{data_type_icon} {i}. {formula.get('formula', 'Unknown')}</h4>
                <p style='margin: 0 0 5px 0;'><strong>Source:</strong> {formula.get('source', 'Unknown')}</p>
                                 <p style='margin: 0 0 5px 0;'><strong>Relevance:</strong> {formula.get('score', 0):.1%}</p>
            """
            
            # 如果有实验数据，显示DOI
            if validation.get('found_in_db') and validation.get('experimental_data'):
                exp_data = validation['experimental_data'][0]
                if exp_data.get('doi'):
                    html += f"<p style='margin: 0 0 5px 0;'><strong>DOI:</strong> <a href='https://doi.org/{exp_data['doi']}' target='_blank' style='color: #1976D2;'>{exp_data['doi']}</a></p>"
            
            html += f"<p style='margin: 0;'><strong>Description:</strong> {formula.get('description', 'No description available')}</p>"
            html += "</div>"
            
        html += "</div>"
        return html
    
    def _format_synthesis(self, methods: List[Dict[str, Any]]) -> str:
        """格式化合成工艺推荐"""
        if not methods:
            return "<p>No synthesis method recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        for i, method in enumerate(methods, 1):
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #F8F9FA; border-left: 4px solid #4CAF50; border-radius: 5px;'>
                <h4 style='margin: 0 0 8px 0; color: #4CAF50;'>{i}. {method.get('method', 'Unknown')}</h4>
                <p style='margin: 0 0 5px 0;'><strong>Source:</strong> {method.get('source', 'Unknown')}</p>
                                 <p style='margin: 0 0 5px 0;'><strong>Relevance:</strong> {method.get('score', 0):.1%}</p>
                <p style='margin: 0;'><strong>Description:</strong> {method.get('description', 'No description available')}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def _format_testing(self, procedures: List[Dict[str, Any]]) -> str:
        """格式化测试流程推荐"""
        if not procedures:
            return "<p>No testing procedure recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        for i, procedure in enumerate(procedures, 1):
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #F8F9FA; border-left: 4px solid #FF9800; border-radius: 5px;'>
                <h4 style='margin: 0 0 8px 0; color: #FF9800;'>{i}. {procedure.get('procedure', 'Unknown')}</h4>
                <p style='margin: 0 0 5px 0;'><strong>Source:</strong> {procedure.get('source', 'Unknown')}</p>
                                 <p style='margin: 0 0 5px 0;'><strong>Relevance:</strong> {procedure.get('score', 0):.1%}</p>
                <p style='margin: 0;'><strong>Description:</strong> {procedure.get('description', 'No description available')}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def _format_predictions(self, formulas: List[Dict[str, Any]]) -> str:
        """格式化AI预测结果"""
        if not formulas:
            return "<p>No prediction data available</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        
        # 统计信息
        total_formulas = len(formulas)
        experimental_count = sum(1 for f in formulas if f.get('has_experimental_data', False))
        prediction_count = sum(1 for f in formulas if f.get('validation', {}).get('prediction'))
        
        html += f"""
        <div style='margin-bottom: 20px; padding: 15px; background-color: #E3F2FD; border-left: 4px solid #2196F3;'>
            <h4 style='margin: 0 0 10px 0; color: #1976D2;'>🤖 AI Prediction Overview</h4>
            <p style='margin: 0 0 5px 0;'><strong>Total Recommended Materials:</strong> {total_formulas}</p>
            <p style='margin: 0 0 5px 0;'><strong>With Experimental Data:</strong> {experimental_count} ({experimental_count/total_formulas*100:.1f}%)</p>
            <p style='margin: 0;'><strong>AI Prediction Supplement:</strong> {prediction_count} ({prediction_count/total_formulas*100:.1f}%)</p>
        </div>
        """
        
        for i, formula in enumerate(formulas, 1):
            validation = formula.get('validation', {})
            formula_name = formula.get('formula', 'Unknown')
            
            if validation.get('found_in_db') and validation.get('experimental_data'):
                # 有实验数据的情况
                exp_data = validation['experimental_data'][0]
                
                html += f"""
                <div style='margin-bottom: 15px; padding: 10px; background-color: #E8F5E8; border-left: 4px solid #4CAF50; border-radius: 5px;'>
                    <h4 style='margin: 0 0 8px 0; color: #4CAF50;'>🧪 {i}. {formula_name}</h4>
                    <p style='margin: 0 0 8px 0; color: #2E7D32; font-weight: bold;'>✅ Experimental data available in database, no AI prediction needed</p>
                """
                
                if exp_data.get('properties'):
                    html += f"<p style='margin: 0 0 5px 0;'><strong>Experimental Performance:</strong> "
                    props = exp_data['properties']
                    import html as html_module
                    prop_strs = [f"{k}: {html_module.escape(str(v))}" for k, v in props.items()]
                    html += ", ".join(prop_strs) + "</p>"
                
                if exp_data.get('doi'):
                    html += f"<p style='margin: 0 0 5px 0;'><strong>Data Source:</strong> <a href='https://doi.org/{exp_data['doi']}' target='_blank' style='color: #1976D2;'>{exp_data['doi']}</a></p>"
                
                html += "</div>"
                
            elif validation.get('prediction'):
                # 有AI预测的情况
                pred = validation['prediction']
                
                html += f"""
                <div style='margin-bottom: 15px; padding: 10px; background-color: #FFF3E0; border-left: 4px solid #FF9800; border-radius: 5px;'>
                    <h4 style='margin: 0 0 8px 0; color: #FF9800;'>🔮 {i}. {formula_name}</h4>
                    <p style='margin: 0 0 10px 0; color: #E65100; font-weight: bold;'>🤖 No data in database, using AI prediction</p>
                    
                    <div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <h5 style='margin: 0 0 8px 0; color: #FF9800;'>Prediction Results:</h5>
                        <ul style='margin: 0; padding-left: 20px;'>
                            <li><strong>EAB (Effective Absorption Bandwidth):</strong> {html_module.escape(str(pred['eab_prediction']))} - {html_module.escape(str(pred['eab_meaning']))}</li>
                            <li><strong>RL (Reflection Loss):</strong> {html_module.escape(str(pred['rl_prediction']))} - {html_module.escape(str(pred['rl_meaning']))}</li>
                        </ul>
                    </div>
                    
                    <div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px;'>
                        <h5 style='margin: 0 0 8px 0; color: #FF9800;'>Confidence Analysis:</h5>
                        <p style='margin: 0 0 5px 0;'><strong>Overall Confidence:</strong> {pred['confidence']:.3f}</p>
                        <p style='margin: 0 0 5px 0;'><strong>EAB Confidence:</strong> {pred['eab_confidence']:.3f}</p>
                        <p style='margin: 0;'><strong>RL Confidence:</strong> {pred['rl_confidence']:.3f}</p>
                    </div>
                    
                    <p style='margin: 10px 0 0 0; color: #E65100; font-size: 12px; font-style: italic;'>
                        ⚠️ This is an AI prediction result, recommend combining with experimental validation
                    </p>
                </div>
                """
                
            else:
                # 无数据且无预测的情况
                html += f"""
                <div style='margin-bottom: 15px; padding: 10px; background-color: #F5F5F5; border-left: 4px solid #9E9E9E; border-radius: 5px;'>
                    <h4 style='margin: 0 0 8px 0; color: #757575;'>📋 {i}. {formula_name}</h4>
                    <p style='margin: 0; color: #757575;'>❌ No experimental data available, AI prediction feature unavailable</p>
                </div>
                """
        
        html += "</div>"
        return html

    def _format_value_for_ui(self, value):
        """UI专用的通用值格式化函数，处理各种数据类型"""
        import html as html_module
        
        if isinstance(value, dict):
            # 字典类型：尝试提取关键信息
            if 'name' in value:
                # 如果有name字段，优先使用name
                result = str(value['name'])
                if 'amount' in value:
                    result += f" ({value['amount']})"
                elif 'volume' in value:
                    result += f" ({value['volume']})"
                return html_module.escape(result)
            else:
                # 否则转换为key: value格式
                pairs = [f"{k}: {v}" for k, v in value.items()]
                return html_module.escape("; ".join(pairs))
        elif isinstance(value, list):
            # 列表类型：递归处理每个元素
            return "、".join([self._format_value_for_ui(item) for item in value])
        else:
            # 其他类型直接转字符串并转义
            return html_module.escape(str(value))

    def _format_new_materials(self, content: List[Dict[str, Any]]) -> str:
        """格式化新格式的材料信息"""
        if not content:
            return "<p>No material recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        
        for i, item in enumerate(content, 1):
            material = item.get('material', {})
            performance = item.get('performance', {})
            confidence = item.get('confidence', 'medium')
            
            # 基本信息
            formula = material.get('chemical_formula', 'Unknown')
            name = material.get('name', '')
            composition_type = material.get('composition_type', '')
            structure_type = material.get('structure_type', '')
            morphology = material.get('morphology', '')
            heterostructure_type = material.get('heterostructure_type', '')
            
            # 置信度颜色
            confidence_colors = {
                'high': '#4CAF50',
                'medium': '#FF9800',
                'low': '#F44336'
            }
            border_color = confidence_colors.get(confidence, '#007ACC')
            confidence_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}.get(confidence, confidence)
            
            html += f"""
            <div style='margin-bottom: 15px; padding: 15px; background-color: #F8F9FA; border-left: 4px solid {border_color}; border-radius: 8px;'>
                <h4 style='margin: 0 0 10px 0; color: {border_color}; display: flex; justify-content: space-between; align-items: center;'>
                    <span>🧪 {i}. {formula}</span>
                    <span style='font-size: 12px; background-color: {border_color}; color: white; padding: 2px 8px; border-radius: 12px;'>Confidence: {confidence_text}</span>
                </h4>
                
                <div style='background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <h5 style='margin: 0 0 8px 0; color: #333;'>Material Characteristics</h5>
            """
            
            if name:
                html += f"<p style='margin: 2px 0;'><strong>Name:</strong> {name}</p>"
            if composition_type:
                html += f"<p style='margin: 2px 0;'><strong>Composition Type:</strong> {composition_type}</p>"
            if structure_type:
                html += f"<p style='margin: 2px 0;'><strong>Structure Type:</strong> {structure_type}</p>"
            if morphology:
                html += f"<p style='margin: 2px 0;'><strong>Morphology:</strong> {morphology}</p>"
            if heterostructure_type:
                html += f"<p style='margin: 2px 0;'><strong>Heterostructure Type:</strong> {heterostructure_type}</p>"
                
            html += "</div>"
            
            # 分子式标签页不显示性能数据，性能数据在AI预测标签页中显示
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _format_new_synthesis(self, content: List[Dict[str, Any]]) -> str:
        """格式化新格式的合成工艺"""
        if not content:
            return "<p>No synthesis method recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        
        for item in content:
            material = item.get('material', {})
            synthesis_steps = item.get('synthesis_steps', [])
            formula = material.get('chemical_formula', 'Unknown')
            
            if not synthesis_steps:
                continue
                
            html += f"""
            <div style='margin-bottom: 20px; padding: 15px; background-color: #F0F8F0; border-left: 4px solid #4CAF50; border-radius: 8px;'>
                <h4 style='margin: 0 0 15px 0; color: #4CAF50;'>⚗️ {formula} - Synthesis Method</h4>
            """
            
            for step in synthesis_steps:
                step_num = step.get('step', 0)
                step_name = step.get('step_name', 'Unknown step')
                role = step.get('role', '')
                method = step.get('method', '')
                inputs = step.get('inputs', [])
                conditions = step.get('conditions', [])
                outputs = step.get('outputs', '')
                notes = step.get('notes', '')
                inferred = step.get('inferred', False)
                justification = step.get('justification', '')
                
                html += f"""
                <div style='margin-bottom: 10px; padding: 10px; background-color: #FFFFFF; border-radius: 5px; border: 1px solid #E0E0E0;'>
                    <div style='font-weight: bold; color: #4CAF50; margin-bottom: 5px;'>
                        Step {step_num}: {step_name}
                """
                
                if role:
                    html += f" <span style='background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-left: 8px;'>{role}</span>"
                
                if inferred:
                    html += "<span style='margin-left: 8px; background-color: #FF9800; color: white; padding: 1px 6px; border-radius: 8px; font-size: 10px;'>Inferred</span>"
                
                html += "</div>"
                
                if method:
                    html += f"<p style='margin: 4px 0;'><strong>Method:</strong> {method}</p>"
                
                if inputs:
                    inputs_text = self._format_value_for_ui(inputs)
                    html += f"<p style='margin: 4px 0;'><strong>Ingredients:</strong> {inputs_text}</p>"
                
                if conditions:
                    html += f"<p style='margin: 4px 0;'><strong>Conditions:</strong></p><ul style='margin: 2px 0; padding-left: 20px;'>"
                    for condition in conditions:
                        condition_text = self._format_value_for_ui(condition)
                        html += f"<li style='margin: 1px 0;'>{condition_text}</li>"
                    html += "</ul>"
                
                if outputs:
                    outputs_text = self._format_value_for_ui(outputs)
                    html += f"<p style='margin: 4px 0;'><strong>Products:</strong> {outputs_text}</p>"
                
                if notes:
                    notes_text = self._format_value_for_ui(notes)
                    html += f"<p style='margin: 4px 0;'><strong>Notes:</strong> {notes_text}</p>"
                
                if inferred and justification:
                    justification_text = self._format_value_for_ui(justification)
                    html += f"<p style='margin: 4px 0; color: #FF9800; font-style: italic;'><strong>Justification:</strong> {justification_text}</p>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _format_new_testing(self, content: List[Dict[str, Any]]) -> str:
        """格式化新格式的测试流程"""
        if not content:
            return "<p>No testing procedure recommendations</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        
        for i, item in enumerate(content, 1):
            material = item.get('material', {})
            testing_steps = item.get('testing_steps', [])
            formula = material.get('chemical_formula', 'Unknown')
            
            if not testing_steps:
                continue
                
            html += f"""
            <div style='margin-bottom: 20px; padding: 15px; background-color: #F3E5F5; border-left: 4px solid #9C27B0; border-radius: 8px;'>
                <h4 style='margin: 0 0 15px 0; color: #9C27B0;'>🔬 {formula} - Testing Procedure</h4>
            """
            
            for step in testing_steps:
                step_num = step.get('step', 0)
                step_name = step.get('step_name', 'Unknown step')
                method = step.get('method', '')
                parameters = step.get('parameters', [])
                notes = step.get('notes', '')
                inferred = step.get('inferred', False)
                justification = step.get('justification', '')
                
                html += f"""
                <div style='margin-bottom: 10px; padding: 10px; background-color: #FFFFFF; border-radius: 5px; border: 1px solid #E0E0E0;'>
                    <div style='font-weight: bold; color: #9C27B0; margin-bottom: 5px;'>
                        Step {step_num}: {step_name}
                """
                
                if inferred:
                    html += "<span style='margin-left: 8px; background-color: #FF9800; color: white; padding: 1px 6px; border-radius: 8px; font-size: 10px;'>Inferred</span>"
                
                html += "</div>"
                
                if method:
                    method_text = self._format_value_for_ui(method)
                    html += f"<p style='margin: 4px 0;'><strong>Method:</strong> {method_text}</p>"
                
                if parameters:
                    html += f"<p style='margin: 4px 0;'><strong>Parameters:</strong></p><ul style='margin: 2px 0; padding-left: 20px;'>"
                    for param in parameters:
                        param_text = self._format_value_for_ui(param)
                        html += f"<li style='margin: 1px 0;'>{param_text}</li>"
                    html += "</ul>"
                
                if notes:
                    notes_text = self._format_value_for_ui(notes)
                    html += f"<p style='margin: 4px 0;'><strong>Notes:</strong> {notes_text}</p>"
                
                if inferred and justification:
                    justification_text = self._format_value_for_ui(justification)
                    html += f"<p style='margin: 4px 0; color: #FF9800; font-style: italic;'><strong>Justification:</strong> {justification_text}</p>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _format_new_performance(self, content: List[Dict[str, Any]], recommendations: Dict[str, Any]) -> str:
        """Format new format performance data"""
        if not content:
            return "<p>No performance data available</p>"
        
        html = "<div style='font-family: 微软雅黑; line-height: 1.6;'>"
        
        # 全局信息
        source = recommendations.get('source', 'Unknown Source')
        doi = recommendations.get('doi', '')
        query = recommendations.get('query', '')
        
        html += f"""
        <div style='margin-bottom: 20px; padding: 15px; background-color: #E3F2FD; border-left: 4px solid #2196F3; border-radius: 8px;'>
            <h4 style='margin: 0 0 10px 0; color: #1976D2;'>📊 Data Source Information</h4>
            <p style='margin: 2px 0;'><strong>Query:</strong> {query}</p>
            <p style='margin: 2px 0;'><strong>Source:</strong> {source}</p>
        """
        
        if doi:
            html += f"<p style='margin: 2px 0;'><strong>DOI:</strong> <a href='https://doi.org/{doi}' target='_blank' style='color: #1976D2;'>{doi}</a></p>"
        
        html += "</div>"
        
        for i, item in enumerate(content, 1):
            material = item.get('material', {})
            performance = item.get('performance', {})
            confidence = item.get('confidence', 'medium')
            
            formula = material.get('chemical_formula', 'Unknown')
            
            # 置信度颜色
            confidence_colors = {
                'high': '#4CAF50',
                'medium': '#FF9800', 
                'low': '#F44336'
            }
            border_color = confidence_colors.get(confidence, '#FF9800')
            confidence_text = {'high': 'High Confidence', 'medium': 'Medium Confidence', 'low': 'Low Confidence'}.get(confidence, confidence)
            
            html += f"""
            <div style='margin-bottom: 15px; padding: 15px; background-color: #FFF8E1; border-left: 4px solid {border_color}; border-radius: 8px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                    <h4 style='margin: 0; color: {border_color};'>🎯 {formula}</h4>
                    <span style='background-color: {border_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;'>{confidence_text}</span>
                </div>
            """
            
            # 性能指标网格
            if performance:
                html += """
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 10px;'>
                """
                
                for key, value in performance.items():
                    if value and value != "":
                        display_name = {
                            'rl_min': 'RL Minimum',
                            'matching_thickness': 'Matching Thickness',
                            'eab': 'Effective Absorption Bandwidth',
                            'other': 'Other Performance'
                        }.get(key, key.upper())
                        
                        html += f"""
                        <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #E0E0E0;'>
                            <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>{display_name}</div>
                            <div style='font-size: 16px; font-weight: bold; color: {border_color};'>{value}</div>
                        </div>
                        """
                
                html += "</div>"
            else:
                html += "<p style='color: #666; text-align: center; padding: 20px;'>No specific performance data</p>"
            
            # 材料特征信息
            material_info = []
            if material.get('composition_type'):
                material_info.append(f"Composition: {material['composition_type']}")
            if material.get('structure_type'):
                material_info.append(f"Structure: {material['structure_type']}")
            if material.get('morphology'):
                material_info.append(f"Morphology: {material['morphology']}")
            
            if material_info:
                html += f"""
                <div style='background-color: rgba(255,255,255,0.7); padding: 8px; border-radius: 4px; font-size: 13px; color: #666;'>
                    <strong>Material Characteristics:</strong> {' | '.join(material_info)}
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXLLM Material Recommendation System")
        self.setMinimumSize(1400, 800)
        
        # 初始化组件
        self.rag_system = None
        self.llm_handler = LLMHandler()
        self.conversation_history = []
        self.current_recommendations = {}
        
        # 设置样式
        self.setup_style()
        
        # 创建界面
        self.setup_ui()
        
        # 显示欢迎消息并自动初始化
        self.add_system_message("🎉 MXLLM Material Recommendation System started! Automatically initializing RAG system...")
        
        # 延迟自动初始化RAG系统
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self.initialize_system)  # 0.5秒后自动初始化
        
        # 尝试加载缓存的配置
        QTimer.singleShot(1000, self.load_cached_config)  # 1秒后加载缓存配置

        self._ignore_init_result = False
    

    def load_cached_config(self):
        """加载缓存的配置"""
        try:
            cached_config = config_cache.load_config()
            if cached_config:
                logger.info("发现缓存配置，自动加载...")
                
                # 自动配置LLM处理器
                if self.llm_handler.set_api_config(cached_config):
                    self.send_button.setEnabled(True)
                    self.api_button.setText("🔑 AI Service Configured")
                    self.api_button.setStyleSheet("background-color: #4CAF50;")
                    self.statusBar().showMessage(f"Cached config loaded - Model: {cached_config['model']}")
                    self.add_system_message(f"✅ Auto-loaded cached configuration successfully!\n• Model: {cached_config['model']}\n• You can start chatting now.")
                else:
                    self.add_system_message("⚠️ Failed to load cached configuration, please reconfigure AI service.")
            else:
                logger.info("未找到有效的缓存配置")
                
        except Exception as e:
            logger.error(f"加载缓存配置失败: {e}")
    
    def setup_style(self):
        """设置应用样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QLineEdit {
                border: 2px solid #E0E0E0;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #007ACC;
            }
            QLabel {
                font-family: 微软雅黑;
            }
        """)
    
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧聊天区域
        chat_widget = self.create_chat_widget()
        
        # 右侧推荐区域
        recommendation_widget = self.create_recommendation_widget()
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(chat_widget)
        splitter.addWidget(recommendation_widget)
        splitter.setSizes([700, 500])  # 设置初始比例
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage("Ready - Please configure API key first")
    
    def create_chat_widget(self) -> QWidget:
        """创建聊天区域"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 标题和API设置
        header_layout = QHBoxLayout()
        
        title_label = QLabel("💬 MXLLM Material AI Assistant")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color: #1976D2; margin-bottom: 10px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.init_button = QPushButton("🔄 Auto-initializing...")
        self.init_button.clicked.connect(self.initialize_system)
        self.init_button.setEnabled(False)  # 初始时禁用，因为会自动初始化
        header_layout.addWidget(self.init_button)
        
        self.api_button = QPushButton("🔑 Configure AI Service")
        self.api_button.clicked.connect(self.setup_api_config)
        header_layout.addWidget(self.api_button)
        
        layout.addLayout(header_layout)
        
        # 聊天记录区域
        self.chat_scroll = QScrollArea()
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.addStretch()  # 添加伸缩项使消息从底部开始
        self.chat_widget.setLayout(self.chat_layout)
        self.chat_scroll.setWidget(self.chat_widget)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(self.chat_scroll)
        
        # 输入区域
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Enter your question (e.g., I need high-frequency absorption MXLLM materials)...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # 进度条（初始隐藏）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        widget.setLayout(layout)
        return widget
    
    def create_recommendation_widget(self) -> QWidget:
        """创建推荐区域"""
        self.recommendation_panel = RecommendationPanel()
        return self.recommendation_panel
    
    def initialize_system(self):
        """初始化系统"""
        if self.rag_system is not None:
            self.add_system_message("ℹ️ RAG system already initialized.")
            return
            
        try:
            # 更新按钮状态
            self.init_button.setEnabled(False)
            self.init_button.setText("�� Initializing...")
            
            # 显示初始化消息
            self.add_system_message("🔄 Initializing RAG recommendation system, please wait...")
            self.statusBar().showMessage("Initializing RAG system...")
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 不确定进度
            
            logger.info("Creating RAGSystem instance...")
            self.rag_system = RAGSystem()
            logger.info("RAGSystem instance created successfully")
            
            # 创建初始化线程
            self.init_worker = InitializationWorker(self.rag_system)
            self.init_worker.progress.connect(self.update_init_progress)
            self.init_worker.finished.connect(self.on_init_finished)
            self.init_worker.error.connect(self.on_init_error)
            
            logger.info("Starting initialization thread...")
            self.init_worker.start()
            logger.info("Initialization thread started")
            
            # 添加快速模式提示
            # QTimer.singleShot(15000, self.show_quick_mode_option)  # 15秒后显示快速模式选项
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.add_system_message(f"❌ System initialization failed: {str(e)}")
            self.init_button.setEnabled(True)
            self.init_button.setText("🤖 Initialize RAG System")
            QMessageBox.critical(self, "Error", f"System initialization failed: {str(e)}")
    
    @QtCore.pyqtSlot(str)
    def update_init_progress(self, message: str):
        """更新初始化进度"""
        self.statusBar().showMessage(message)
    
    @QtCore.pyqtSlot(bool)
    def on_init_finished(self, success: bool):
        """初始化完成"""
        logger.info(f"Initialization callback finished, success: {success}")
        self.progress_bar.setVisible(False)

        # 如果用户已切到快速模式，忽略后台线程的回调结果，避免状态被覆盖
        if getattr(self, "_ignore_init_result", False):
            logger.info("Quick mode enabled: ignoring background initialization result")
            return
        
        if success:
            self.init_button.setText("✅ RAG System Ready")
            self.init_button.setStyleSheet("background-color: #4CAF50;")
            self.statusBar().showMessage("System initialized - Please configure API key to start chatting")
            self.add_system_message("🎉 RAG System initialized! Please configure AI service to start chatting.")
            logger.info("RAG System initialized successfully")
        else:
            self.init_button.setEnabled(True)
            self.init_button.setText("❌ Re-initialize")
            self.init_button.setStyleSheet("background-color: #F44336;")
            self.statusBar().showMessage("System initialization failed")
            self.add_system_message("⚠️ RAG System initialization failed, please click Re-initialize or check data files.")
            logger.error("RAG System initialization failed")
    
    @QtCore.pyqtSlot(str)
    def on_init_error(self, error_msg: str):
        """初始化错误"""
        logger.error(f"Initialization error callback: {error_msg}")
        self.progress_bar.setVisible(False)
        self.init_button.setEnabled(True)
        self.init_button.setText("❌ Re-initialize")
        self.init_button.setStyleSheet("background-color: #F44336;")
        self.add_system_message(f"❌ Initialization error: {error_msg}")
        QMessageBox.critical(self, "Initialization Error", f"An error occurred during system initialization:\n\n{error_msg}")
    
    def show_quick_mode_option(self):
        """显示快速模式选项"""
        if hasattr(self, 'init_worker') and self.init_worker.isRunning():
            reply = QMessageBox.question(
                self, 
                "Initialization takes time", 
                "Database initialization takes a long time, would you like to use quick mode?\n\n"
                "Quick mode will skip vector database initialization, but recommendation functionality may be limited.\n"
                "You can re-initialize the full database later in settings.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # 停止当前初始化
                self.init_worker.terminate()
                self.init_worker.wait()
                
                # 启动快速模式初始化
                self.start_quick_initialization()
    
    def start_quick_initialization(self):
        """启动快速模式初始化"""
        try:
            logger.info("User chose quick mode initialization...")
            
            # 使用快速模式重新初始化
            success = self.rag_system.initialize_database(quick_mode=True)
            
            if success:
                self.progress_bar.setVisible(False)
                self.init_button.setText("✅ Quick Mode Ready")
                self.init_button.setStyleSheet("background-color: #FF9800;")  # Orange for quick mode
                self.statusBar().showMessage("Quick mode initialization complete - Please configure API key to start chatting")
                self.add_system_message("🚀 Quick mode initialization complete! Recommendation functionality may be limited, can be re-initialized later.")
                logger.info("Quick mode initialization successful")
            else:
                self.on_init_error("Quick mode initialization failed")
                
        except Exception as e:
            logger.error(f"Quick mode initialization failed: {e}")
            self.on_init_error(f"Quick mode initialization failed: {str(e)}")
    
    def setup_api_config(self):
        """配置AI服务"""
        dialog = APIConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            
            if not config['api_key']:
                QMessageBox.warning(self, "Warning", "Please enter API key")
                return
            
            if not config['api_base']:
                QMessageBox.warning(self, "Warning", "Please enter API address")
                return
                
            if not config['model']:
                QMessageBox.warning(self, "Warning", "Please enter model name")
                return
            
            self.statusBar().showMessage("Configuring AI service...")
            
            # 配置LLM处理器
            if self.llm_handler.set_api_config(config):
                # 保存配置到缓存
                if config_cache.save_config(config):
                    logger.info("Configuration saved to cache")
                
                self.send_button.setEnabled(True)
                self.api_button.setText("🔑 AI Service Configured")
                self.api_button.setStyleSheet("background-color: #4CAF50;")
                self.statusBar().showMessage(f"AI service configured successfully - Using model: {config['model']}")
                self.add_system_message(f"✅ AI service configured successfully!\n• Model: {config['model']}\n• Configuration automatically saved, will be loaded on next startup\n• You can start chatting now.")
            else:
                QMessageBox.critical(self, "Error", "AI service configuration failed, please check if the configuration is correct.")
                self.statusBar().showMessage("AI service configuration failed")
    
    def add_system_message(self, message: str):
        """添加系统消息"""
        system_label = QLabel(f"🤖 {message}")
        system_label.setStyleSheet("""
            background-color: #FFF3E0;
            border: 1px solid #FFB74D;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            color: #E65100;
        """)
        system_label.setWordWrap(True)
        
        # 插入到伸缩项之前
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, system_label)
        self.scroll_to_bottom()
    
    def send_message(self):
        """发送消息"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        if not self.llm_handler.is_api_ready():
            QMessageBox.warning(self, "Tip", "Please configure AI service first.")
            return
        
        # 清空输入框并禁用发送
        self.message_input.clear()
        self.send_button.setEnabled(False)
        self.message_input.setEnabled(False)
        
        # 添加用户消息
        user_message = ChatMessage(message, True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_message)
        self.scroll_to_bottom()
        
        # 显示初始处理消息
        import re
        if re.search(r'[\u4e00-\u9fff]', message):
            initial_message = "🌐✨ Translating and polishing query..."
        else:
            initial_message = "✨ Polishing query..."
        
        self.thinking_message = QLabel(initial_message)
        self.thinking_message.setStyleSheet("""
            background-color: #F3E5F5;
            border: 1px solid #CE93D8;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            color: #7B1FA2;
        """)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.thinking_message)
        self.scroll_to_bottom()
        
        # 添加到对话历史
        self.conversation_history.append({"role": "user", "content": message})
        
        # 启动聊天处理线程
        self.chat_worker = ChatWorker(
            self.llm_handler, self.rag_system, message, self.conversation_history
        )
        self.chat_worker.response_ready.connect(self.on_response_ready)
        self.chat_worker.error_occurred.connect(self.on_chat_error)
        self.chat_worker.translation_ready.connect(self.on_translation_ready)
        self.chat_worker.rag_search_ready.connect(self.on_rag_search_ready)
        self.chat_worker.doi_search_ready.connect(self.on_doi_search_ready)
        self.chat_worker.recommendations_ready.connect(self.on_recommendations_ready)
        self.chat_worker.start()
    
    @QtCore.pyqtSlot(str, dict)
    def on_response_ready(self, response: str, recommendations: Dict[str, Any]):
        """处理AI回复"""
        # 移除"正在思考"消息
        self.thinking_message.setParent(None)
        
        # 如果有推荐数据，先添加提示信息
        if recommendations and (
            (recommendations.get('content') and len(recommendations['content']) > 0) or 
            any(recommendations.get(key, []) for key in ['chemical_formulas', 'synthesis_methods', 'testing_procedures'])
        ):
            # 添加LLM推荐结果提示
            llm_info = QLabel("🤖 LLM recommendation results are as follows, see the right panel.→→→")
            llm_info.setStyleSheet("""
                background-color: #E8F5E8;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                padding: 8px;
                margin: 5px;
                color: #2E7D32;
                font-weight: bold;
            """)
            llm_info.setWordWrap(True)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, llm_info)
        
        # 添加AI回复
        ai_message = ChatMessage(response, False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, ai_message)
        self.scroll_to_bottom()
        
        # 英文查询已在翻译步骤中显示，这里不再重复显示
        
        # 更新推荐面板
        if recommendations:
            self.current_recommendations = recommendations
            self.recommendation_panel.update_recommendations(recommendations)
        
        # 添加到对话历史
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # 重新启用输入
        self.send_button.setEnabled(True)
        self.message_input.setEnabled(True)
        self.message_input.setFocus()
    
    @QtCore.pyqtSlot(str)
    def on_chat_error(self, error_message: str):
        """处理聊天错误"""
        # 移除"正在思考"消息
        self.thinking_message.setParent(None)
        
        # 显示错误消息
        error_label = QLabel(f"❌ Error: {error_message}")
        error_label.setStyleSheet("""
            background-color: #FFEBEE;
            border: 1px solid #EF5350;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            color: #C62828;
        """)
        error_label.setWordWrap(True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, error_label)
        self.scroll_to_bottom()
        
        # 重新启用输入
        self.send_button.setEnabled(True)
        self.message_input.setEnabled(True)
        self.message_input.setFocus()
    
    @QtCore.pyqtSlot(str)
    def on_translation_ready(self, english_query: str):
        """处理翻译完成"""
        # 更新"正在思考"消息
        self.thinking_message.setText("🌐✨ Translation and polishing complete...")
        
        # 添加翻译并润色结果显示
        translation_info = QLabel(f"🌐✨ AI translation and polishing result: {english_query}")
        translation_info.setStyleSheet("""
            background-color: #E8F5E8;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 8px;
            margin: 5px;
            color: #2E7D32;
            font-style: italic;
        """)
        translation_info.setWordWrap(True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, translation_info)
        self.scroll_to_bottom()
    
    @QtCore.pyqtSlot(int)
    def on_rag_search_ready(self, num_results: int):
        """处理RAG搜索完成"""
        # 更新"正在思考"消息
        self.thinking_message.setText("�� Searching for DOIs in literature...")
        
        # 添加RAG搜索结果显示
        rag_info = QLabel(f"🔍 RAG search complete: Found {num_results} relevant documents, extracting DOIs...")
        rag_info.setStyleSheet("""
            background-color: #FFF3E0;
            border: 1px solid #FF9800;
            border-radius: 5px;
            padding: 8px;
            margin: 5px;
            color: #F57C00;
            font-style: italic;
        """)
        rag_info.setWordWrap(True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, rag_info)
        self.scroll_to_bottom()

    @QtCore.pyqtSlot(list, int)
    def on_doi_search_ready(self, dois: list, literature_count: int):
        """处理DOI检索完成"""
        # 更新思考消息
        self.thinking_message.setText("🤖 Analyzing literature data...")
        self.scroll_to_bottom()

    @QtCore.pyqtSlot(dict)
    def on_recommendations_ready(self, recommendations: Dict[str, Any]):
        """处理推荐结果完成"""
        # 立即更新推荐面板
        if recommendations:
            self.current_recommendations = recommendations
            self.recommendation_panel.update_recommendations(recommendations)
            
            # 更新思考消息
            self.thinking_message.setText("✅ Recommendation analysis complete, generating response...")
            self.scroll_to_bottom()

    def scroll_to_bottom(self):
        """滚动到底部"""
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))
    
    def closeEvent(self, event):
        """关闭事件"""
        # 确保所有线程正确结束
        if hasattr(self, 'init_worker') and self.init_worker.isRunning():
            try:
                self._ignore_init_result = True
                self.init_worker.requestInterruption()
            except Exception:
                pass
            self.init_worker.wait(2000)
        
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            try:
                self.chat_worker.requestInterruption()
            except Exception:
                pass
            self.chat_worker.wait(2000)
        
        event.accept()


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("MXLLM Material Recommendation System")
    app.setApplicationVersion("1.0.0")
    
    # 设置应用图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 