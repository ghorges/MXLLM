"""
现代化主窗口界面
使用Material Design风格设计
支持深色/浅色主题切换，具有现代化动画效果
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter,
    QScrollArea, QFrame, QMessageBox, QProgressBar, QDialog,
    QDialogButtonBox, QFormLayout, QTextBrowser, QTabWidget,
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QButtonGroup,
    QStackedWidget, QGridLayout, QSpacerItem, QSizePolicy, QSlider,
    QComboBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, 
    QEasingCurve, QParallelAnimationGroup, QSequentialAnimationGroup,
    QRect, QSize, pyqtProperty
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QIcon, QPainter, QPen, QBrush,
    QLinearGradient, QPixmap, QFontDatabase
)
from PyQt5 import QtCore

# 添加父目录到路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.rag_system import RAGSystem
from core.llm_handler import LLMHandler
from core.config_cache import config_cache
from .modern_components import (
    ModernLineEdit, ModernProgressBar, LoadingSpinner, 
    NotificationToast, show_notification, GlowButton,
    ModernScrollArea, AnimatedStackedWidget
)

logger = logging.getLogger(__name__)


class ThemeManager:
    """主题管理器"""
    
    # 浅色主题
    LIGHT_THEME = {
        'primary': '#6366f1',
        'primary_dark': '#4f46e5',
        'primary_light': '#8b5cf6',
        'secondary': '#06b6d4',
        'accent': '#10b981',
        'background': '#ffffff',
        'surface': '#f8fafc',
        'surface_variant': '#f1f5f9',
        'on_primary': '#ffffff',
        'on_secondary': '#ffffff',
        'on_surface': '#1e293b',
        'on_surface_variant': '#64748b',
        'outline': '#e2e8f0',
        'outline_variant': '#cbd5e1',
        'shadow': 'rgba(0, 0, 0, 0.1)',
        'chat_user_bg': '#e0e7ff',
        'chat_user_border': '#c7d2fe',
        'chat_ai_bg': '#ecfdf5',
        'chat_ai_border': '#bbf7d0',
        'card_bg': '#ffffff',
        'card_hover': '#f8fafc',
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'info': '#3b82f6'
    }
    
    # 深色主题
    DARK_THEME = {
        'primary': '#8b5cf6',
        'primary_dark': '#7c3aed',
        'primary_light': '#a855f7',
        'secondary': '#0891b2',
        'accent': '#059669',
        'background': '#0f172a',
        'surface': '#1e293b',
        'surface_variant': '#334155',
        'on_primary': '#ffffff',
        'on_secondary': '#ffffff',
        'on_surface': '#f1f5f9',
        'on_surface_variant': '#94a3b8',
        'outline': '#475569',
        'outline_variant': '#64748b',
        'shadow': 'rgba(0, 0, 0, 0.3)',
        'chat_user_bg': '#312e81',
        'chat_user_border': '#4338ca',
        'chat_ai_bg': '#1e3a2e',
        'chat_ai_border': '#059669',
        'card_bg': '#1e293b',
        'card_hover': '#334155',
        'success': '#059669',
        'warning': '#d97706',
        'error': '#dc2626',
        'info': '#2563eb'
    }
    
    def __init__(self):
        self.current_theme = self.LIGHT_THEME
        self.is_dark_mode = False
    
    def toggle_theme(self):
        """切换主题"""
        self.is_dark_mode = not self.is_dark_mode
        self.current_theme = self.DARK_THEME if self.is_dark_mode else self.LIGHT_THEME
        return self.current_theme
    
    def get_color(self, color_name: str) -> str:
        """获取颜色值"""
        return self.current_theme.get(color_name, '#000000')


class ModernCard(QFrame):
    """现代化卡片组件"""
    
    def __init__(self, parent=None, elevation=2, radius=12):
        super().__init__(parent)
        self.elevation = elevation
        self.radius = radius
        self.theme_manager = ThemeManager()
        self.setup_card()
    
    def setup_card(self):
        """设置卡片样式"""
        self.setFrameStyle(QFrame.NoFrame)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(self.elevation * 4)
        shadow.setXOffset(0)
        shadow.setYOffset(self.elevation)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)
        
        self.update_theme()
    
    def update_theme(self):
        """更新主题"""
        bg_color = self.theme_manager.get_color('card_bg')
        hover_color = self.theme_manager.get_color('card_hover')
        
        self.setStyleSheet(f"""
            ModernCard {{
                background-color: {bg_color};
                border: none;
                border-radius: {self.radius}px;
                padding: 16px;
            }}
            ModernCard:hover {{
                background-color: {hover_color};
            }}
        """)


class ModernButton(QPushButton):
    """现代化按钮组件"""
    
    def __init__(self, text="", button_type="primary", size="medium", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.size = size
        self.theme_manager = ThemeManager()
        self.setup_button()
        self.setup_animations()
    
    def setup_button(self):
        """设置按钮样式"""
        self.setFixedHeight(self.get_height())
        self.setFont(QFont("微软雅黑", self.get_font_size(), QFont.Medium))
        self.update_theme()
    
    def get_height(self):
        """获取按钮高度"""
        sizes = {"small": 32, "medium": 40, "large": 48}
        return sizes.get(self.size, 40)
    
    def get_font_size(self):
        """获取字体大小"""
        sizes = {"small": 12, "medium": 14, "large": 16}
        return sizes.get(self.size, 14)
    
    def setup_animations(self):
        """设置动画效果"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(200)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def update_theme(self):
        """更新主题"""
        if self.button_type == "primary":
            bg_color = self.theme_manager.get_color('primary')
            hover_color = self.theme_manager.get_color('primary_dark')
            text_color = self.theme_manager.get_color('on_primary')
        elif self.button_type == "secondary":
            bg_color = self.theme_manager.get_color('secondary')
            hover_color = self.theme_manager.get_color('primary_dark')
            text_color = self.theme_manager.get_color('on_secondary')
        elif self.button_type == "outline":
            bg_color = "transparent"
            hover_color = self.theme_manager.get_color('surface_variant')
            text_color = self.theme_manager.get_color('primary')
        else:
            bg_color = self.theme_manager.get_color('surface_variant')
            hover_color = self.theme_manager.get_color('primary_light')
            text_color = self.theme_manager.get_color('on_surface')
        
        border_style = f"2px solid {self.theme_manager.get_color('primary')}" if self.button_type == "outline" else "none"
        
        self.setStyleSheet(f"""
            ModernButton {{
                background-color: {bg_color};
                color: {text_color};
                border: {border_style};
                border-radius: {self.get_height() // 2}px;
                padding: 0 16px;
                font-weight: 500;
                text-align: center;
            }}
            ModernButton:hover {{
                background-color: {hover_color};
            }}
            ModernButton:pressed {{
                background-color: {hover_color};
                transform: scale(0.98);
            }}
            ModernButton:disabled {{
                background-color: {self.theme_manager.get_color('outline_variant')};
                color: {self.theme_manager.get_color('on_surface_variant')};
            }}
        """)
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.9)
        self.fade_animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.fade_animation.setStartValue(0.9)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
        super().leaveEvent(event)


class ModernChatMessage(QFrame):
    """现代化聊天消息组件"""
    
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.theme_manager = ThemeManager()
        self.setup_ui(text)
        self.setup_animations()
    
    def setup_ui(self, text: str):
        """设置UI"""
        self.setFrameStyle(QFrame.NoFrame)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # 发送者信息
        header_layout = QHBoxLayout()
        
        # 头像
        avatar_label = QLabel()
        avatar_label.setFixedSize(32, 32)
        avatar_label.setStyleSheet(f"""
            QLabel {{
                background-color: {'#6366f1' if self.is_user else '#10b981'};
                border-radius: 16px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        avatar_label.setAlignment(Qt.AlignCenter)
        avatar_label.setText("You" if self.is_user else "AI Assistant")
        
        # 发送者名称
        sender_label = QLabel("You" if self.is_user else "AI Assistant")
        sender_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        sender_label.setStyleSheet(f"color: {self.theme_manager.get_color('on_surface')};")
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(sender_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # 消息内容
        message_text = QTextBrowser()
        message_text.setPlainText(text)
        # 移除最大高度限制，让内容完全展示
        message_text.setMaximumHeight(500)
        message_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        message_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        message_text.setFrameStyle(QFrame.NoFrame)
        message_text.setStyleSheet(f"""
            QTextBrowser {{
                background-color: transparent;
                border: none;
                color: {self.theme_manager.get_color('on_surface')};
                font-size: 14px;
                line-height: 1.5;
            }}
        """)
        
        layout.addWidget(message_text)
        self.setLayout(layout)
        
        # 设置消息样式
        self.update_message_style()
    
    def update_message_style(self):
        """更新消息样式"""
        if self.is_user:
            bg_color = self.theme_manager.get_color('chat_user_bg')
            border_color = self.theme_manager.get_color('chat_user_border')
        else:
            bg_color = self.theme_manager.get_color('chat_ai_bg')
            border_color = self.theme_manager.get_color('chat_ai_border')
        
        self.setStyleSheet(f"""
            ModernChatMessage {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 16px;
                margin: 4px;
            }}
        """)
    
    def setup_animations(self):
        """设置进入动画"""
        self.setFixedHeight(0)
        
        self.expand_animation = QPropertyAnimation(self, b"geometry")
        self.expand_animation.setDuration(300)
        self.expand_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 延迟启动动画
        QTimer.singleShot(50, self.start_expand_animation)
        pass
    
    def start_expand_animation(self):
        """开始展开动画"""
        target_height = self.sizeHint().height()
        current_rect = self.geometry()
        
        self.expand_animation.setStartValue(QRect(current_rect.x(), current_rect.y(), current_rect.width(), 0))
        self.expand_animation.setEndValue(QRect(current_rect.x(), current_rect.y(), current_rect.width(), target_height))
        self.expand_animation.start()
        pass


class ModernRecommendationPanel(QWidget):
    """现代化推荐结果面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = ThemeManager()
        self.setup_ui()
    
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
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 标题区域
        title_layout = QHBoxLayout()
        
        title_label = QLabel("🔬 AI Recommendation Results")
        title_label.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.theme_manager.get_color('primary')};")
        
        # 刷新按钮
        refresh_btn = ModernButton("🔄", "outline", "small")
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.setToolTip("Refresh recommendations")
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(refresh_btn)
        
        layout.addLayout(title_layout)
        
        # 创建现代化标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self.get_tab_style())
        
        # 各个标签页
        self.formulas_tab = self.create_content_tab()
        self.tab_widget.addTab(self.formulas_tab, "🧪 Chemical Formulas")
        
        self.synthesis_tab = self.create_content_tab()
        self.tab_widget.addTab(self.synthesis_tab, "⚗️ Synthesis Methods")
        
        self.testing_tab = self.create_content_tab()
        self.tab_widget.addTab(self.testing_tab, "🔬 Testing Procedures")
        
        self.prediction_tab = self.create_content_tab()
        self.tab_widget.addTab(self.prediction_tab, "🤖 AI Prediction")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def create_content_tab(self):
        """创建内容标签页"""
        tab = QTextBrowser()
        tab.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {self.theme_manager.get_color('surface')};
                border: none;
                border-radius: 12px;
                padding: 16px;
                color: {self.theme_manager.get_color('on_surface')};
                font-size: 17px;
                line-height: 1.6;
            }}
        """)
        return tab
    
    def get_tab_style(self):
        """获取标签页样式"""
        return f"""
            QTabWidget::pane {{
                border: none;
                background-color: transparent;
                border-radius: 12px;
                margin-top: 8px;
            }}
            QTabBar::tab {{
                background-color: {self.theme_manager.get_color('surface_variant')};
                color: {self.theme_manager.get_color('on_surface_variant')};
                padding: 12px 20px;
                margin-right: 4px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: 500;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.theme_manager.get_color('primary')};
                color: {self.theme_manager.get_color('on_primary')};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {self.theme_manager.get_color('primary_light')};
                color: {self.theme_manager.get_color('on_primary')};
            }}
        """
    
    def update_recommendations(self, recommendations: Dict[str, Any]):
        """更新推荐结果"""
        try:
            # 检查新格式的数据结构
            if 'content' in recommendations and isinstance(recommendations['content'], list):
                # 新的JSON格式
                content = recommendations['content']
                if content:
                    # 格式化各个标签页内容
                    formulas_html = self._format_materials(content)
                    self.formulas_tab.setHtml(formulas_html)
                    
                    synthesis_html = self._format_synthesis_steps(content)
                    self.synthesis_tab.setHtml(synthesis_html)
                    
                    testing_html = self._format_testing_steps(content)
                    self.testing_tab.setHtml(testing_html)
                    
                    prediction_html = self._format_performance_data(content)
                    self.prediction_tab.setHtml(prediction_html)
                else:
                    # 空内容处理
                    empty_html = self._get_empty_content_html()
                    self.formulas_tab.setHtml(empty_html)
                    self.synthesis_tab.setHtml(empty_html)
                    self.testing_tab.setHtml(empty_html)
                    self.prediction_tab.setHtml(empty_html)
            else:
                # 旧格式的兼容处理
                formulas_html = self._format_modern_content(recommendations.get('chemical_formulas', []), "Chemical Formulas")
                self.formulas_tab.setHtml(formulas_html)
                
                synthesis_html = self._format_modern_content(recommendations.get('synthesis_methods', []), "Synthesis Methods")
                self.synthesis_tab.setHtml(synthesis_html)
                
                testing_html = self._format_modern_content(recommendations.get('testing_procedures', []), "Testing Procedures")
                self.testing_tab.setHtml(testing_html)
                
                prediction_html = self._format_modern_content(recommendations.get('ai_predictions', []), "AI Prediction")
                self.prediction_tab.setHtml(prediction_html)
            
        except Exception as e:
            logger.error(f"更新推荐结果失败: {e}")
    
    def _format_modern_content(self, items: List[Dict[str, Any]], content_type: str) -> str:
        """格式化现代化内容"""
        if not items:
            return f"""
            <div style='text-align: center; padding: 40px; color: #64748b;'>
                <h3>No {content_type} recommendations</h3>
                <p>Please try asking again or check your input.</p>
            </div>
            """
        
        html = f"""
        <style>
            .recommendation-card {{
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border-left: 4px solid #6366f1;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .recommendation-title {{
                color: #1e293b;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 12px;
            }}
            .recommendation-meta {{
                color: #64748b;
                font-size: 14px;
                margin-bottom: 8px;
            }}
            .recommendation-description {{
                color: #475569;
                line-height: 1.6;
            }}
            .confidence-bar {{
                background: #e2e8f0;
                height: 6px;
                border-radius: 3px;
                margin: 8px 0;
                overflow: hidden;
            }}
            .confidence-fill {{
                background: linear-gradient(90deg, #10b981, #059669);
                height: 100%;
                border-radius: 3px;
            }}
        </style>
        """
        
        for i, item in enumerate(items, 1):
            score = item.get('score', 0)
            confidence_width = int(score * 100) if isinstance(score, float) else 85
            
            # 格式化相关度
            if isinstance(score, float):
                score_text = f"{score:.1%}"
            else:
                score_text = f"{score}%"
            
            html += f"""
            <div class="recommendation-card">
                <div class="recommendation-title">
                    🔸 {i}. {item.get('formula', item.get('method', item.get('procedure', 'Unknown')))}
                </div>
                <div class="recommendation-meta">
                    <strong>Source:</strong> {item.get('source', 'Unknown')} | 
                    <strong>Confidence:</strong> {score_text}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_width}%;"></div>
                </div>
                <div class="recommendation-description">
                    {item.get('description', 'No detailed description available')}
                </div>
            </div>
            """
        
        return html

    def _get_empty_content_html(self) -> str:
        """获取空内容的HTML"""
        return f"""
        <div style='text-align: center; padding: 40px; color: #64748b;'>
            <h3>No recommendations</h3>
            <p>Please try asking again or check your input.</p>
        </div>
        """

    def _format_materials(self, content: List[Dict[str, Any]]) -> str:
        """格式化材料信息"""
        if not content:
            return self._get_empty_content_html()
        
        html = f"""
        <style>
            .material-card {{
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border-left: 4px solid #0ea5e9;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .material-title {{
                color: #0c4a6e;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .material-meta {{
                color: #64748b;
                font-size: 14px;
                margin-bottom: 12px;
                background: rgba(255,255,255,0.7);
                padding: 8px 12px;
                border-radius: 6px;
            }}
            .performance-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }}
            .performance-item {{
                background: rgba(255,255,255,0.8);
                padding: 8px;
                border-radius: 6px;
                text-align: center;
                font-size: 13px;
            }}
            .confidence-indicator {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            .confidence-high {{ background: #dcfce7; color: #166534; }}
            .confidence-medium {{ background: #fef3c7; color: #92400e; }}
            .confidence-low {{ background: #fee2e2; color: #991b1b; }}
        </style>
        """
        
        for i, item in enumerate(content, 1):
            material = item.get('material', {})
            performance = item.get('performance', {})
            confidence = item.get('confidence', 'medium')
            
            # 化学式和基本信息
            formula = material.get('chemical_formula', 'Unknown')
            name = material.get('name', '')
            composition_type = material.get('composition_type', '')
            structure_type = material.get('structure_type', '')
            morphology = material.get('morphology', '')
            
            confidence_class = f"confidence-{confidence}"
            confidence_text = {"high": "High", "medium": "Medium", "low": "Low"}.get(confidence, confidence)
            
            html += f"""
            <div class="material-card">
                <div class="material-title">
                    🧪 {i}. {formula}
                    <span class="confidence-indicator {confidence_class}">Confidence: {confidence_text}</span>
                </div>
                
                <div class="material-meta">
                    <div style="margin-bottom: 8px;">
                        <strong>Material Information:</strong>
                    </div>
            """
            
            if name:
                html += f"<div>• Name: {name}</div>"
            if composition_type:
                html += f"<div>• Composition Type: {composition_type}</div>"
            if structure_type:
                html += f"<div>• Structure Type: {structure_type}</div>"
            if morphology:
                html += f"<div>• Morphology: {morphology}</div>"
            
            html += "</div>"
            
            # 分子式标签页不显示性能数据，性能数据在AI预测标签页中显示
            
            html += "</div>"
        
        return html

    def _format_synthesis_steps(self, content: List[Dict[str, Any]]) -> str:
        """格式化合成步骤"""
        if not content:
            return self._get_empty_content_html()
        
        html = f"""
        <style>
            .synthesis-container {{
                font-family: 'Microsoft YaHei', sans-serif;
            }}
            .synthesis-card {{
                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border-left: 4px solid #16a34a;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .step-item {{
                background: rgba(255,255,255,0.8);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 12px;
                border-left: 3px solid #22c55e;
            }}
            .step-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}
            .step-number {{
                background: #16a34a;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
            }}
            .step-name {{
                font-weight: 600;
                color: #15803d;
                flex: 1;
            }}
            .role-badge {{
                background: #059669;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
            }}
            .inferred-badge {{
                background: #f59e0b;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
            }}
            .step-details {{
                display: grid;
                gap: 8px;
            }}
            .detail-row {{
                display: grid;
                grid-template-columns: 80px 1fr;
                gap: 10px;
                align-items: start;
            }}
            .detail-label {{
                font-weight: 600;
                color: #374151;
                font-size: 13px;
            }}
            .detail-value {{
                color: #6b7280;
                font-size: 13px;
                line-height: 1.4;
            }}
            .conditions-list {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .conditions-list li {{
                background: rgba(34, 197, 94, 0.1);
                padding: 4px 8px;
                margin: 2px 0;
                border-radius: 4px;
                font-size: 12px;
            }}
        </style>
        <div class="synthesis-container">
        """
        
        for i, item in enumerate(content, 1):
            synthesis_steps = item.get('synthesis_steps', [])
            material = item.get('material', {})
            formula = material.get('chemical_formula', 'Unknown')
            
            if not synthesis_steps:
                continue
                
            html += f"""
            <div class="synthesis-card">
                <div style="font-size: 16px; font-weight: 600; color: #15803d; margin-bottom: 15px;">
                    ⚗️ {formula} - Synthesis Steps
                </div>
            """
            
            for step in synthesis_steps:
                step_num = step.get('step', 0)
                step_name = step.get('step_name', 'Unknown Step')
                role = step.get('role', '')
                method = step.get('method', '')
                inputs = step.get('inputs', [])
                conditions = step.get('conditions', [])
                outputs = step.get('outputs', '')
                notes = step.get('notes', '')
                inferred = step.get('inferred', False)
                justification = step.get('justification', '')
                
                html += f"""
                <div class="step-item">
                    <div class="step-header">
                        <div class="step-number">{step_num}</div>
                        <div class="step-name">{step_name}</div>
                """
                
                if role:
                    html += f'<span class="role-badge">{role}</span>'
                if inferred:
                    html += f'<span class="inferred-badge">Inferred Step</span>'
                
                html += """
                    </div>
                    <div class="step-details">
                """
                
                if method:
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Method:</div>
                        <div class="detail-value">{method}</div>
                    </div>
                    """
                
                if inputs:
                    inputs_text = self._format_value_for_ui(inputs)
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Ingredients:</div>
                        <div class="detail-value">{inputs_text}</div>
                    </div>
                    """
                
                if conditions:
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Conditions:</div>
                        <div class="detail-value">
                            <ul class="conditions-list">
                    """
                    for condition in conditions:
                        condition_text = self._format_value_for_ui(condition)
                        html += f"<li>{condition_text}</li>"
                    html += """
                            </ul>
                        </div>
                    </div>
                    """
                
                if outputs:
                    outputs_text = self._format_value_for_ui(outputs)
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Product:</div>
                        <div class="detail-value">{outputs_text}</div>
                    </div>
                    """
                
                if notes:
                    notes_text = self._format_value_for_ui(notes)
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Notes:</div>
                        <div class="detail-value">{notes_text}</div>
                    </div>
                    """
                
                if inferred and justification:
                    justification_text = self._format_value_for_ui(justification)
                    html += f"""
                    <div class="detail-row">
                        <div class="detail-label">Justification:</div>
                        <div class="detail-value" style="color: #f59e0b; font-style: italic;">{justification_text}</div>
                    </div>
                    """
                
                html += """
                    </div>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _format_testing_steps(self, content: List[Dict[str, Any]]) -> str:
        """格式化测试步骤"""
        if not content:
            return self._get_empty_content_html()
        
        html = f"""
        <style>
            .testing-container {{
                font-family: 'Microsoft YaHei', sans-serif;
            }}
            .testing-card {{
                background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border-left: 4px solid #9333ea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .test-step-item {{
                background: rgba(255,255,255,0.8);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 12px;
                border-left: 3px solid #a855f7;
            }}
            .test-step-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}
            .test-step-number {{
                background: #9333ea;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
            }}
            .test-step-name {{
                font-weight: 600;
                color: #7c2d12;
                flex: 1;
            }}
            .test-inferred-badge {{
                background: #f59e0b;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
            }}
            .test-step-details {{
                display: grid;
                gap: 8px;
            }}
            .test-detail-row {{
                display: grid;
                grid-template-columns: 80px 1fr;
                gap: 10px;
                align-items: start;
            }}
            .test-detail-label {{
                font-weight: 600;
                color: #374151;
                font-size: 13px;
            }}
            .test-detail-value {{
                color: #6b7280;
                font-size: 13px;
                line-height: 1.4;
            }}
            .parameters-list {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .parameters-list li {{
                background: rgba(147, 51, 234, 0.1);
                padding: 4px 8px;
                margin: 2px 0;
                border-radius: 4px;
                font-size: 12px;
            }}
        </style>
        <div class="testing-container">
        """
        
        for i, item in enumerate(content, 1):
            testing_steps = item.get('testing_steps', [])
            material = item.get('material', {})
            formula = material.get('chemical_formula', 'Unknown')
            
            if not testing_steps:
                continue
                
            html += f"""
            <div class="testing-card">
                <div style="font-size: 16px; font-weight: 600; color: #7c2d12; margin-bottom: 15px;">
                    🔬 {formula} - Testing Steps
                </div>
            """
            
            for step in testing_steps:
                step_num = step.get('step', 0)
                step_name = step.get('step_name', 'Unknown Step')
                method = step.get('method', '')
                parameters = step.get('parameters', [])
                notes = step.get('notes', '')
                inferred = step.get('inferred', False)
                justification = step.get('justification', '')
                
                html += f"""
                <div class="test-step-item">
                    <div class="test-step-header">
                        <div class="test-step-number">{step_num}</div>
                        <div class="test-step-name">{step_name}</div>
                """
                
                if inferred:
                    html += f'<span class="test-inferred-badge">Inferred Step</span>'
                
                html += """
                    </div>
                    <div class="test-step-details">
                """
                
                if method:
                    method_text = self._format_value_for_ui(method)
                    html += f"""
                    <div class="test-detail-row">
                        <div class="test-detail-label">Method:</div>
                        <div class="test-detail-value">{method_text}</div>
                    </div>
                    """
                
                if parameters:
                    html += f"""
                    <div class="test-detail-row">
                        <div class="test-detail-label">Parameters:</div>
                        <div class="test-detail-value">
                            <ul class="parameters-list">
                    """
                    for param in parameters:
                        param_text = self._format_value_for_ui(param)
                        html += f"<li>{param_text}</li>"
                    html += """
                            </ul>
                        </div>
                    </div>
                    """
                
                if notes:
                    notes_text = self._format_value_for_ui(notes)
                    html += f"""
                    <div class="test-detail-row">
                        <div class="test-detail-label">Notes:</div>
                        <div class="test-detail-value">{notes_text}</div>
                    </div>
                    """
                
                if inferred and justification:
                    justification_text = self._format_value_for_ui(justification)
                    html += f"""
                    <div class="test-detail-row">
                        <div class="test-detail-label">Justification:</div>
                        <div class="test-detail-value" style="color: #f59e0b; font-style: italic;">{justification_text}</div>
                    </div>
                    """
                
                html += """
                    </div>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _format_performance_data(self, content: List[Dict[str, Any]]) -> str:
        """格式化性能数据"""
        if not content:
            return self._get_empty_content_html()
        
        html = f"""
        <style>
            .performance-container {{
                font-family: 'Microsoft YaHei', sans-serif;
            }}
            .performance-card {{
                background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border-left: 4px solid #ea580c;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .perf-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .perf-title {{
                font-size: 16px;
                font-weight: 600;
                color: #9a3412;
            }}
            .confidence-tag {{
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
            }}
            .conf-high {{ background: #dcfce7; color: #166534; }}
            .conf-medium {{ background: #fef3c7; color: #92400e; }}
            .conf-low {{ background: #fee2e2; color: #991b1b; }}
            .perf-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            .perf-metric {{
                background: rgba(255,255,255,0.9);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid rgba(234, 88, 12, 0.2);
            }}
            .perf-metric-label {{
                font-size: 12px;
                color: #9a3412;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            .perf-metric-value {{
                font-size: 18px;
                color: #7c2d12;
                font-weight: bold;
            }}
            .source-info {{
                background: rgba(255,255,255,0.7);
                padding: 10px;
                border-radius: 6px;
                font-size: 13px;
                color: #6b7280;
            }}
        </style>
        <div class="performance-container">
        """
        
        # 获取来源信息
        source = content[0].get('source', 'Unknown source') if content else 'Unknown source'
        doi = content[0].get('doi', '') if content else ''
        
        # 添加全局信息
        html += f"""
        <div style="background: rgba(234, 88, 12, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0; color: #9a3412;">📊 Performance Data Overview</h3>
            <div><strong>Data Source:</strong> {source}</div>
        """
        
        if doi:
            html += f'<div><strong>DOI:</strong> <a href="https://doi.org/{doi}" target="_blank" style="color: #ea580c;">{doi}</a></div>'
        
        html += "</div>"
        
        for i, item in enumerate(content, 1):
            material = item.get('material', {})
            performance = item.get('performance', {})
            confidence = item.get('confidence', 'medium')
            
            formula = material.get('chemical_formula', 'Unknown')
            confidence_class = f"conf-{confidence}"
            confidence_text = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence"}.get(confidence, confidence)
            
            html += f"""
            <div class="performance-card">
                <div class="perf-header">
                    <div class="perf-title">🎯 {formula}</div>
                    <div class="confidence-tag {confidence_class}">{confidence_text}</div>
                </div>
            """
            
            if performance:
                html += '<div class="perf-grid">'
                
                for key, value in performance.items():
                    if value and value != "":
                        display_name = {
                            'rl_min': 'RL Min',
                            'matching_thickness': 'Matching Thickness', 
                            'eab': 'Effective Absorption Bandwidth',
                            'other': 'Other Performance'
                        }.get(key, key.upper())
                        
                        # HTML转义特殊字符
                        import html as html_module
                        escaped_value = html_module.escape(str(value))
                        html += f"""
                        <div class="perf-metric">
                            <div class="perf-metric-label">{display_name}</div>
                            <div class="perf-metric-value">{escaped_value}</div>
                        </div>
                        """
                
                html += '</div>'
            else:
                html += '<div style="text-align: center; color: #9a3412; padding: 20px;">No specific performance data available</div>'
            
            # 材料附加信息
            additional_info = []
            if material.get('composition_type'):
                additional_info.append(f"Composition Type: {material['composition_type']}")
            if material.get('structure_type'):
                additional_info.append(f"Structure Type: {material['structure_type']}")
            if material.get('morphology'):
                additional_info.append(f"Morphology: {material['morphology']}")
            
            if additional_info:
                html += f"""
                <div class="source-info">
                    <strong>Material Characteristics:</strong> {' | '.join(additional_info)}
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html


class ModernMainWindow(QMainWindow):
    """现代化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        self.setup_window()
        self.setup_ui()
        self.setup_animations()
        
        # 初始化组件
        self.rag_system = None
        self.llm_handler = LLMHandler()
        self.conversation_history = []
        self.current_recommendations = {}
        
        # 启动欢迎动画
        self.start_welcome_sequence()
    
    def setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("MXLLM Material Recommendation System - Modern Edition")
        # 设置为全屏启动
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置窗口图标
        self.setWindowIcon(QIcon())  # 可以设置自定义图标
        
        # 应用全局主题
        self.apply_global_theme()
    
    def apply_global_theme(self):
        """应用全局主题"""
        bg_color = self.theme_manager.get_color('background')
        surface_color = self.theme_manager.get_color('surface')
        text_color = self.theme_manager.get_color('on_surface')
        
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg_color};
                color: {text_color};
            }}
            QWidget {{
                font-family: '微软雅黑', 'Segoe UI', sans-serif;
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {self.theme_manager.get_color('surface_variant')};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.theme_manager.get_color('outline')};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {self.theme_manager.get_color('primary')};
            }}
        """)
    
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(24)
        main_layout.setContentsMargins(24, 24, 24, 24)
        central_widget.setLayout(main_layout)
        
        # 左侧聊天区域
        chat_container = ModernCard(elevation=3, radius=16)
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        chat_widget = self.create_modern_chat_widget()
        chat_layout.addWidget(chat_widget)
        chat_container.setLayout(chat_layout)
        
        # 右侧推荐区域
        recommendation_container = ModernCard(elevation=3, radius=16)
        recommendation_layout = QVBoxLayout()
        recommendation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.recommendation_panel = ModernRecommendationPanel()
        recommendation_layout.addWidget(self.recommendation_panel)
        recommendation_container.setLayout(recommendation_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(chat_container)
        splitter.addWidget(recommendation_container)
        splitter.setSizes([800, 600])  # 设置初始比例
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {self.theme_manager.get_color('outline')};
                width: 2px;
            }}
        """)
        
        main_layout.addWidget(splitter)
        
        # 现代化状态栏
        self.setup_modern_statusbar()
    
    def create_modern_chat_widget(self) -> QWidget:
        """创建现代化聊天区域"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 顶部工具栏
        toolbar_layout = QHBoxLayout()
        
        # 标题
        title_label = QLabel("💬 MXLLM Intelligent Assistant")
        title_label.setFont(QFont("微软雅黑", 20, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.theme_manager.get_color('primary')};")
        
        # 工具按钮
        self.theme_toggle_btn = ModernButton("🌙", "outline", "small")
        self.theme_toggle_btn.setFixedSize(40, 40)
        self.theme_toggle_btn.setToolTip("Toggle Dark/Light Mode")
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)
        
        self.api_config_btn = ModernButton("⚙️ Config", "outline", "medium")
        self.api_config_btn.clicked.connect(self.show_api_config)
        
        self.init_system_btn = ModernButton("🔄 Initialize", "secondary", "medium")
        self.init_system_btn.clicked.connect(self.initialize_system)
        
        toolbar_layout.addWidget(title_label)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.theme_toggle_btn)
        toolbar_layout.addWidget(self.api_config_btn)
        toolbar_layout.addWidget(self.init_system_btn)
        
        layout.addLayout(toolbar_layout)
        
        # 聊天记录区域
        self.chat_scroll = ModernScrollArea()
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.addStretch()
        self.chat_widget.setLayout(self.chat_layout)
        self.chat_scroll.setWidget(self.chat_widget)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(f"""
            ModernScrollArea {{
                background-color: {self.theme_manager.get_color('surface')};
                border-radius: 12px;
                border: 1px solid {self.theme_manager.get_color('outline')};
            }}
        """)
        
        layout.addWidget(self.chat_scroll)
        
        # 输入区域
        input_container = ModernCard(elevation=2, radius=12)
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(16, 12, 16, 12)
        
        self.message_input = ModernLineEdit("✨ Please enter your question (e.g., I need a MXLLM material for high-frequency absorption)...")
        self.message_input.returnPressed.connect(self.send_message)
        
        self.send_button = GlowButton("Send")
        self.send_button.setFixedSize(80, 44)
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        input_container.setLayout(input_layout)
        
        layout.addWidget(input_container)
        
        # 进度指示器
        self.progress_container = QWidget()
        self.progress_container.setVisible(False)
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 8, 0, 8)
        
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setIndeterminate(True)
        progress_layout.addWidget(self.progress_bar)
        self.progress_container.setLayout(progress_layout)
        
        layout.addWidget(self.progress_container)
        
        widget.setLayout(layout)
        return widget
    
    def setup_modern_statusbar(self):
        """设置现代化状态栏"""
        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(24, 8, 24, 8)
        
        self.status_label = QLabel("🚀 System Ready")
        self.status_label.setFont(QFont("微软雅黑", 12))
        self.status_label.setStyleSheet(f"color: {self.theme_manager.get_color('on_surface_variant')};")
        
        # 状态指示点
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {self.theme_manager.get_color('success')};")
        
        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        status_widget.setLayout(status_layout)
        self.statusBar().addWidget(status_widget, 1)
        
        # 移除默认状态栏样式
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {self.theme_manager.get_color('surface')};
                border-top: 1px solid {self.theme_manager.get_color('outline')};
            }}
        """)
    
    def setup_animations(self):
        """设置动画效果"""
        # 窗口淡入动画
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_in_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def start_welcome_sequence(self):
        """启动欢迎动画序列"""
        # 启动淡入动画
        self.fade_in_animation.start()
        
        # 添加欢迎消息
        QTimer.singleShot(300, self.add_welcome_message)
    
    def add_welcome_message(self):
        """添加欢迎消息"""
        welcome_text = """🎉 Welcome to the MXLLM Material Recommendation System Modern Edition!

✨ New Features:
• 🎨 New modernized interface design
• 🌓 Dark/Light mode switching
• 🎭 Smooth animation effects
• 📱 Responsive layout design

🚀 Quick Start:
1. Click "⚙️ Config" to set up your AI service
2. Click "🔄 Initialize" to start the RAG system
3. Start chatting with the AI assistant!"""
        
        self.add_system_message(welcome_text)
    
    def toggle_theme(self):
        """切换主题"""
        # 切换主题
        self.theme_manager.toggle_theme()
        
        # 更新按钮图标
        icon = "☀️" if self.theme_manager.is_dark_mode else "🌙"
        self.theme_toggle_btn.setText(icon)
        
        # 更新所有组件主题
        self.update_all_themes()
        
        # 添加主题切换消息
        theme_name = "Dark Mode" if self.theme_manager.is_dark_mode else "Light Mode"
        self.add_system_message(f"�� Switched to {theme_name}")
    
    def update_all_themes(self):
        """更新所有组件的主题"""
        # 重新应用全局样式
        self.apply_global_theme()
        
        # 更新推荐面板主题
        if hasattr(self, 'recommendation_panel'):
            self.recommendation_panel.theme_manager = self.theme_manager
            self.recommendation_panel.setup_ui()
        
        # 更新聊天滚动区域样式
        if hasattr(self, 'chat_scroll'):
            self.chat_scroll.setStyleSheet(f"""
                ModernScrollArea {{
                    background-color: {self.theme_manager.get_color('surface')};
                    border-radius: 12px;
                    border: 1px solid {self.theme_manager.get_color('outline')};
                }}
            """)
        
        # 更新状态栏
        if hasattr(self, 'status_label'):
            self.status_label.setStyleSheet(f"color: {self.theme_manager.get_color('on_surface_variant')};")
        
        # 显示主题切换通知
        theme_name = "Dark Mode" if self.theme_manager.is_dark_mode else "Light Mode"
        show_notification(self, f"Switched to {theme_name}", "success", 2000)
    
    def show_api_config(self):
        """显示API配置对话框"""
        # 这里可以创建一个现代化的配置对话框
        QMessageBox.information(self, "Configuration", "API configuration functionality under development...")
    
    def initialize_system(self):
        """初始化系统"""
        self.add_system_message("🔄 Initializing RAG system, please wait...")
        # 实际初始化逻辑
    
    def add_system_message(self, message: str):
        """添加系统消息"""
        system_card = ModernCard(elevation=1, radius=12)
        system_layout = QVBoxLayout()
        system_layout.setContentsMargins(16, 12, 16, 12)
        
        # 系统消息图标和标题
        header_layout = QHBoxLayout()
        icon_label = QLabel("🤖")
        icon_label.setFont(QFont("微软雅黑", 16))
        
        title_label = QLabel("System Message")
        title_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.theme_manager.get_color('primary')};")
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # 消息内容
        content_label = QLabel(message)
        content_label.setWordWrap(True)
        content_label.setFont(QFont("微软雅黑", 13))
        content_label.setStyleSheet(f"""
            color: {self.theme_manager.get_color('on_surface')};
            line-height: 1.5;
            padding: 8px 0;
        """)
        
        system_layout.addLayout(header_layout)
        system_layout.addWidget(content_label)
        system_card.setLayout(system_layout)
        
        # 设置系统消息样式
        primary_color = self.theme_manager.get_color('primary')
        secondary_color = self.theme_manager.get_color('secondary')
        system_card.setStyleSheet(f"""
            ModernCard {{
                background: linear-gradient(135deg, {primary_color}15, {secondary_color}15);
                border: 1px solid {primary_color}30;
            }}
        """)
        
        # 添加到聊天区域并显示通知
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, system_card)
        self.scroll_to_bottom()
        
        # 显示toast通知（可选）
        if "Success" in message or "✅" in message:
            show_notification(self, "System operation successful!", "success", 2000)
    
    def send_message(self):
        """发送消息"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # 添加用户消息
        user_message = ModernChatMessage(message, True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_message)
        
        # 清空输入框
        self.message_input.clear()
        
        # 模拟AI回复
        QTimer.singleShot(1000, lambda: self.add_ai_reply("This is a simulated AI reply."))
        
        self.scroll_to_bottom()
    
    def add_ai_reply(self, reply: str):
        """添加AI回复"""
        ai_message = ModernChatMessage(reply, False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, ai_message)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """滚动到底部"""
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("MXLLM Material Recommendation System - Modern Edition")
    app.setApplicationVersion("2.0.0")
    
    # 设置应用样式
    app.setStyle('Fusion')  # 使用Fusion样式作为基础
    
    # 创建主窗口
    window = ModernMainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 