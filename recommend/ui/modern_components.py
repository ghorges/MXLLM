"""
现代化UI组件库
包含各种高级UI组件：输入框、按钮、卡片、加载动画等
"""

import math
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QFrame, QVBoxLayout, 
    QHBoxLayout, QGraphicsOpacityEffect, QGraphicsDropShadowEffect,
    QProgressBar, QTextEdit, QScrollArea, QStackedWidget
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, 
    pyqtSignal, QPoint, QSize, QSequentialAnimationGroup,
    QParallelAnimationGroup, pyqtProperty
)
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient,
    QRadialGradient, QPalette, QPixmap, QIcon, QFontMetrics,
    QPainterPath
)


class RippleEffect(QWidget):
    """水波纹效果组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.ripples = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_ripples)
        self.animation_timer.start(16)  # 60 FPS
    
    def add_ripple(self, position: QPoint, color: QColor = QColor(255, 255, 255, 100)):
        """添加水波纹"""
        ripple = {
            'pos': position,
            'radius': 0,
            'max_radius': max(self.width(), self.height()),
            'opacity': 1.0,
            'color': color
        }
        self.ripples.append(ripple)
    
    def update_ripples(self):
        """更新水波纹动画"""
        to_remove = []
        
        for ripple in self.ripples:
            ripple['radius'] += 5
            ripple['opacity'] -= 0.02
            
            if ripple['opacity'] <= 0:
                to_remove.append(ripple)
        
        for ripple in to_remove:
            self.ripples.remove(ripple)
        
        if self.ripples:
            self.update()
    
    def paintEvent(self, event):
        """绘制水波纹"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for ripple in self.ripples:
            color = QColor(ripple['color'])
            color.setAlphaF(ripple['opacity'])
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            painter.drawEllipse(
                ripple['pos'].x() - ripple['radius'],
                ripple['pos'].y() - ripple['radius'],
                ripple['radius'] * 2,
                ripple['radius'] * 2
            )


class ModernLineEdit(QLineEdit):
    """现代化输入框"""
    
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        
        # 设置样式
        self.setup_style()
        
        # 动画效果
        self.setup_animations()
        
        # 水波纹效果
        self.ripple_effect = RippleEffect(self)
        
    def setup_style(self):
        """设置样式"""
        self.setFont(QFont("微软雅黑", 14))
        self.setFixedHeight(50)
        
        self.setStyleSheet("""
            ModernLineEdit {
                background-color: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 25px;
                padding: 0 20px;
                color: #1e293b;
                font-size: 14px;
                selection-background-color: #6366f1;
            }
            ModernLineEdit:focus {
                border-color: #6366f1;
                background-color: #ffffff;
            }
            ModernLineEdit:hover {
                border-color: #94a3b8;
            }
        """)
    
    def setup_animations(self):
        """设置动画"""
        # 缩放动画
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(200)
        self.scale_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def focusInEvent(self, event):
        """获得焦点时的动画"""
        super().focusInEvent(event)
        
        # 添加水波纹效果
        self.ripple_effect.add_ripple(QPoint(self.width()//2, self.height()//2))
        
        # 轻微缩放
        current = self.geometry()
        self.scale_animation.setStartValue(current)
        
        new_rect = QRect(
            current.x() - 2,
            current.y() - 2, 
            current.width() + 4,
            current.height() + 4
        )
        self.scale_animation.setEndValue(new_rect)
        self.scale_animation.start()
    
    def focusOutEvent(self, event):
        """失去焦点时的动画"""
        super().focusOutEvent(event)
        
        # 恢复原始大小
        current = self.geometry()
        self.scale_animation.setStartValue(current)
        
        new_rect = QRect(
            current.x() + 2,
            current.y() + 2,
            current.width() - 4, 
            current.height() - 4
        )
        self.scale_animation.setEndValue(new_rect)
        self.scale_animation.start()
    
    def resizeEvent(self, event):
        """调整大小时更新水波纹效果位置"""
        super().resizeEvent(event)
        self.ripple_effect.resize(self.size())


class ModernProgressBar(QWidget):
    """现代化进度条"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self._progress = 0
        self._indeterminate = False
        
        # 动画
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update)
        self.animation_offset = 0
        
        self.setStyleSheet("background-color: transparent;")
    
    def setIndeterminate(self, indeterminate: bool):
        """设置不确定进度模式"""
        self._indeterminate = indeterminate
        if indeterminate:
            self.animation_timer.start(50)
        else:
            self.animation_timer.stop()
        self.update()
    
    def setValue(self, value: int):
        """设置进度值 (0-100)"""
        self._progress = max(0, min(100, value))
        self.update()
    
    def paintEvent(self, event):
        """绘制进度条"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 背景
        bg_rect = self.rect()
        painter.setBrush(QBrush(QColor("#e2e8f0")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bg_rect, 3, 3)
        
        if self._indeterminate:
            # 不确定进度的动画
            self.animation_offset = (self.animation_offset + 2) % (self.width() + 60)
            
            # 创建渐变
            gradient = QLinearGradient(0, 0, 60, 0)
            gradient.setColorAt(0, QColor("#6366f1"))
            gradient.setColorAt(0.5, QColor("#8b5cf6"))
            gradient.setColorAt(1, QColor("#6366f1"))
            
            painter.setBrush(QBrush(gradient))
            
            # 绘制移动的进度块
            progress_rect = QRect(
                self.animation_offset - 60,
                0,
                60,
                self.height()
            )
            painter.drawRoundedRect(progress_rect, 3, 3)
        else:
            # 确定进度
            if self._progress > 0:
                progress_width = int(self.width() * self._progress / 100)
                
                # 创建渐变
                gradient = QLinearGradient(0, 0, progress_width, 0)
                gradient.setColorAt(0, QColor("#6366f1"))
                gradient.setColorAt(1, QColor("#8b5cf6"))
                
                painter.setBrush(QBrush(gradient))
                
                progress_rect = QRect(0, 0, progress_width, self.height())
                painter.drawRoundedRect(progress_rect, 3, 3)


class LoadingSpinner(QWidget):
    """加载动画组件"""
    
    def __init__(self, size=40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        
        self.angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)
        
    def start(self):
        """开始动画"""
        self.timer.start(50)  # 20 FPS
        
    def stop(self):
        """停止动画"""
        self.timer.stop()
        
    def rotate(self):
        """旋转动画"""
        self.angle = (self.angle + 10) % 360
        self.update()
        
    def paintEvent(self, event):
        """绘制加载动画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 移动到中心
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)
        
        # 绘制圆点
        radius = min(self.width(), self.height()) / 2 - 5
        dot_radius = 3
        
        for i in range(8):
            angle = i * 45
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            
            # 计算透明度
            opacity = 1.0 - (i * 0.1)
            color = QColor("#6366f1")
            color.setAlphaF(opacity)
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(x, y), dot_radius, dot_radius)


class NotificationToast(QFrame):
    """通知吐司组件"""
    
    def __init__(self, message, notification_type="info", duration=3000, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.setup_ui(message, notification_type)
        self.setup_animations()
        
    def setup_ui(self, message, notification_type):
        """设置UI"""
        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedHeight(60)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        
        # 图标
        icon_map = {
            "success": "✅",
            "error": "❌", 
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        icon_label = QLabel(icon_map.get(notification_type, "ℹ️"))
        icon_label.setFont(QFont("微软雅黑", 16))
        
        # 消息文本
        message_label = QLabel(message)
        message_label.setFont(QFont("微软雅黑", 13))
        message_label.setWordWrap(True)
        
        layout.addWidget(icon_label)
        layout.addWidget(message_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # 设置样式
        color_map = {
            "success": {"bg": "#10b981", "text": "#ffffff"},
            "error": {"bg": "#ef4444", "text": "#ffffff"},
            "warning": {"bg": "#f59e0b", "text": "#ffffff"},
            "info": {"bg": "#3b82f6", "text": "#ffffff"}
        }
        
        colors = color_map.get(notification_type, color_map["info"])
        
        self.setStyleSheet(f"""
            NotificationToast {{
                background-color: {colors["bg"]};
                color: {colors["text"]};
                border-radius: 30px;
                border: none;
            }}
        """)
        
        # 阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)
    
    def setup_animations(self):
        """设置动画"""
        # 滑入动画
        self.slide_animation = QPropertyAnimation(self, b"geometry")
        self.slide_animation.setDuration(300)
        self.slide_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 淡出动画
        self.opacity_effect = QGraphicsOpacityEffect()
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def show_notification(self, parent_widget):
        """显示通知"""
        self.setParent(parent_widget)
        
        # 计算位置
        parent_rect = parent_widget.rect()
        self.resize(min(400, parent_rect.width() - 40), 60)
        
        start_pos = QRect(
            parent_rect.width() // 2 - self.width() // 2,
            -self.height(),
            self.width(),
            self.height()
        )
        
        end_pos = QRect(
            parent_rect.width() // 2 - self.width() // 2,
            20,
            self.width(),
            self.height()
        )
        
        # 设置初始位置
        self.setGeometry(start_pos)
        self.show()
        
        # 开始滑入动画
        self.slide_animation.setStartValue(start_pos)
        self.slide_animation.setEndValue(end_pos)
        self.slide_animation.start()
        
        # 自动隐藏
        if self.duration > 0:
            QTimer.singleShot(self.duration, self.hide_notification)
    
    def hide_notification(self):
        """隐藏通知"""
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.deleteLater)
        self.fade_animation.start()


class ModernScrollArea(QScrollArea):
    """现代化滚动区域"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self):
        """设置样式"""
        self.setStyleSheet("""
            ModernScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: rgba(0, 0, 0, 0.1);
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background-color: rgba(99, 102, 241, 0.6);
                border-radius: 6px;
                min-height: 30px;
                margin: 2px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: rgba(99, 102, 241, 0.8);
            }
            
            QScrollBar::handle:vertical:pressed {
                background-color: rgba(99, 102, 241, 1.0);
            }
            
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)


class AnimatedStackedWidget(QStackedWidget):
    """带动画的堆叠组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fade_animation = QPropertyAnimation(self, b"currentIndex")
        
    def setCurrentIndexAnimated(self, index):
        """带动画的切换页面"""
        if index == self.currentIndex():
            return
            
        # 创建淡入淡出效果
        current_widget = self.currentWidget()
        if current_widget:
            opacity_effect = QGraphicsOpacityEffect()
            current_widget.setGraphicsEffect(opacity_effect)
            
            fade_out = QPropertyAnimation(opacity_effect, b"opacity")
            fade_out.setDuration(150)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            
            def switch_page():
                self.setCurrentIndex(index)
                new_widget = self.currentWidget()
                if new_widget:
                    new_opacity_effect = QGraphicsOpacityEffect()
                    new_widget.setGraphicsEffect(new_opacity_effect)
                    
                    fade_in = QPropertyAnimation(new_opacity_effect, b"opacity")
                    fade_in.setDuration(150)
                    fade_in.setStartValue(0.0)
                    fade_in.setEndValue(1.0)
                    fade_in.start()
            
            fade_out.finished.connect(switch_page)
            fade_out.start()
        else:
            self.setCurrentIndex(index)


class GlowButton(QPushButton):
    """发光按钮"""
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.glow_effect = QGraphicsDropShadowEffect()
        self.setup_glow()
        self.setup_animations()
    
    def setup_glow(self):
        """设置发光效果"""
        self.glow_effect.setBlurRadius(20)
        self.glow_effect.setXOffset(0)
        self.glow_effect.setYOffset(0)
        self.glow_effect.setColor(QColor("#6366f1"))
        self.setGraphicsEffect(self.glow_effect)
        
        self.setStyleSheet("""
            GlowButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7c3aed, stop:1 #6366f1);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            GlowButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8b5cf6, stop:1 #7c3aed);
            }
            GlowButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
            }
        """)
    
    def setup_animations(self):
        """设置动画"""
        self.glow_animation = QPropertyAnimation(self.glow_effect, b"blurRadius")
        self.glow_animation.setDuration(200)
        self.glow_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        super().enterEvent(event)
        self.glow_animation.setStartValue(20)
        self.glow_animation.setEndValue(30)
        self.glow_animation.start()
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        super().leaveEvent(event)
        self.glow_animation.setStartValue(30)
        self.glow_animation.setEndValue(20)
        self.glow_animation.start()


def show_notification(parent, message, notification_type="info", duration=3000):
    """显示通知的便捷函数"""
    notification = NotificationToast(message, notification_type, duration)
    notification.show_notification(parent)
    return notification 