#!/usr/bin/env python3
"""
MXene材料推荐系统主启动脚本
基于RAG和AI预测的智能材料推荐系统
"""

import sys
import os
import logging
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # —— 捕捉原生层崩溃（segfault/abort）到文件 —— 
    """主函数"""
    # —— 新增：把原生层崩溃 & 死锁/卡住的栈写到文件 —— 
    import faulthandler, threading, time
    crash_log_dir = current_dir / "logs"
    crash_log_dir.mkdir(exist_ok=True)
    crash_log_file = open(crash_log_dir / "faulthandler.log", "w", buffering=1, encoding="utf-8")
    faulthandler.enable(file=crash_log_file)

    # 线程异常也写日志（Python 3.8+）
    def _thread_excepthook(args):
        import traceback
        print(f"[ThreadException] {args.exc_type.__name__}: {args.exc_value}", file=crash_log_file)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=crash_log_file)
    try:
        import threading as _t
        _t.excepthook = _thread_excepthook
    except Exception:
        pass
    """主函数"""
    # 设置全局异常处理
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print(f"\n💥 程序发生致命错误:")
        print(f"错误类型: {exc_type.__name__}")
        print(f"错误信息: {exc_value}")
        
        import traceback
        print("\n📋 详细堆栈信息:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        print(f"\n🔧 建议运行诊断脚本:")
        print(f"python {current_dir}/diagnose_crash.py")
    
    sys.excepthook = handle_exception
    
    try:
        # 确保日志目录存在
        log_dir = current_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("启动MXene材料推荐系统...")
        
        print("🧪 MXene材料推荐系统")
        print("📊 RAG + AI预测 + 智能推荐")
        print("=" * 50)
        
        # 检查依赖
        print("🔍 检查依赖...")
        try:
            from PyQt5.QtWidgets import QApplication
            print("  ✅ PyQt5")
            
            import openai
            print("  ✅ openai")
            
            import faiss
            print("  ✅ faiss")
            
            import importlib, threading, time
            st_mod = importlib.import_module("sentence_transformers")
            print("  ✅ sentence-transformers")

            # —— 新增：诊断补丁，只在调试期间用 —— 
            orig_ST = st_mod.SentenceTransformer

            class _PatchedST(orig_ST):
                def __init__(self, *a, **kw):
                    logging.getLogger("diag").info(
                        f"[ST.__init__] args={a} kw={kw} thread={threading.current_thread().name}")
                    super().__init__(*a, **kw)

                def encode(self, *a, **kw):
                    logging.getLogger("diag").info(
                        f"[ST.encode.start] thread={threading.current_thread().name} kw={kw}")
                    # 如果 20 秒还没返回，周期性把所有线程栈写入 faulthandler.log，定位死锁/卡住点
                    import faulthandler
                    faulthandler.dump_traceback_later(20, repeat=True, file=crash_log_file)
                    try:
                        return super().encode(*a, **kw, show_progress_bar=False)
                    finally:
                        faulthandler.cancel_dump_traceback_later()
                        logging.getLogger("diag").info("[ST.encode.end]")

            st_mod.SentenceTransformer = _PatchedST
            
            print("✅ 所有依赖检查通过")
            
        except ImportError as e:
            print(f"❌ 缺少必要依赖: {e}")
            print("请运行: pip install -r requirements.txt")
            print("或运行诊断脚本: python diagnose_crash.py")
            return 1
        
        # 启动GUI应用
        print("🚀 启动GUI应用...")
        try:
            from ui.main_window import main as start_gui
            return start_gui()
            
        except Exception as gui_error:
            print(f"❌ GUI启动失败: {gui_error}")
            logger.exception("GUI启动失败")
            
            import traceback
            traceback.print_exc()
            
            # 提供调试建议
            print("\n🐛 调试建议:")
            print("1. 运行诊断脚本: python diagnose_crash.py")
            print("2. 检查日志文件: logs/app.log")
            print("3. 确保所有依赖正确安装")
            return 1
        
    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
        return 0
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        
        # 确保能记录日志
        try:
            logging.exception("程序启动失败")
        except:
            pass
            
        import traceback
        traceback.print_exc()
        
        print("\n🐛 调试建议:")
        print("1. 运行诊断脚本: python diagnose_crash.py")
        print("2. 检查错误信息和堆栈跟踪")
        return 1

if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    sys.exit(main()) 