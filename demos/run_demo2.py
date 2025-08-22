#!/usr/bin/env python3
"""
运行器脚本 - Demo 2: JSON解析能力对比测试

简化的执行脚本，用于快速运行JSON解析对比测试
"""

import os
import sys
from pathlib import Path

# Add the demo directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from demo2_json_parsing import main as demo2_main

    print("🚀 启动Demo 2: JSON解析能力对比测试...")
    demo2_main()
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖项:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ 执行错误: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
