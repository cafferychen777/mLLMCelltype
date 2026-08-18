#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载PopV的肠道相关预训练模型
"""

import os
from huggingface_hub import snapshot_download

# 创建模型保存目录
# 设置数据目录
BASE_DIR = "data"
intestine_dir = os.path.join(BASE_DIR, "intestine")
os.makedirs(intestine_dir, exist_ok=True)

# 要下载的模型列表
models = [
    "popV/tabula_sapiens_Small_Intestine",
    "popV/tabula_sapiens_Large_Intestine"
]

# 下载每个模型
for model_name in models:
    print(f"\n正在下载模型: {model_name}")
    try:
        # 使用原始模型名作为目录名
        model_id = model_name.split('/')[-1]
        local_dir = os.path.join(intestine_dir, model_id)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"模型已保存到: {local_dir}")
    except Exception as e:
        print(f"下载 {model_name} 时出错: {str(e)}")

print("\n下载完成！")
print(f"\n模型已下载到： {intestine_dir}")
