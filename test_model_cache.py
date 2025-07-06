# file: find_cache.py
import os

try:
    # sentence-transformers 库提供了一个工具函数来获取其配置的缓存路径
    from sentence_transformers.util import get_cache_folder

    print("=" * 60)
    print("正在检测 'sentence-transformers' 的缓存目录...")
    print("=" * 60)

    # 这个函数会考虑所有环境变量和默认设置，返回最终生效的路径
    cache_path = get_cache_folder()

    print(f"✅ 程序实际使用的缓存根目录是: \n{cache_path}\n")
    print("请将您手动下载的模型文件夹（models--...）移动到上面显示的这个目录中。")
    print("=" * 60)

except ImportError:
    print("❌ 错误: 'sentence-transformers' 库似乎没有正确安装。")