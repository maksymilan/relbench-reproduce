# file: debug_sampler.py

import torch
# 导入torch_geometric.typing来检查后端库
try:
    import torch_geometric.typing
    TORCH_GEOMETRIC_INSTALLED = True
except ImportError:
    TORCH_GEOMETRIC_INSTALLED = False

def check_sampler_dependencies():
    """
    一个用于调试的测试函数。
    它会检查 PyTorch Geometric 的 NeighborSampler 所需的后端依赖是否已正确安装。
    """
    print("="*50)
    print("开始检查 PyTorch Geometric Sampler 依赖项...")
    print("="*50)

    if not TORCH_GEOMETRIC_INSTALLED:
        print("❌ 错误: 'torch_geometric' 库未安装。")
        print("请先通过 'pip install torch_geometric' 安装。")
        return

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"PyG 可用: {TORCH_GEOMETRIC_INSTALLED}")
    print("-" * 50)

    # 模仿您代码中的逻辑进行检查
    if torch_geometric.typing.WITH_PYG_LIB:
        print("✅ 成功: 检测到 'pyg-lib' 已安装。")
        print("NeighborSampler 将使用最高性能的后端。您的环境配置正确！")

    elif torch_geometric.typing.WITH_TORCH_SPARSE:
        print("⚠️ 注意: 检测到只安装了 'torch-sparse'。")
        print("程序可以运行，但为了获得最佳性能和完整功能，强烈建议您安装 'pyg-lib'。")

    else:
        print("❌ 错误: 未检测到 'pyg-lib' 或 'torch-sparse'。")
        print("这是导致您遇到 'ImportError' 的直接原因。")
        print("\n解决方案: 请访问 PyG 官网获取正确的安装命令并执行：")
        print("https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")

    print("="*50)
    print("检查完毕。")
    print("="*50)


if __name__ == "__main__":
    check_sampler_dependencies()