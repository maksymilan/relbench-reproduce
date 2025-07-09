# 读取一个parquet文件

import pandas as pd
from pathlib import Path

def read_parquet_file(file_path: Path) -> pd.DataFrame:
    """
    读取一个parquet文件并返回DataFrame。
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"已成功读取 {file_path} 文件。")
        return df
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
# 示例用法
if __name__ == "__main__":
    # 请替换为实际的parquet文件路径
    file_path = Path("~/.cache/relbench/rel-amazon/tasks/user-churn/train.parquet")
    df = read_parquet_file(file_path)
    if df is not None:
        print("DataFrame内容:")
        print(len(df))
        print(df.head())