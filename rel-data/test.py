# # 读取一个parquet文件

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
    
# # 示例用法
# if __name__ == "__main__":
#     # 请替换为实际的parquet文件路径
#     file_path = Path("~/relbench-reproduce/rel-data/rel-amazon/all_customer_trees.jsonl")
#     #  读取一个jsonl文件并输出这个jsonl文件的行数

# 读取一个jsonl文件并输出这个jsonl文件的行数
import json
from pathlib import Path
def count_jsonl_lines(file_path: Path) -> int:
    """
    读取一个jsonl文件并返回行数。
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"文件 {file_path} 有 {line_count} 行。")
        return line_count
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return 0
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return 0
if __name__ == "__main__":
    # 请替换为实际的jsonl文件路径
    # file_path = Path("~/relbench-reproduce/rel-data/rel-trial/all_clinical_studies.jsonl").expanduser()
    # 读取jsonl文件并输出行数
    # count_jsonl_lines(file_path)
    filepath =  "~/.cache/relbench/rel-f1/tasks/driver-dnf/train.parquet"
    df = read_parquet_file(Path(filepath))
    if df is not None:
        print(df.head())  # 打印前几行数据
    else:
        print("未能读取数据。")
