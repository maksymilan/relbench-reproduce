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
    filepath1 =  "~/.cache/relbench/rel-amazon/tasks/user-churn/train.parquet"
    filepath2 = "~/.cache/relbench/rel-amazon/tasks/item-churn/train.parquet"
    filepath3 =  "~/.cache/relbench/rel-avito/tasks/user-clicks/train.parquet"
    filepath4 = "~/.cache/relbench/rel-avito/tasks/user-visits/train.parquet"
    filepath5 = "~/.cache/relbench/rel-event/tasks/user-repeat/train.parquet"
    filepath6 = "~/.cache/relbench/rel-event/tasks/user-ignore/train.parquet"
    filepath7 = "~/.cache/relbench/rel-f1/tasks/driver-dnf/train.parquet"
    filepath8 = "~/.cache/relbench/rel-f1/tasks/driver-top3/train.parquet"
    filepath9 = "~/.cache/relbench/rel-hm/tasks/user-churn/train.parquet"
    filepath10 = "~/.cache/relbench/rel-stack/tasks/user-engagement/train.parquet"
    filepath11 = "~/.cache/relbench/rel-stack/tasks/user-badge/train.parquet"
    filepath12 = "~/.cache/relbench/rel-trial/tasks/study-outcome/train.parquet"
    df1 = read_parquet_file(Path(filepath1))
    # df_val = read_parquet_file(Path(filepath1) / Path("val.parquet"))
    # df_test = read_parquet_file(Path(filepath1) / Path("test.parquet"))
    if df1 is not None:
        print(f"Path: {filepath1}")
        print(df1.head())  # 打印前几行数据
        print(f"Number of rows: {len(df1)}")  # 打印行数
        # print(f"val lines: {len(df_val)}")
        # print(f"test lines: {len(df_test)}")
    df2 = read_parquet_file(Path(filepath2))
    # df_val = read_parquet_file(Path(filepath2) / Path("val.parquet"))
    # df_test = read_parquet_file(Path(filepath2) / Path("test.parquet"))
    if df2 is not None:
        print(f"Path: {filepath2}")
        print(df2.head())
        print(f"Number of rows: {len(df2)}")
        # print(f"val lines: {len(df_val)}")
        # print(f"test lines: {len(df_test)}")
    # df3 = read_parquet_file(Path(filepath3))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-avito/tasks/user-visits/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-avito/tasks/user-visits/test.parquet"))
    # if df3 is not None:
    #     print(f"Path: {filepath3}")
    #     print(df3.head())
    #     print(f"Number of rows: {len(df3)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df4 = read_parquet_file(Path(filepath4))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-event/tasks/user-repeat/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-event/tasks/user-repeat/test.parquet"))
    # if df4 is not None:
    #     print(f"Path: {filepath4}")
    #     print(df4.head())
    #     print(f"Number of rows: {len(df4)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df5 = read_parquet_file(Path(filepath5))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-event/tasks/user-ignore/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-event/tasks/user-ignore/test.parquet"))

    # if df5 is not None:
    #     print(f"Path: {filepath5}")
    #     print(df5.head())
    #     print(f"Number of rows: {len(df5)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df6 = read_parquet_file(Path(filepath6))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-f1/tasks/driver-dnf/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-f1/tasks/driver-dnf/test.parquet"))
    # if df6 is not None:
    #     print(f"Path: {filepath6}")
    #     print(df6.head())
    #     print(f"Number of rows: {len(df6)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df7 = read_parquet_file(Path(filepath7))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-f1/tasks/driver-top3/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-f1/tasks/driver-top3/test.parquet"))
    # if df7 is not None:
    #     print(f"Path: {filepath7}")
    #     print(df7.head())
    #     print(f"Number of rows: {len(df7)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df8 = read_parquet_file(Path(filepath8))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-hm/tasks/user-churn/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-hm/tasks/user-churn/test.parquet"))
    # if df8 is not None:
    #     print(f"Path: {filepath8}")
    #     print(df8.head())
    #     print(f"Number of rows: {len(df8)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df9 = read_parquet_file(Path(filepath9))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-stack/tasks/user-engagement/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-stack/tasks/user-engagement/test.parquet"))
    # if df9 is not None:
    #     print(f"Path: {filepath9}")
    #     print(df9.head())
    #     print(f"Number of rows: {len(df9)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df10 = read_parquet_file(Path(filepath10))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-stack/tasks/user-badge/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-stack/tasks/user-badge/test.parquet"))
    # if df10 is not None:
    #     print(f"Path: {filepath10}")
    #     print(df10.head())
    #     print(f"Number of rows: {len(df10)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")
    # df11 = read_parquet_file(Path(filepath11))
    # df_val = read_parquet_file(Path("~/.cache/relbench/rel-trial/tasks/study-outcome/val.parquet"))
    # df_test = read_parquet_file(Path("~/.cache/relbench/rel-trial/tasks/study-outcome/test.parquet"))
    # if df11 is not None:
    #     print(f"Path: {filepath11}")
    #     print(df11.head())
    #     print(f"Number of rows: {len(df11)}")
    #     print(f"val lines: {len(df_val)}")
    #     print(f"test lines: {len(df_test)}")