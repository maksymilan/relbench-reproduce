# 读取jsonl指定某一行的数据，并保存为json文件
import json
from pathlib import Path
from typing import Dict, Any
def read_jsonl_line(file_path: Path, line_num: int) -> None:
    """
    读取 JSONL 文件中的一行，并将其解析为 JSON 对象。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_num:
                out_putfile = "example.json"
                with open(out_putfile, 'w', encoding='utf-8') as output_file:
                    try:
                        json_object = json.loads(line)
                        json.dump(json_object, output_file, ensure_ascii=False, indent=2)
                        print(f"已将 JSONL 行写入 {out_putfile}")
                    except json.JSONDecodeError as e:
                        print(f"解析 JSONL 行时出错: {e}")
                return

read_jsonl_line("./all_f1_drivers.jsonl", 11)