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

# 从原jsonl文件读取指定数量的行存入新的jsonl文件
def copy_jsonl_lines(input_file: Path, output_file: Path, num_lines: int) -> None:
    """
    从 JSONL 文件中读取指定数量的行，并将其写入新的 JSONL 文件。
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= num_lines:
                break
            outfile.write(line)
    print(f"已将前 {num_lines} 行写入 {output_file}")

if __name__ == "__main__":
    copy_jsonl_lines(Path("all_avito_users.jsonl"), Path("mini_avito_users.jsonl"), 8)