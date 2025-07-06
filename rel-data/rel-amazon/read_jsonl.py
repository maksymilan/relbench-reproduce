# 读取一个jsonl文件的第一行，并写入一个json文件中

import json
from pathlib import Path
def read_first_line_and_write_jsonl(input_file: Path, output_file: Path):
    """
    读取一个jsonl文件的第一行，并将其写入一个json文件中。
    """
    if not input_file.exists():
        print(f"错误：输入文件 {input_file} 不存在。")
        return

    try:
        with input_file.open('r', encoding='utf-8') as infile:
            first_line = infile.readline().strip()
            if not first_line:
                print("输入文件为空。")
                return
            
            data = json.loads(first_line)
            
            with output_file.open('w', encoding='utf-8') as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
                
            print(f"第一行数据已成功写入 {output_file}。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

read_first_line_and_write_jsonl(Path('all_customer_trees.jsonl'),
                                  Path('example.json'))
