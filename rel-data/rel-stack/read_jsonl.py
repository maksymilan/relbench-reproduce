# 读取all_user_trees_final.jsonl的一行，并写入example.json
import json
from pathlib import Path

input_file = Path.home() / "rel-data/rel-stack/all_user_trees_final.jsonl"
output_file = Path.home() / "rel-data/rel-stack/example.json"

with open(input_file, 'r', encoding='utf-8') as f:
    line = f.readline()
    if line:
        data = json.loads(line)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=4)
