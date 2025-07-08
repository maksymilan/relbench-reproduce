import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any
from pandas import Timestamp
import numpy as np

def make_json_serializable(obj):
    """
    递归地遍历一个对象，将所有非JSON序列化的pandas/numpy类型转换为兼容格式。
    - numpy整数/浮点数 -> python int/float
    - pandas Timestamp -> 'YYYY-MM-DD HH:MM:SS' 格式的字符串
    - NaN (及其他缺失值) -> None
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(elem) for elem in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if pd.isna(obj):
        return None
    return obj

def load_and_prepare_data(db_directory: Path) -> Dict[str, Any]:
    """
    加载H&M数据，并进行预处理和分组，为高效建树做准备。
    """
    tables_to_load = ["customer", "article", "transactions"]
    data = {}
    print("--- 正在加载数据文件 ---")
    try:
        for name in tables_to_load:
            data[name] = pd.read_parquet(db_directory / f"{name}.parquet")
            print(f"- 已加载 {name}.parquet")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 {e.filename}。")
        print(f"请确认您的数据目录路径正确: {db_directory}")
        return {}

    print("\n--- 正在预处理和分组数据以提高效率 ---")
    
    prepared_data = {
        # 将customer和article转为字典，用于O(1)快速查找
        'customer_dict': data['customer'].set_index('customer_id').to_dict('index'),
        'article_dict': data['article'].set_index('article_id').to_dict('index'),
        # 将巨大的transactions表按customer_id分组
        'transactions_by_customer': data['transactions'].groupby('customer_id')
    }

    print("--- 数据准备就绪 ---")
    return prepared_data

def build_single_customer_tree(customer_id: str, prepared_data: Dict[str, Any]) -> Dict:
    """
    根据一个customer_id和预处理好的数据，构建一棵完整的用户信息树。
    """
    # 1. 初始化根节点 - 用户的静态信息
    customer_info = prepared_data['customer_dict'].get(customer_id, {})
    customer_node = {"customer_id": customer_id, **customer_info}
    
    purchase_history = []
    
    try:
        # 2. 从分组中一次性获取该用户的所有交易记录
        customer_transactions_df = prepared_data['transactions_by_customer'].get_group(customer_id)
        
        # 3. 遍历交易记录，构建购买历史
        for _, transaction_row in customer_transactions_df.iterrows():
            article_id = transaction_row['article_id']
            
            # 构建交易节点
            transaction_node = {
                "transaction_date": transaction_row['t_dat'],
                "price": transaction_row['price'],
                "sales_channel_id": transaction_row['sales_channel_id'],
                # 嵌套商品信息
                "article_purchased": prepared_data['article_dict'].get(article_id, {})
            }
            purchase_history.append(transaction_node)

    except KeyError:
        # 如果用户在transactions表中没有记录，则get_group会报错
        # 在这种情况下，purchase_history将保持为空列表，是正确的行为
        pass
        
    customer_node['purchase_history'] = purchase_history
    return customer_node

def save_all_customer_trees_to_jsonl(prepared_data: Dict[str, Any], output_path: Path):
    """
    遍历所有用户，为每人构建树，并以流式写入JSONL文件。
    """
    all_customer_ids = sorted(prepared_data['customer_dict'].keys())
    print(f"\n正在将所有 {len(all_customer_ids)} 位用户的数据流式写入到 JSONL 文件: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for customer_id in all_customer_ids:
            customer_tree = build_single_customer_tree(customer_id, prepared_data)
            # 在写入前，清理所有非标准JSON类型
            serializable_tree = make_json_serializable(customer_tree)
            json_line = json.dumps(serializable_tree, ensure_ascii=False)
            f.write(json_line + '\n')
            count += 1
            if count % 10000 == 0: # 每处理10000个用户，打印一次进度
                print(f"...已处理 {count} 位用户...")
    
    print(f"\n文件保存成功！共 {count} 位用户的数据已保存。")

if __name__ == "__main__":
    # !!! 请将此路径修改为您存放H&M数据的实际路径 !!!
    db_path = Path.home() / ".cache/relbench/rel-hm/db/"
    
    prepared_data = load_and_prepare_data(db_path)

    if prepared_data:
        print("\n" + "="*50)
        print("方法一：构建单棵用户树进行验证")
        print("="*50)
        
        # 从字典中随机选择一个用户ID进行测试
        sample_customer_id = next(iter(prepared_data['customer_dict']))
        print(f"正在为示例用户 (customer_id: {sample_customer_id}) 构建信息树...")
        
        single_tree = build_single_customer_tree(sample_customer_id, prepared_data)
        serializable_tree = make_json_serializable(single_tree)
        
        print("单棵用户树构建成功！结构如下（购买历史只显示一条用于示例）：")
        if serializable_tree and serializable_tree.get('purchase_history'):
             # 复制一份以避免修改原始数据
            display_tree = json.loads(json.dumps(serializable_tree))
            # 只保留第一条购买记录以简化打印输出
            output_file = Path.home() / "relbench-reproduce/rel-data/rel-hm/hm_customer_sample.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(display_tree, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(serializable_tree, indent=2))

        print("\n" + "="*50)
        print("方法二：将所有用户的完整数据保存到 JSONL 文件")
        print("="*50)

        output_file = Path.home() / "relbench-reproduce/rel-data/rel-hm/all_hm_customers.jsonl"
        save_all_customer_trees_to_jsonl(prepared_data, output_file)
        
    else:
        print("\n未能加载数据，程序退出。")