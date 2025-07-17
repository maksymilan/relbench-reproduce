import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any
from pandas import Timestamp
import numpy as np

def make_json_serializable(obj):
    """
    递归地遍历一个对象，将所有非JSON序列化的pandas/numpy类型转换为兼容格式。
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
    加载所有Avito数据，并进行预处理和分组，为高效建树做准备。
    """
    # 使用您提供的确切文件名
    table_names = ["UserInfo", "AdsInfo", "Category", "Location", "SearchInfo", 
                   "SearchStream", "VisitStream", "PhoneRequestsStream"]
    data = {}
    print("--- 正在加载数据文件 ---")
    try:
        for name in table_names:
            data[name] = pd.read_parquet(db_directory / f"{name}.parquet")
            print(f"- 已加载 {name}.parquet")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 {e.filename}。")
        print(f"请确认您的数据目录路径正确: {db_directory}")
        return {}

    print("\n--- 正在预处理和分组数据以提高效率 ---")
    
    prepared_data = {
        # 核心实体的查找字典
        'users_dict': data['UserInfo'].set_index('UserID').to_dict('index'),
        'ads_dict': data['AdsInfo'].set_index('AdID').to_dict('index'),
        'locations_dict': data['Location'].set_index('LocationID').to_dict('index'),
        'categories_dict': data['Category'].set_index('CategoryID').to_dict('index'),
        
        # 按不同键进行分组，用于快速提取关联信息
        'searches_by_user': data['SearchInfo'].groupby('UserID'),
        'search_results_by_search': data['SearchStream'].groupby('SearchID'),
        'visits_by_user': data['VisitStream'].groupby('UserID'),
        'phone_requests_by_user': data['PhoneRequestsStream'].groupby('UserID'),
    }

    print("--- 数据准备就绪 ---")
    return prepared_data

def build_single_user_tree(user_id: int, prepared_data: Dict[str, Any]) -> Dict:
    """
    根据一个用户ID和预处理好的数据，构建一棵完整的、深度嵌套的信息树。
    """
    user_info = prepared_data['users_dict'].get(user_id, {})
    user_node = {"UserID": user_id, **user_info}
    
    # 1. 构建搜索历史 (Search History)
    search_history_list = []
    try:
        user_searches_df = prepared_data['searches_by_user'].get_group(user_id)
        for _, search_row in user_searches_df.iterrows():
            search_id = search_row['SearchID']
            search_event_node = search_row.to_dict()
            
            # 丰富地点和品类信息
            search_event_node['search_location'] = prepared_data['locations_dict'].get(search_row['LocationID'], {})
            search_event_node['search_category'] = prepared_data['categories_dict'].get(search_row['CategoryID'], {})
            
            # 嵌套：获取该次搜索展示的所有结果
            results_shown_list = []
            try:
                search_results_df = prepared_data['search_results_by_search'].get_group(search_id)
                for _, result_row in search_results_df.iterrows():
                    ad_id = result_row['AdID']
                    result_node = result_row.to_dict()
                    # 嵌套广告的完整信息
                    result_node['ad_shown'] = build_ad_node(ad_id, prepared_data)
                    results_shown_list.append(result_node)
            except KeyError:
                pass # 本次搜索没有展示结果
            search_event_node['search_results_shown'] = results_shown_list
            search_history_list.append(search_event_node)
    except KeyError:
        pass # 该用户没有搜索历史
    user_node['search_history'] = search_history_list

    # 2. 构建广告浏览历史 (Ad Visits)
    visits_list = []
    try:
        user_visits_df = prepared_data['visits_by_user'].get_group(user_id)
        for _, visit_row in user_visits_df.iterrows():
            ad_id = visit_row['AdID']
            visit_node = visit_row.to_dict()
            visit_node['ad_viewed'] = build_ad_node(ad_id, prepared_data)
            visits_list.append(visit_node)
    except KeyError:
        pass
    user_node['ad_visits'] = visits_list

    # 3. 构建电话请求历史 (Phone Requests)
    requests_list = []
    try:
        user_requests_df = prepared_data['phone_requests_by_user'].get_group(user_id)
        for _, request_row in user_requests_df.iterrows():
            ad_id = request_row['AdID']
            request_node = request_row.to_dict()
            request_node['ad_requested'] = build_ad_node(ad_id, prepared_data)
            requests_list.append(request_node)
    except KeyError:
        pass
    user_node['phone_requests'] = requests_list

    return user_node

def build_ad_node(ad_id: int, prepared_data: Dict[str, Any]) -> Dict:
    """
    一个辅助函数，用于构建一个包含地点和品类详情的完整广告节点。
    """
    ad_info = prepared_data['ads_dict'].get(ad_id, {})
    if not ad_info:
        return {"AdID": ad_id}
        
    ad_node = {"AdID": ad_id, **ad_info}
    ad_node['ad_location'] = prepared_data['locations_dict'].get(ad_info.get('LocationID'), {})
    ad_node['ad_category'] = prepared_data['categories_dict'].get(ad_info.get('CategoryID'), {})
    return ad_node

def save_all_user_trees_to_jsonl(prepared_data: Dict[str, Any], output_path: Path):
    """
    遍历所有用户，为每人构建树，并以流式写入JSONL文件。
    """
    all_user_ids = sorted(prepared_data['users_dict'].keys())
    print(f"\n正在将所有 {len(all_user_ids)} 位用户的数据流式写入到 JSONL 文件: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id in all_user_ids:
            user_tree = build_single_user_tree(user_id, prepared_data)
            serializable_tree = make_json_serializable(user_tree)
            json_line = json.dumps(serializable_tree, ensure_ascii=False)
            f.write(json_line + '\n')
            count += 1
            if count % 1000 == 0:
                print(f"...已处理 {count} 位用户...")
    
    print(f"\n文件保存成功！共 {count} 位用户的数据已保存。")

if __name__ == "__main__":
    # 使用您提供的确切路径
    db_path = Path.home() / ".cache/relbench/rel-avito/db"
    
    prepared_data = load_and_prepare_data(db_path)

    if prepared_data:
        print("\n" + "="*50)
        print("方法一：构建单棵用户树进行验证")
        print("="*50)
        
        sample_user_id = next(iter(prepared_data['users_dict']))
        print(f"正在为示例用户 (UserID: {sample_user_id}) 构建信息树...")
        
        single_tree = build_single_user_tree(sample_user_id, prepared_data)
        serializable_tree = make_json_serializable(single_tree)
        output_file = Path.home() / "relbench-reproduce/rel-data/rel-avito/avito_user_sample.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_tree, f, ensure_ascii=False, indent=2)
        print("单棵用户树构建成功！结构如下（长列表将被截断）：")
        display_tree = json.loads(json.dumps(serializable_tree))
        for key in ['search_history', 'ad_visits', 'phone_requests']:
            if len(display_tree.get(key, [])) > 1:
                display_tree[key] = display_tree[key][:1] + [f"... and {len(display_tree[key])-1} more ..."]
        print(json.dumps(display_tree, indent=2))

        print("\n" + "="*50)
        print("方法二：将所有用户的完整数据保存到 JSONL 文件")
        print("="*50)
        
        output_file = Path.home() / "avito_users_complete.jsonl"
        save_all_user_trees_to_jsonl(prepared_data, output_file)
        
    else:
        print("\n未能加载数据，程序退出。")