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
    加载所有Event数据，并进行预处理、连接和分组，为高效建树做准备。
    """
    tables_to_load = ["users", "user_friends", "events", "event_attendees", "event_interest"]
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
    
    # --- NEW: 聚合c_1到c_100特征列 ---
    events_df = data['events']
    # 定义c_1到c_100的列名
    feature_cols = [f'c_{i}' for i in range(1, 101)]
    # 确保所有特征列都存在
    existing_feature_cols = [col for col in feature_cols if col in events_df.columns]
    # 将这些列的值转换为一个列表，存入新列'c_vector'
    events_df['c_vector'] = events_df[existing_feature_cols].values.tolist()
    # 删除原始的100个特征列
    events_df.drop(columns=existing_feature_cols, inplace=True)
    print("- 已聚合 events 表中的 c_1 到 c_100 特征为 c_vector。")
    # --- 结束新逻辑 ---

    prepared_data = {
        'users_dict': data['users'].set_index('user_id').to_dict('index'),
        'events_dict': events_df.set_index('event_id').to_dict('index'), # 使用修改后的events_df
        
        'friends_by_user': data['user_friends'].groupby('user'),
        'events_by_creator': events_df.groupby('user_id'), # 使用修改后的events_df
        'attendees_by_user': data['event_attendees'].groupby('user_id'),
        'attendees_by_event': data['event_attendees'].groupby('event'),
        'interest_by_user': data['event_interest'].groupby('user'),
        'interest_by_event': data['event_interest'].groupby('event'),
    }

    print("--- 数据准备就绪 ---")
    return prepared_data

def build_single_user_tree(user_id: int, prepared_data: Dict[str, Any]) -> Dict:
    """
    根据一个用户ID和预处理好的数据，构建一棵完整的、深度嵌套的信息树。
    """
    user_info = prepared_data['users_dict'].get(user_id, {})
    user_node = {"user_id": user_id, **user_info}
    
    # --- FIX: 过滤掉friends列表中的null值 ---
    try:
        friends_series = prepared_data['friends_by_user'].get_group(user_id)['friend']
        # 使用 .dropna() 移除所有NaN/None值
        user_node['friends'] = friends_series.dropna().tolist()
    except KeyError:
        user_node['friends'] = []

    events_created_list = []
    try:
        created_events_df = prepared_data['events_by_creator'].get_group(user_id)
        for _, event_row in created_events_df.iterrows():
            event_id = event_row['event_id']
            # event_row.to_dict()现在会自动包含c_vector字段
            event_node = event_row.to_dict()
            
            try:
                event_node['attendees'] = prepared_data['attendees_by_event'].get_group(event_id).to_dict('records')
            except KeyError:
                event_node['attendees'] = []
            
            try:
                event_node['interest_shown'] = prepared_data['interest_by_event'].get_group(event_id).to_dict('records')
            except KeyError:
                event_node['interest_shown'] = []
            
            events_created_list.append(event_node)
    except KeyError:
        pass
    user_node['events_created'] = events_created_list

    attended_event_ids = set(prepared_data['attendees_by_user'].get_group(user_id)['event'].dropna()) if user_id in prepared_data['attendees_by_user'].groups else set()
    interested_event_ids = set(prepared_data['interest_by_user'].get_group(user_id)['event'].dropna()) if user_id in prepared_data['interest_by_user'].groups else set()
    all_engaged_event_ids = attended_event_ids.union(interested_event_ids)
    
    created_event_ids = set(event['event_id'] for event in events_created_list)
    engaged_event_ids_other = all_engaged_event_ids - created_event_ids
    
    engagements_list = []
    for event_id in engaged_event_ids_other:
        engagement_details = {}
        try:
            attendee_record = prepared_data['attendees_by_event'].get_group(event_id)
            user_attendee_record = attendee_record[attendee_record['user_id'] == user_id]
            if not user_attendee_record.empty:
                engagement_details.update(user_attendee_record.iloc[0].to_dict())
        except KeyError:
            pass
        try:
            interest_record = prepared_data['interest_by_event'].get_group(event_id)
            user_interest_record = interest_record[interest_record['user'] == user_id]
            if not user_interest_record.empty:
                engagement_details.update(user_interest_record.iloc[0].to_dict())
        except KeyError:
            pass

        engagements_list.append({
            "user_engagement_details": engagement_details,
            "event": prepared_data['events_dict'].get(event_id, {})
        })
    user_node['event_engagements'] = engagements_list

    return user_node

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
    db_path = Path.home() / ".cache/relbench/rel-event/db"
    
    prepared_data = load_and_prepare_data(db_path)

    if prepared_data:
        print("\n" + "="*50)
        print("方法一：构建单棵用户树进行验证")
        print("="*50)
        
        sample_user_id = next(iter(prepared_data['users_dict']))
        print(f"正在为示例用户 (user_id: {sample_user_id}) 构建信息树...")
        
        single_tree = build_single_user_tree(sample_user_id, prepared_data)
        serializable_tree = make_json_serializable(single_tree)
        output_file = Path.home() / "relbench-reproduce/rel-data/rel-event/event_user_sample.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_tree, f, ensure_ascii=False, indent=2)
        print("单棵用户树构建成功！结构如下（c_1-c_100已合并为c_vector）：")
        display_tree = json.loads(json.dumps(serializable_tree))
        for key in ['friends', 'events_created', 'event_engagements']:
            if len(display_tree.get(key, [])) > 1:
                display_tree[key] = display_tree[key][:1] + [f"... and {len(display_tree[key])-1} more ..."]
        print(json.dumps(display_tree, indent=2))

        print("\n" + "="*50)
        print("方法二：将所有用户的完整数据保存到 JSONL 文件")
        print("="*50)
        
        output_file = Path.home() / "event_users_complete_vectorized.jsonl"
        save_all_user_trees_to_jsonl(prepared_data, output_file)
        
    else:
        print("\n未能加载数据，程序退出。")