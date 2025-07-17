import pandas as pd
from pathlib import Path
import orjson  # 使用 orjson 替代内置 json
import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

# 忽略Pandas在处理空切片时可能产生的性能警告
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def prepare_data(db_directory: Path):
    """
    加载、清理、规范化数据。
    """
    tables = ['users', 'posts', 'comments', 'votes', 'badges', 'postHistory', 'postLinks']
    data = {}
    
    print("1/3: 正在加载数据文件...")
    try:
        table_name_map = {'postHistory': 'postHistory.parquet', 'postLinks': 'postLinks.parquet'}
        for table in tables:
            path = db_directory / table_name_map.get(table, f"{table}.parquet")
            if not path.exists(): raise FileNotFoundError(f"{path} 文件未找到！")
            data[table] = pd.read_parquet(path)
    except (FileNotFoundError, Exception) as e:
        print(f"数据加载时发生错误: {e}"); return None

    print("2/3: 正在重命名和清理列...")
    data['users'].rename(columns={'Id': 'UserId'}, inplace=True)
    data['posts'].rename(columns={'Id': 'PostId', 'OwnerUserId': 'UserId'}, inplace=True)
    data['comments'].rename(columns={'Id': 'CommentId', 'UserId': 'CommenterUserId'}, inplace=True)
    data['votes'].rename(columns={'Id': 'VoteId', 'UserId': 'VoterUserId'}, inplace=True)
    data['badges'].rename(columns={'Id': 'BadgeId'}, inplace=True)
    data['postHistory'].rename(columns={'Id': 'HistoryId', 'UserId': 'EditorUserId'}, inplace=True) 
    data['postLinks'].rename(columns={'Id': 'LinkId'}, inplace=True)

    print("3/3: 正在统一ID和时间戳格式...")
    id_cols = {
        'users': ['UserId'], 'posts': ['PostId', 'UserId'],
        'comments': ['CommentId', 'PostId', 'CommenterUserId'],
        'votes': ['VoteId', 'PostId', 'VoterUserId'], 'badges': ['BadgeId', 'UserId'],
        'postHistory': ['HistoryId', 'PostId', 'EditorUserId'], 'postLinks': ['LinkId', 'PostId', 'RelatedPostId']
    }
    for table, cols in id_cols.items():
        for col in cols:
            if col in data[table].columns:
                data[table][col] = pd.to_numeric(data[table][col], errors='coerce').astype('Int64')

    for table in data.values():
        for col in table.select_dtypes(include=['datetime64[ns]']).columns:
            table[col] = table[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    print("数据准备完成。")
    return data

def build_trees_in_parallel(prepared_data):
    """
    使用向量化和并行处理来构建所有用户树。
    """
    print("\n--- 开始向量化聚合 ---")

    print("步骤 1/3: 聚合所有帖子的关联信息...")
    
    def aggregate_to_list(df):
        return df.to_dict('records')

    # 【已修正】: 使用传入的 prepared_data 变量，而不是不存在的 data
    post_children_tables = {
        'comments': ('PostId', prepared_data['comments']),
        'votes': ('PostId', prepared_data['votes']),
        'post_history': ('PostId', prepared_data['postHistory']),
        'post_links': ('PostId', prepared_data['postLinks'])
    }
    
    posts_df = prepared_data['posts'].copy()
    
    for name, (key, df) in tqdm(post_children_tables.items(), desc="  - 聚合帖子子项"):
        if not df.empty and key in df.columns:
            df.dropna(subset=[key], inplace=True)
            aggregated = df.groupby(key).apply(aggregate_to_list)
            aggregated.name = name
            posts_df = posts_df.merge(aggregated, on=key, how='left')

    print("步骤 2/3: 聚合所有用户的关联信息...")
    users_df = prepared_data['users'].copy()

    posts_aggregated = posts_df.dropna(subset=['UserId']).groupby('UserId').apply(aggregate_to_list)
    posts_aggregated.name = 'posts'
    users_df = users_df.merge(posts_aggregated, on='UserId', how='left')

    # 【已修正】: 使用传入的 prepared_data 变量，而不是不存在的 data
    user_children_tables = {
        'badges': ('UserId', prepared_data['badges']),
        'user_comments': ('CommenterUserId', prepared_data['comments']),
        'user_votes': ('VoterUserId', prepared_data['votes']),
    }

    for name, (key, df) in tqdm(user_children_tables.items(), desc="  - 聚合用户子项"):
         if not df.empty and key in df.columns:
            # 修正：在 merge 前，将 key 重命名为 'UserId' 以便统一合并
            df_agg = df.dropna(subset=[key])
            aggregated = df_agg.groupby(key).apply(aggregate_to_list)
            aggregated.name = name
            aggregated_df = aggregated.reset_index().rename(columns={key: 'UserId'})
            users_df = users_df.merge(aggregated_df, on='UserId', how='left')

    # 填充NaN为None或空列表，以便 orjson 处理
    for col in users_df.columns:
        if users_df[col].dtype == 'object' and any(isinstance(i, list) or isinstance(i, np.ndarray) for i in users_df[col].dropna()):
            users_df[col] = users_df[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [])

    print("步骤 3/3: 并行生成JSONL文件...")
    # 重置索引，以便将UserId作为普通列包含在字典中
    users_df.reset_index(inplace=True)
    return users_df.to_dict('records')


def worker_serializer(records):
    """
    多进程worker：将字典记录列表序列化为JSON字符串列表。
    """
    return [orjson.dumps(rec).decode('utf-8') for rec in records]

def save_all_trees_to_jsonl(user_records, output_path: Path):
    """
    使用多进程池将记录列表写入JSONL文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_processes = max(1, cpu_count() - 1) # 留一个核心给系统
    chunk_size = len(user_records) // num_processes
    if chunk_size == 0: chunk_size = 1

    print(f"使用 {num_processes} 个CPU核心并行写入到: {output_path}")

    chunks = [user_records[i:i + chunk_size] for i in range(0, len(user_records), chunk_size)]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with Pool(processes=num_processes) as pool:
            for json_lines in tqdm(pool.imap(worker_serializer, chunks), total=len(chunks), desc="  - 写入文件"):
                if json_lines:
                    f.write('\n'.join(json_lines) + '\n')

    print(f"\n文件保存成功！共 {len(user_records)} 位用户的数据已保存。")


if __name__ == "__main__":
    db_path = Path.home() / ".cache/relbench/rel-stack/db"
    
    prepared_data = prepare_data(db_path)

    if prepared_data:
        all_user_records = build_trees_in_parallel(prepared_data)
        
        output_file = Path.home() / "rel-data/rel-stack/all_user_trees_final.jsonl"
        save_all_trees_to_jsonl(all_user_records, output_file)
    else:
        print("未能加载或准备数据，程序退出。")