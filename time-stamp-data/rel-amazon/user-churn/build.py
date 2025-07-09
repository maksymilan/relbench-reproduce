import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

def load_and_merge_data(db_directory: Path):
    """
    加载所有Parquet文件并根据外键关系将它们合并成一个大的DataFrame。
    此函数与您提供的版本基本相同，但增加了对时间列的显式转换。
    """
    customer_path = db_directory / "customer.parquet"
    review_path = db_directory / "review.parquet"
    product_path = db_directory / "product.parquet"

    if not all([customer_path.exists(), review_path.exists(), product_path.exists()]):
        print(f"错误：在 {db_directory} 中数据文件缺失。")
        return None

    try:
        print("正在加载数据文件...")
        df_customer = pd.read_parquet(customer_path)
        df_review = pd.read_parquet(review_path)
        df_product = pd.read_parquet(product_path)
        
        print("正在合并数据...")
        merged_df = pd.merge(df_review, df_product, on='product_id', how='left')
        full_data = pd.merge(merged_df, df_customer, on='customer_id', how='left')
        
        # --- 关键改动: 确保时间列是datetime类型，以便进行比较 ---
        full_data['review_time'] = pd.to_datetime(full_data['review_time'])
        
        print("数据准备完成。")
        return full_data
    except Exception as e:
        print(f"数据加载或合并时发生错误: {e}")
        return None

def build_tree_for_task_row(task_row: pd.Series, full_data: pd.DataFrame):
    """
    为任务表中的单行数据构建一个时间感知的、带标签的树。

    :param task_row: 任务表中的一行，应包含 'customer_id', 'timestamp', 和标签列。
    :param full_data: 包含所有历史信息的完整DataFrame。
    :return: 一个代表该任务样本的嵌套字典，或在没有历史数据时返回一个基本节点。
    """
    customer_id = task_row['customer_id']
    cutoff_time = task_row['timestamp']
    label = task_row['churn']

    # --- 核心逻辑: 1. 时间过滤 ---
    # 只选择任务时间点之前的数据
    historical_data = full_data[full_data['review_time'] < cutoff_time]

    # --- 核心逻辑: 2. 实体过滤 ---
    # 从历史数据中筛选出当前客户的数据
    customer_historical_data = historical_data[historical_data['customer_id'] == customer_id].copy()

    # 获取客户基本信息 (即使没有评论历史，也应该有客户信息)
    customer_info_rows = full_data[full_data['customer_id'] == customer_id]
    if customer_info_rows.empty:
        return None # 如果在整个数据库中都找不到这个客户，则跳过
    customer_name = customer_info_rows.iloc[0]['customer_name']

    # 构建根节点，并加入标签
    customer_node = {
        'customer_id': int(customer_id),
        'customer_name': str(customer_name),
        'timestamp': str(cutoff_time), # 加入时间戳信息
        'label': int(label), # 加入标签 (ground truth)
        'reviews': [] # 即使没有历史评论，也保留此字段
    }

    if customer_historical_data.empty:
        # 如果该客户在截止时间前没有任何评论，返回只包含基本信息和标签的树
        return customer_node

    # 按时间排序客户的评论历史
    customer_historical_data.sort_values(by='review_time', ascending=False, inplace=True)

    # 遍历该客户的历史评论，构建子节点
    for _, review_data in customer_historical_data.iterrows():
        product_node = {
            'product_id': int(review_data['product_id']),
            'title': str(review_data['title']),
            'brand': str(review_data['brand']),
            'price': float(review_data['price']) if pd.notna(review_data['price']) else None,
            'category': str(review_data['category']),
            'description': str(review_data['description'])
        }
        
        review_node = {
            'review_text': str(review_data['review_text']),
            'summary': str(review_data['summary']),
            'review_time': str(review_data['review_time']),
            'rating': float(review_data['rating']),
            'verified': str(review_data['verified']),
            'product': product_node
        }
        customer_node['reviews'].append(review_node)

    return customer_node

def generate_dataset_from_task_table(
    full_data: pd.DataFrame, 
    task_table_path: Path, 
    output_path: Path
):
    """
    读取一个任务表，为其中每一行生成数据树，并保存到JSONL文件中。
    """
    if not task_table_path.exists():
        print(f"任务文件未找到: {task_table_path}")
        return

    print(f"正在处理任务文件: {task_table_path.name}")
    task_df = pd.read_parquet(task_table_path)
    
    # 确保任务表中的时间戳也是datetime类型
    task_df['timestamp'] = pd.to_datetime(task_df['timestamp'])
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在生成树并流式写入到: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用tqdm显示处理进度
        for _, task_row in tqdm(task_df.iterrows(), total=task_df.shape[0]):
            customer_tree = build_tree_for_task_row(task_row, full_data)
            if customer_tree:
                json_line = json.dumps(customer_tree, ensure_ascii=False)
                f.write(json_line + '\n')
                count += 1
    
    print(f"文件保存成功！共 {count} 个任务样本已保存到文件中。\n")

if __name__ == "__main__":
    # --- 配置路径 ---
    # 数据库文件所在目录
    db_path = Path.home() / ".cache/relbench/rel-amazon/db"
    
    # 任务表所在目录
    task_path = Path.home() / ".cache/relbench/rel-amazon/tasks/user-churn"
    
    # 输出目录
    output_dir = Path.home() / "rel-data/rel-amazon-user-churn"

    # --- 主流程 ---
    # 1. 加载并合并一次所有数据
    full_merged_data = load_and_merge_data(db_path)

    if full_merged_data is not None and not full_merged_data.empty:
        # 2. 遍历不同的数据集（train, val, test）
        for split in ['train', 'val', 'test']:
            task_file = task_path / f"{split}.parquet"
            output_file = output_dir / f"{split}_trees.jsonl"
            
            generate_dataset_from_task_table(
                full_data=full_merged_data,
                task_table_path=task_file,
                output_path=output_file
            )
    else:
        print("未能加载或处理数据，程序退出。")