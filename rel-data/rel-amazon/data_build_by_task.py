import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import os
import warnings

# --- 工作进程初始化函数 ---
worker_full_data = None
def init_worker(db_path_str: str):
    """
    每个工作进程的初始化函数。它加载数据并将其存储在进程的全局变量中。
    """
    global worker_full_data
    print(f"进程 {os.getpid()} 正在加载和准备数据...")
    db_directory = Path(db_path_str)
    
    customer_path = db_directory / "customer.parquet"
    review_path = db_directory / "review.parquet"
    product_path = db_directory / "product.parquet"

    df_customer = pd.read_parquet(customer_path)
    df_review = pd.read_parquet(review_path)
    df_product = pd.read_parquet(product_path)
    
    merged_df = pd.merge(df_review, df_product, on='product_id', how='left')
    full_data = pd.merge(merged_df, df_customer, on='customer_id', how='left')
    
    full_data['review_time'] = pd.to_datetime(full_data['review_time'])
    
    full_data.sort_values(['customer_id', 'review_time'], inplace=True)
    full_data.set_index(['customer_id', 'review_time'], inplace=True)
    
    worker_full_data = full_data
    print(f"进程 {os.getpid()} 数据准备完毕。")


# --- 针对新模式优化的工作函数 ---
def process_task_row_isolated(task_row: dict, task_config: dict):
    """
    工作函数现在直接使用它自己所在进程的全局变量 `worker_full_data`。
    """
    global worker_full_data

    entity_id_col = task_config['entity_id_col']
    label_col = task_config['label_col']
    
    customer_id = task_row[entity_id_col]
    cutoff_time = task_row['timestamp']
    label_value = task_row[label_col]

    try:
        customer_historical_data = worker_full_data.loc[customer_id].loc[:cutoff_time]
        customer_name = customer_historical_data.iloc[0]['customer_name']
    except (KeyError, IndexError):
        try:
            customer_name = worker_full_data.xs(customer_id, level='customer_id').iloc[0]['customer_name']
        except (KeyError, IndexError):
            return None
        customer_historical_data = pd.DataFrame()

    # --- 这里是唯一的修改点 ---
    # 使用了新的名称 NumpyExtensionArray 来避免 FutureWarning
    if isinstance(label_value, (list, pd.Series, pd.arrays.NumpyExtensionArray)):
         processed_label = list(label_value)
    else:
        processed_label = label_value

    customer_node = {
        'customer_id': int(customer_id),
        'customer_name': str(customer_name),
        'timestamp': str(cutoff_time),
        'label': processed_label,
        'reviews': []
    }

    if customer_historical_data.empty:
        return customer_node

    for _, review_data in customer_historical_data.reset_index().iterrows():
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


def generate_dataset_master(
    task_table_path: Path, 
    output_path: Path,
    task_config: dict,
    num_workers: int,
    db_path_str: str
):
    """
    主控制函数，负责创建进程池并分发任务。
    """
    if not task_table_path.exists():
        print(f"任务文件未找到: {task_table_path}")
        return

    print(f"正在处理任务: '{task_config['name']}', 文件: {task_table_path.name} (使用 {num_workers} 个CPU核心)")
    task_df = pd.read_parquet(task_table_path)
    task_df['timestamp'] = pd.to_datetime(task_df['timestamp'])
    tasks = task_df.to_dict('records')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    worker_func = partial(process_task_row_isolated, task_config=task_config)

    with open(output_path, 'w', encoding='utf-8') as f:
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(db_path_str,)) as pool:
            results_iterator = pool.imap_unordered(worker_func, tasks)
            for result_tree in tqdm(results_iterator, total=len(tasks), desc=f"Writing to {output_path.name}"):
                if result_tree:
                    json_line = json.dumps(result_tree, ensure_ascii=False)
                    f.write(json_line + '\n')
    
    print(f"文件保存成功: {output_path}\n")

if __name__ == "__main__":
    NUM_CPU_CORES = 5
    available_cores = os.cpu_count() or 1
    if NUM_CPU_CORES > available_cores:
        NUM_CPU_CORES = available_cores

    TASKS_TO_PROCESS = [
        {"name": "user-churn", "entity_id_col": "customer_id", "label_col": "churn"},
        {"name": "user-ltv", "entity_id_col": "customer_id", "label_col": "ltv"},
        {"name": "user-item-purchase", "entity_id_col": "customer_id", "label_col": "product_id"},
        {"name": "user-item-rate", "entity_id_col": "customer_id", "label_col": "product_id"},
        {"name": "user-item-review", "entity_id_col": "customer_id", "label_col": "product_id"}
    ]

    db_base_path = Path.home() / ".cache/relbench/rel-amazon"
    db_path = db_base_path / "db"
    tasks_base_path = db_base_path / "tasks"
    output_base_dir = Path.home() / "rel-data/rel-amazon"

    for task_config in TASKS_TO_PROCESS:
        task_name = task_config['name']
        task_path = tasks_base_path / task_name
        output_dir = output_base_dir / f"{task_name}-initializer-pattern"
        
        print(f"========== 开始处理任务: {task_name} ==========")
        
        for split in ['train', 'val', 'test']:
            task_file = task_path / f"{split}.parquet"
            output_file = output_dir / f"{split}_trees.jsonl"
            
            generate_dataset_master(
                task_table_path=task_file,
                output_path=output_file,
                task_config=task_config,
                num_workers=NUM_CPU_CORES,
                db_path_str=str(db_path)
            )
    print("所有任务处理完毕！")