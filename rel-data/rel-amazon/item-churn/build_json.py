import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import os
import warnings

# 忽略 pandas 的一些未来版本警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 配置区域 ---
# 1. 任务和采样配置
TASK_NAME = "rel-amazon/tasks/item-churn"
ENTITY_ID_COL = "product_id"  # 任务表中的ID列名
NUM_TRAIN_SAMPLES = 2000      # 从训练集中提取的行数
NUM_VAL_SAMPLES = 500         # 从验证集中提取的行数
NUM_TEST_SAMPLES = 500        # 从测试集中提取的行数

# 2. 并行处理配置
NUM_CPU_CORES = 10

# --- 路径配置 ---
try:
    base_dir = Path.home()
except Exception:
    base_dir = Path(os.path.expanduser("~"))

# 输入路径
db_base_path = base_dir / ".cache/relbench/"
db_path = db_base_path / "rel-amazon/db"
original_tasks_base_path = db_base_path 

# 输出路径 (为了清晰，输出目录会包含任务名和样本数)
output_base_dir = base_dir / f"relbench-reproduce/rel-data/sampling/{TASK_NAME}"
final_output_path = output_base_dir 

# 中间文件（采样后的任务表）的存储路径
sampled_task_path = final_output_path 

# --- 全局变量，用于工作进程 ---
worker_data = None

def init_worker(db_path_str: str):
    """
    工作进程初始化函数：加载并索引一次数据以供所有任务使用。
    """
    global worker_data
    if worker_data is not None:
        return

    print(f"进程 {os.getpid()} 正在加载和准备数据...")
    db_directory = Path(db_path_str)
    
    try:
        df_customer = pd.read_parquet(db_directory / "customer.parquet")
        df_review = pd.read_parquet(db_directory / "review.parquet")
        df_product = pd.read_parquet(db_directory / "product.parquet")
    except FileNotFoundError as e:
        print(f"致命错误：数据文件在 {db_directory} 未找到。进程将退出。错误: {e}")
        exit(1)

    merged_df = pd.merge(df_review, df_product, on='product_id', how='left')
    full_data = pd.merge(merged_df, df_customer, on='customer_id', how='left')
    full_data['review_time'] = pd.to_datetime(full_data['review_time'])
    
    # 为快速查找物品，以 product_id 和 review_time 建立索引
    full_data.sort_values([ENTITY_ID_COL, 'review_time'], inplace=True)
    full_data.set_index([ENTITY_ID_COL, 'review_time'], inplace=True)
    
    worker_data = full_data
    print(f"进程 {os.getpid()} 数据准备完毕。")

def subsample_and_extract_ids(
    original_task_path: Path, 
    sampled_output_path: Path,
    id_col: str,
    n_train: int, n_val: int, n_test: int
) -> list:
    """
    步骤1和2：从任务文件中采样并提取唯一的ID。
    """
    print("步骤 1: 正在从任务文件中提取指定数量的样本...")
    sampled_output_path.mkdir(parents=True, exist_ok=True)
    
    all_ids = []
    sample_config = {'train': n_train, 'val': n_val, 'test': n_test}
    
    for split, num_samples in sample_config.items():
        original_file = original_task_path / f"{split}.parquet"
        sampled_file = sampled_output_path / f"{split}.parquet"
        
        if not original_file.exists():
            print(f"警告：原始任务文件 {original_file} 不存在，跳过。")
            continue
            
        df = pd.read_parquet(original_file)
        sampled_df = df.head(num_samples)
        sampled_df.to_parquet(sampled_file)
        
        print(f"  - 已从 '{split}.parquet' 提取 {len(sampled_df)} 行并保存到 {sampled_file}")
        all_ids.extend(sampled_df[id_col].tolist())
        
    print("\n步骤 2: 正在对提取的ID进行去重...")
    unique_ids = sorted(list(set(all_ids)))
    print(f"完成。共找到 {len(unique_ids)} 个唯一的物品ID。")
    
    return unique_ids

def build_complete_item_tree(product_id: int):
    """
    步骤3：为单个product_id构建一个包含其完整历史的树。
    """
    global worker_data
    if worker_data is None:
        return None

    try:
        # 查找该物品的所有评论历史，不使用时间戳截断
        item_full_history = worker_data.loc[product_id]
        if item_full_history.empty:
            return None
        # 从第一条记录中获取静态的物品信息
        product_info = item_full_history.iloc[0]

    except KeyError:
        # 如果在整个数据集中都找不到该物品，则返回None
        print(f"警告: 进程 {os.getpid()} 找不到产品ID {product_id} 的任何评论历史。跳过。")
        return None
    
    # 构建物品根节点 (不包含 'label' 和 'timestamp')
    product_node = {
        'product_id': int(product_id),
        'title': str(product_info['title']),
        'brand': str(product_info['brand']),
        'price': float(product_info['price']) if pd.notna(product_info['price']) else None,
        'category': str(product_info['category']),
        'description': str(product_info['description']),
        'reviews': []
    }
    
    # 遍历所有历史记录并填充评论
    for _, review_data in item_full_history.reset_index().iterrows():
        customer_node = {
            'customer_id': int(review_data['customer_id']),
            'customer_name': str(review_data['customer_name'])
        }
        review_node = {
            'review_text': str(review_data['review_text']),
            'summary': str(review_data['summary']),
            'review_time': str(review_data['review_time']),
            'rating': float(review_data['rating']),
            'verified': str(review_data['verified']),
            'customer': customer_node
        }
        product_node['reviews'].append(review_node)
        
    return product_node

def generate_trees_from_ids(
    unique_ids: list,
    output_jsonl_path: Path,
    num_workers: int,
    db_path_str: str
):
    """
    步骤4：并行化构建树并存储为jsonl文件。
    """
    print(f"\n步骤 3 & 4: 正在为 {len(unique_ids)} 个唯一ID并行构建完整的历史树...")
    
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(db_path_str,)) as pool:
            # 使用 imap_unordered 以获得最佳性能
            results_iterator = pool.imap_unordered(build_complete_item_tree, unique_ids)
            
            progress_bar = tqdm(results_iterator, total=len(unique_ids), desc=f"构建并写入到 {output_jsonl_path.name}")
            for result_tree in progress_bar:
                if result_tree:
                    json_line = json.dumps(result_tree, ensure_ascii=False)
                    f.write(json_line + '\n')
    
    print(f"\n处理完成！所有物品树已保存到: {output_jsonl_path}")


if __name__ == "__main__":
    # 调整CPU核心数
    available_cores = os.cpu_count() or 1
    if NUM_CPU_CORES > available_cores:
        NUM_CPU_CORES = available_cores
        
    # --- 执行主流程 ---
    
    # 1 & 2: 采样并获取唯一ID
    unique_product_ids = subsample_and_extract_ids(
        original_task_path=original_tasks_base_path / TASK_NAME,
        sampled_output_path=sampled_task_path,
        id_col=ENTITY_ID_COL,
        n_train=NUM_TRAIN_SAMPLES,
        n_val=NUM_VAL_SAMPLES,
        n_test=NUM_TEST_SAMPLES,
    )

    if not unique_product_ids:
        print("在采样的数据中没有找到任何物品ID，程序退出。")
    else:
        # 3 & 4: 根据ID列表构建并存储树
        generate_trees_from_ids(
            unique_ids=unique_product_ids,
            output_jsonl_path=final_output_path / "rel-amazon-product_id.jsonl",
            num_workers=NUM_CPU_CORES,
            db_path_str=str(db_path)
        )