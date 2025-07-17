import pandas as pd
from pathlib import Path
import json

def load_and_merge_data(db_directory: Path):
    """
    加载所有Parquet文件并根据外键关系将它们合并成一个大的DataFrame。
    """
    customer_path = db_directory / "customer.parquet"
    review_path = db_directory / "review.parquet"
    product_path = db_directory / "product.parquet"

    if not all([customer_path.exists(), review_path.exists(), product_path.exists()]):
        print("错误：数据文件缺失。")
        return None

    try:
        print("正在加载数据文件...")
        df_customer = pd.read_parquet(customer_path)
        df_review = pd.read_parquet(review_path)
        df_product = pd.read_parquet(product_path)
        
        print("正在合并数据（使用 left merge 保证数据完整性）...")
        merged_df = pd.merge(df_review, df_product, on='product_id', how='left')
        full_data = pd.merge(merged_df, df_customer, on='customer_id', how='left')
        print("数据准备完成。")
        return full_data
    except Exception as e:
        print(f"数据加载或合并时发生错误: {e}")
        return None

def build_single_customer_tree_simple(customer_id: int, full_data: pd.DataFrame):
    """
    为一个指定的customer_id构建一个简化的树。
    """
    customer_reviews_df = full_data[full_data['customer_id'] == customer_id].copy()

    if customer_reviews_df.empty:
        return None

    customer_info = customer_reviews_df.iloc[0]

    customer_node = {
        'customer_id': int(customer_id),
        'customer_name': str(customer_info['customer_name']),
        'reviews': []
    }

    for _, review_data in customer_reviews_df.iterrows():
        product_node = {
            'product_id': int(review_data['product_id']),
            'title': str(review_data['title']),
            'brand': str(review_data['brand']),
            'price': float(review_data['price']),
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

def save_all_trees_to_jsonl(full_data: pd.DataFrame, output_path: Path):
    print(f"正在将所有用户树流式写入到 JSONL 文件: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for customer_id, group in full_data.groupby('customer_id'):
            customer_tree = build_single_customer_tree_simple(customer_id, group)
            if customer_tree:
                json_line = json.dumps(customer_tree, ensure_ascii=False)
                f.write(json_line + '\n')
                count += 1
    
    print(f"文件保存成功！共 {count} 位客户的数据已保存到文件中。")

if __name__ == "__main__":
    db_path = Path.home() / ".cache/relbench/rel-amazon/db"
    full_merged_data = load_and_merge_data(db_path)

    if full_merged_data is not None and not full_merged_data.empty:
        output_file = Path.home() / "rel-data/rel-amazon/customer_trees_all.jsonl"
        save_all_trees_to_jsonl(full_merged_data, output_file)
        
    else:
        print("未能加载或处理数据，程序退出。")