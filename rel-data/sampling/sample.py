import pandas as pd
import json
from typing import Set
import os


if __name__ == '__main__':
    tasks = [
        "rel-amazon/tasks/user-churn", "rel-avito/tasks/user-visits",
        "rel-avito/tasks/user-clicks", "rel-event/tasks/user-repeat",
        "rel-event/tasks/user-ignore", "rel-hm/tasks/user-churn",
        "rel-stack/tasks/user-engagement", "rel-stack/tasks/user-badge",
        "rel-trial/tasks/study-outcome"
    ]

    tasks_id = {
        "rel-amazon/tasks/user-churn": "customer_id",
        "rel-avito/tasks/user-visits": "UserID",
        "rel-avito/tasks/user-clicks": "UserID",
        "rel-event/tasks/user-repeat": "user",
        "rel-event/tasks/user-ignore": "user",
        "rel-hm/tasks/user-churn": "customer_id",
        "rel-stack/tasks/user-engagement": "OwnerUserId",
        "rel-stack/tasks/user-badge": "UserId",
        "rel-trial/tasks/study-outcome": "nct_id"
    }

    dataset_ids = {
        "rel-amazon": "customer_id", "rel-avito": "UserID",
        "rel-event": "user_id", "rel-hm": "customer_id",
        "rel-stack": "UserId", "rel-trial": "nct_id"
    }

    for task in tasks:
        print(f"\n{'='*20} 开始处理任务: {task} {'='*20}")

        dataset_name = task.split('/')[0]
        parquet_id_column = tasks_id[task]
        jsonl_id_column = dataset_ids[dataset_name]

        # --- 1. 定义路径并创建输出目录 ---
        base_cache_path = os.path.expanduser(f'~/.cache/relbench/{task}')
        # 根据您的要求修改了此处的路径格式
        jsonl_objects_path = os.path.expanduser(f'~/relbench-reproduce/rel-data/{dataset_name}/{dataset_name}-{jsonl_id_column}.jsonl')
        output_dir = os.path.expanduser(f'~/relbench-reproduce/rel-data/sampling/{task}')
        os.makedirs(output_dir, exist_ok=True)

        all_target_ids: Set = set()
        splits_to_process = [
            ('train', 2000),
            ('val', 500),
            ('test', 500)
        ]

        # --- 2. 分别处理 train, val, test: 抽样、保存新Parquet、收集ID ---
        print(f"--- 步骤 1: 抽样任务并收集所有目标ID ---")
        for split_name, num_samples in splits_to_process:
            input_parquet_path = os.path.join(base_cache_path, f'{split_name}.parquet')
            output_parquet_path = os.path.join(output_dir, f'{split_name}.parquet')
            
            print(f"处理中: {input_parquet_path}")

            try:
                task_df = pd.read_parquet(input_parquet_path)
            except Exception as e:
                print(f"  错误: 无法读取Parquet文件 '{input_parquet_path}': {e}")
                continue

            if parquet_id_column not in task_df.columns:
                print(f"  错误: Parquet文件中找不到指定的ID列 '{parquet_id_column}'")
                continue

            # 抽样: 直接提取指定数量的行数
            if num_samples >= len(task_df):
                print(f"  警告: 请求选择 {num_samples} 行，但文件总共只有 {len(task_df)} 行。将使用所有行。")
                selected_tasks_df = task_df
            else:
                selected_tasks_df = task_df.head(num_samples)
            
            # 保存抽样后的任务文件
            selected_tasks_df.to_parquet(output_parquet_path, index=False)
            print(f"  已保存抽样后的任务到: '{output_parquet_path}'")

            # 更新总ID集合，用于后续对JSONL的去重筛选
            current_ids = set(selected_tasks_df[parquet_id_column])
            all_target_ids.update(current_ids)
            print(f"  收集到 {len(current_ids)} 个ID，总目标ID数: {len(all_target_ids)}")

        # --- 3. 根据收集到的所有ID，索引对象文件 ---
        print(f"\n--- 步骤 2: 从 '{jsonl_objects_path}' 中索引所有相关对象 ---")
        objects_by_id = {}
        try:
            with open(jsonl_objects_path, 'r', encoding='utf-8') as f_jsonl:
                for line in f_jsonl:
                    try:
                        obj = json.loads(line)
                        if jsonl_id_column in obj:
                            obj_id = obj[jsonl_id_column]
                            # 如果ID在我们的总目标中，并且尚未加载，则存入
                            if obj_id in all_target_ids and obj_id not in objects_by_id:
                                objects_by_id[obj_id] = obj
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"错误: 找不到JSONL对象文件 '{jsonl_objects_path}'")
            continue
        
        print(f"索引完成。共找到 {len(objects_by_id)} / {len(all_target_ids)} 个匹配的对象。")

        # --- 4. 将所有找到的对象写入统一的JSONL文件 ---
        output_jsonl_name = f"{dataset_name}-{jsonl_id_column}.jsonl"
        final_output_jsonl_path = os.path.join(output_dir, output_jsonl_name)
        
        print(f"\n--- 步骤 3: 将所有对象写入合并后的文件: '{final_output_jsonl_path}' ---")
        
        with open(final_output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for json_object in objects_by_id.values():
                json_line = json.dumps(json_object, ensure_ascii=False)
                f_out.write(json_line + '\n')
        
        print(f"成功写入了 {len(objects_by_id)} 个唯一的JSON对象。")

    print(f"\n{'='*25} 所有任务处理完毕! {'='*25}")
