import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any
from pandas import Timestamp # 导入Timestamp类用于类型检查
import numpy as np # 导入numpy来处理其特定的数据类型

def make_json_serializable(obj):
    """
    递归地遍历一个对象，将所有非JSON序列化的pandas/numpy类型转换为兼容格式。
    - numpy整数 -> python int
    - numpy浮点数 -> python float
    - pandas Timestamp -> 'YYYY-MM-DD HH:MM:SS' 格式的字符串
    - NaN (及其他缺失值) -> None
    """
    # 1. 首先检查容器类型，并进行递归
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(elem) for elem in obj]

    # --- 2. 按顺序处理所有非标量类型 ---

    # NEW: 处理 numpy 整数类型 (如 int64)
    if isinstance(obj, np.integer):
        return int(obj)

    # NEW: 处理 numpy 浮点数类型 (如 float64)
    if isinstance(obj, np.floating):
        return float(obj)

    # 处理 pandas Timestamp 类型
    if isinstance(obj, Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')

    # 处理 NaN 及其他 pandas 缺失值
    if pd.isna(obj):
        return None
    
    # 3. 返回其他合法的标量值
    return obj

def load_and_prepare_data(db_directory: Path) -> Dict[str, Any]:
    """
    加载所有F1数据，并进行预处理、连接和分组，为高效建树做准备。
    """
    tables_to_load = ["drivers", "races", "circuits", "results", "qualifying", 
                      "constructors", "standings", "status"]
    
    data = {}
    print("--- 正在加载数据文件 ---")
    try:
        data['status'] = pd.read_parquet(db_directory / "status.parquet")
        print(f"- 已加载 status.parquet")
    except FileNotFoundError:
        print("- 警告: status.parquet 未找到, 将使用内置的状态映射。")
        status_map = {
            1: "Finished", 2: "Disqualified", 3: "Accident", 4: "Collision", 5: "Engine", 6: "Gearbox", 7: "Transmission", 8: "Clutch", 9: "Hydraulics", 10: "Electrical", 11: "+1 Lap", 12: "+2 Laps",
        }
        data['status'] = pd.DataFrame(list(status_map.items()), columns=['statusId', 'status'])
    
    try:
        for name in tables_to_load:
             if name != "status":
                data[name] = pd.read_parquet(db_directory / f"{name}.parquet")
                print(f"- 已加载 {name}.parquet")
    except FileNotFoundError as e:
        print(f"错误: 文件 {e.filename} 未找到。程序退出。")
        return {}

    print("\n--- 正在预处理和分组数据以提高效率 ---")
    
    races_with_circuits = pd.merge(data['races'], data['circuits'], on='circuitId', how='left')

    prepared_data = {
        'drivers_dict': data['drivers'].set_index('driverId').to_dict('index'),
        'constructors_dict': data['constructors'].set_index('constructorId').to_dict('index'),
        'races_with_circuits_dict': races_with_circuits.set_index('raceId').to_dict('index'),
        'status_dict': data['status'].set_index('statusId').to_dict('index')
    }

    prepared_data['results_by_driver'] = data['results'].groupby('driverId')
    prepared_data['qualifying_by_race_driver'] = data['qualifying'].groupby(['raceId', 'driverId'])
    prepared_data['standings_by_race_driver'] = data['standings'].groupby(['raceId', 'driverId'])

    print("--- 数据准备就绪 ---")
    return prepared_data

def build_single_driver_tree(driver_id: int, prepared_data: Dict[str, Any]) -> Dict:
    """
    根据一个车手ID和预处理好的数据，构建一棵完整的、深度嵌套的生涯信息树。
    """
    driver_info = prepared_data['drivers_dict'].get(driver_id, {})
    driver_node = {"driverId": driver_id, **driver_info}
    
    seasons = {}

    try:
        driver_results_df = prepared_data['results_by_driver'].get_group(driver_id)
    except KeyError:
        driver_node['seasons'] = []
        return driver_node

    for _, result_row in driver_results_df.iterrows():
        race_id = result_row['raceId']
        constructor_id = result_row['constructorId']
        
        race_info = prepared_data['races_with_circuits_dict'].get(race_id, {})
        year = race_info.get('year')
        if not year:
            continue

        if year not in seasons:
            seasons[year] = {"year": year, "races": []}
        
        race_node = {**race_info}
        performance_node = {}
        performance_node['constructor'] = prepared_data['constructors_dict'].get(constructor_id, {})
        
        try:
            q_res = prepared_data['qualifying_by_race_driver'].get_group((race_id, driver_id)).iloc[0]
            performance_node['qualifying'] = q_res.to_dict()
        except KeyError:
            performance_node['qualifying'] = None
        
        result_node = result_row.to_dict()
        status_info = prepared_data['status_dict'].get(result_row['statusId'], {})
        result_node['status'] = {"statusId": result_row['statusId'], **status_info}
        performance_node['result'] = result_node
        
        try:
            s_res = prepared_data['standings_by_race_driver'].get_group((race_id, driver_id)).iloc[0]
            performance_node['championship_standing'] = s_res.to_dict()
        except KeyError:
            performance_node['championship_standing'] = None
        
        race_node['driver_performance'] = performance_node
        seasons[year]['races'].append(race_node)

    driver_node['seasons'] = sorted(seasons.values(), key=lambda x: x['year'])
    return driver_node


def save_all_driver_trees_to_jsonl(prepared_data: Dict[str, Any], output_path: Path):
    """
    遍历所有车手，为每人构建树，并以流式写入JSONL文件。
    """
    all_driver_ids = sorted(prepared_data['drivers_dict'].keys())
    print(f"\n正在将所有 {len(all_driver_ids)} 位车手的数据流式写入到 JSONL 文件: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for driver_id in all_driver_ids:
            driver_tree = build_single_driver_tree(driver_id, prepared_data)
            # 使用我们最终版的、最强大的清理函数
            serializable_tree = make_json_serializable(driver_tree)
            json_line = json.dumps(serializable_tree, ensure_ascii=False)
            f.write(json_line + '\n')
            count += 1
            if count % 50 == 0:
                print(f"...已处理 {count} 位车手...")
    
    print(f"\n文件保存成功！共 {count} 位车手的数据已保存。")


if __name__ == "__main__":
    db_path = Path.home() / ".cache/relbench/rel-f1/db"
    
    prepared_data = load_and_prepare_data(db_path)

    if prepared_data:

        print("\n" + "="*50)
        print("方法二：将所有车手的完整数据保存到 JSONL 文件")
        print("="*50)
        
        output_file = Path.home() / "f1_drivers_complete.jsonl"
        save_all_driver_trees_to_jsonl(prepared_data, output_file)
        
    else:
        print("\n未能加载数据，程序退出。")