import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

def clean_nan_values(obj):
    """
    递归地遍历一个对象（字典或列表），将所有 NaN 值替换为 None，使其兼容JSON。
    *** 已修复：调整了检查顺序以避免对数组进行布尔求值 ***
    """
    # 1. 首先检查对象是否是容器类型，如果是，则递归处理
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(elem) for elem in obj]

    # 2. 如果不是容器，我们才将其视为标量，并检查它是否为NaN
    #    pandas.isna() 是最稳健的检查方式
    if pd.isna(obj):
        return None
    
    # 3. 如果是合法的标量值，直接返回
    return obj

def load_and_prepare_data(db_directory: Path) -> Dict[str, Any]:
    """
    加载所有相关的临床试验数据，并进行预处理和分组，以提高后续建树的效率。
    """
    main_tables = ['studies', 'designs', 'eligibilities', 'outcomes', 'outcome_analyses', 
                   'drop_withdrawals', 'reported_event_totals', 'sponsors', 'facilities', 
                   'conditions', 'interventions']
    link_tables = ['sponsors_studies', 'facilities_studies', 'conditions_studies', 
                   'interventions_studies']
    
    data = {}
    print("--- 正在加载数据文件 ---")
    try:
        for name in main_tables + link_tables:
            data[name] = pd.read_parquet(db_directory / f"{name}.parquet")
            print(f"- 已加载 {name}.parquet")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 {e.filename}。")
        print(f"请确认您的数据目录路径正确: {db_directory}")
        return {}

    print("\n--- 正在预处理和分组数据以提高效率 ---")
    
    data['designs'] = data['designs'].set_index('nct_id')
    data['eligibilities'] = data['eligibilities'].set_index('nct_id')

    grouped_data = {
        'outcomes_by_nct': data['outcomes'].groupby('nct_id'),
        'analyses_by_outcome': data['outcome_analyses'].groupby('outcome_id'),
        'drops_by_nct': data['drop_withdrawals'].groupby('nct_id'),
        'events_by_nct': data['reported_event_totals'].groupby('nct_id')
    }
    
    many_to_many_map = {
        'sponsors': 'sponsor_id', 'facilities': 'facility_id',
        'conditions': 'condition_id', 'interventions': 'intervention_id'
    }

    for name, join_key in many_to_many_map.items():
        link_table_name = f"{name}_studies"
        merged = pd.merge(data[link_table_name], data[name], on=join_key, how='inner')
        grouped_data[f'{name}_by_nct'] = merged.groupby('nct_id')
        print(f"- 已预处理 {name} (多对多)")

    print("--- 数据准备就绪 ---")
    
    return {'studies': data['studies'], 'designs': data['designs'], 'eligibilities': data['eligibilities'], **grouped_data}

def build_single_study_tree(study_row: pd.Series, prepared_data: Dict[str, Any]) -> Dict:
    """
    根据一行研究数据和预处理好的数据，构建一棵完整的信息树。
    """
    nct_id = study_row['nct_id']
    study_node = study_row.to_dict()

    try:
        study_node['design'] = prepared_data['designs'].loc[nct_id].to_dict()
    except KeyError:
        study_node['design'] = None
    try:
        study_node['eligibility'] = prepared_data['eligibilities'].loc[nct_id].to_dict()
    except KeyError:
        study_node['eligibility'] = None

    for name in ['sponsors', 'facilities', 'conditions', 'interventions']:
        try:
            study_node[name] = prepared_data[f'{name}_by_nct'].get_group(nct_id).to_dict('records')
        except KeyError:
            study_node[name] = []

    try:
        outcomes_df = prepared_data['outcomes_by_nct'].get_group(nct_id)
        outcomes_list = []
        for _, outcome_row in outcomes_df.iterrows():
            outcome_node = outcome_row.to_dict()
            outcome_id = outcome_node.get('id')
            if outcome_id and not pd.isna(outcome_id):
                try:
                    outcome_node['analyses'] = prepared_data['analyses_by_outcome'].get_group(outcome_id).to_dict('records')
                except KeyError:
                    outcome_node['analyses'] = []
            else:
                 outcome_node['analyses'] = []
            outcomes_list.append(outcome_node)
        study_node['outcomes'] = outcomes_list
    except KeyError:
        study_node['outcomes'] = []

    try:
        study_node['drop_withdrawals'] = prepared_data['drops_by_nct'].get_group(nct_id).to_dict('records')
    except KeyError:
        study_node['drop_withdrawals'] = []

    try:
        study_node['reported_event_totals'] = prepared_data['events_by_nct'].get_group(nct_id).to_dict('records')
    except KeyError:
        study_node['reported_event_totals'] = []

    return study_node

def save_all_study_trees_to_jsonl(prepared_data: Dict[str, Any], output_path: Path):
    """
    遍历所有研究，构建树，清理NaN值，并以流式写入JSONL文件。
    """
    df_studies = prepared_data['studies']
    print(f"\n正在将所有 {len(df_studies)} 项研究的数据流式写入到 JSONL 文件: {output_path}")
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, study_row in df_studies.iterrows():
            study_tree = build_single_study_tree(study_row, prepared_data)
            cleaned_tree = clean_nan_values(study_tree)
            json_line = json.dumps(cleaned_tree, ensure_ascii=False, default=str)
            f.write(json_line + '\n')
            count += 1
            if count % 1000 == 0:
                print(f"...已处理 {count} 项研究...")
    
    print(f"\n文件保存成功！共 {count} 项研究的数据已保存。")

if __name__ == "__main__":
    db_path = Path.home() / ".cache/relbench/rel-trial/db"
    
    prepared_data = load_and_prepare_data(db_path)

    if prepared_data:
        print("\n" + "="*50)
        print("方法一：构建单棵研究树进行验证")
        print("="*50)
        
        sample_study_row = prepared_data['studies'].iloc[0]
        print(f"正在为示例研究 (nct_id: {sample_study_row['nct_id']}) 构建信息树...")
        
        single_tree = build_single_study_tree(sample_study_row, prepared_data)
        cleaned_single_tree = clean_nan_values(single_tree)
        
        output_file = Path.home() / "relbench-reproduce/rel-data/rel-trial/single_study_tree.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_single_tree, f, ensure_ascii=False, indent=2, default=str)
        print("单棵研究树构建成功！结构如下（NaN将被显示为null）：")
        print(json.dumps(cleaned_single_tree, indent=2, default=str))

        print("\n" + "="*50)
        print("方法二：将所有研究的完整数据保存到 JSONL 文件")
        print("="*50)

        output_file = Path.home() / "relbench-reproduce/rel-data/rel-trial/all_clinical_studies.jsonl"
        save_all_study_trees_to_jsonl(prepared_data, output_file)
        
    else:
        print("\n未能加载数据，程序退出。")