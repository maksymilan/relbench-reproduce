#!/bin/bash

# ==============================================================================
# RelBench GNN 实验运行脚本 (GPU服务器专用)
#
# 功能:
# 1. 自动遍历 RelBench 中的所有数据集和任务。
# 2. 只运行 GNN 相关的模型。
# 3. 为每次运行创建一个带时间戳的结果目录。
# ==============================================================================

# 为本次运行创建一个带时间戳的结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiment_results/gnn_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# 定义数据集和它们的任务
declare -A tasks
tasks["rel-amazon"]="user-churn#node item-churn#node user-ltv#node item-ltv#node user-item-purchase#link user-item-rate#link user-item-review#link"
tasks["rel-avito"]="ad-ctr#node user-clicks#node user-visits#node user-ad-visit#link"
tasks["rel-event"]="user-attendance#node user-repeat#node user-ignore#node"
tasks["rel-f1"]="driver-dnf#node driver-top3#node driver-position#node"
tasks["rel-hm"]="user-churn#node item-sales#node user-item-purchase#link"
tasks["rel-stack"]="user-engagement#node user-badge#node post-votes#node user-post-comment#link post-post-related#link"
tasks["rel-trial"]="study-outcome#node study-adverse#node site-success#node condition-sponsor-run#link site-sponsor-run#link"

# 定义需要运行的 GNN 脚本
node_scripts_gnn=("gnn_node.py")
link_scripts_gnn=("gnn_link.py" "idgnn_link.py")

# 运行单个实验的函数
run_experiment() {
    local script=$1
    local dataset=$2
    local task_name=$3
    local model_name=${script%.py}
    local output_file="${RESULTS_DIR}/${dataset}_${task_name}_${model_name}.log"
    local command="python examples/${script} --dataset ${dataset} --task ${task_name}"

    echo "    -> 正在执行GNN实验: ${command}"
    echo "       日志文件: ${output_file}"
    $command > "$output_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "       ✅ 执行成功."
    else
        echo "       ❌ 执行失败. 请检查日志文件: ${output_file}"
    fi
}

# 获取总任务数
total_tasks=0
for dataset in "${!tasks[@]}"; do
    tasks_in_dataset=(${tasks[$dataset]})
    total_tasks=$((total_tasks + ${#tasks_in_dataset[@]}))
done

# 开始运行
echo "准备开始运行 RelBench GNN 实验，总共 ${total_tasks} 个任务。"
echo "结果将保存在 '${RESULTS_DIR}' 目录下。"
echo "=========================================================="

current_task_num=0
for dataset in "${!tasks[@]}"; do
    echo ""
    echo "*** 开始处理数据集: ${dataset} ***"
    for task_info in ${tasks[$dataset]}; do
        current_task_num=$((current_task_num + 1))
        task_name=${task_info%#*}
        task_type=${task_info#*#}

        echo "----------------------------------------------------------"
        echo "--- [${current_task_num}/${total_tasks}] 任务: ${task_name} (类型: ${task_type}) ---"

        if [ "$task_type" == "node" ]; then
            for script in "${node_scripts_gnn[@]}"; do
                run_experiment "$script" "$dataset" "$task_name"
            done
        elif [ "$task_type" == "link" ]; then
            for script in "${link_scripts_gnn[@]}"; do
                run_experiment "$script" "$dataset" "$task_name"
            done
        fi
    done
done

echo ""
echo "=========================================================="
echo "🎉 所有 GNN 实验已全部运行完毕！"
echo "结果已保存在: ${RESULTS_DIR}"
echo "=========================================================="
