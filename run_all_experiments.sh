#!/bin/bash

# ==============================================================================
# RelBench 全自动实验运行脚本
#
# 功能:
# 1. 自动遍历 RelBench 中的所有数据集和任务。
# 2. 根据任务类型（节点或链接预测）运行相应的 GNN 模型和基线模型。
# 3. 将每个实验的输出保存到 'experiment_results/' 目录下的独立日志文件中。
#
# 使用方法:
# 1. 将此脚本放置在 relbench 代码库的根目录。
# 2. 在终端中给予执行权限: chmod +x run_all_experiments.sh
# 3. 运行脚本: ./run_all_experiments.sh
#
# 注意:
# - 运行此脚本将非常耗时，可能需要数小时甚至数天，具体取决于您的硬件配置。
# - 脚本会下载所有数据集，占用大量磁盘空间。
# - 确保您的 Python 环境已激活，并已安装所有依赖项。
# ==============================================================================

# 创建用于存放结果的目录
RESULTS_DIR="experiment_results"
mkdir -p "$RESULTS_DIR"

# 定义数据集和它们的任务
# 格式: "任务名#任务类型"
# 任务类型: 'node' (实体分类/回归) 或 'link' (推荐)
declare -A tasks

tasks["rel-amazon"]="user-churn#node item-churn#node user-ltv#node item-ltv#node user-item-purchase#link user-item-rate#link user-item-review#link"
tasks["rel-avito"]="ad-ctr#node user-clicks#node user-visits#node user-ad-visit#link"
tasks["rel-event"]="user-attendance#node user-repeat#node user-ignore#node"
tasks["rel-f1"]="driver-dnf#node driver-top3#node driver-position#node"
tasks["rel-hm"]="user-churn#node item-sales#node user-item-purchase#link"
tasks["rel-stack"]="user-engagement#node user-badge#node post-votes#node user-post-comment#link post-post-related#link"
tasks["rel-trial"]="study-outcome#node study-adverse#node site-success#node condition-sponsor-run#link site-sponsor-run#link"

# 定义不同任务类型需要运行的脚本
node_scripts=("gnn_node.py" "lightgbm_node.py" "baseline_node.py")
link_scripts=("gnn_link.py" "idgnn_link.py" "lightgbm_link.py" "baseline_link.py")

# 获取总任务数用于进度显示
total_tasks=0
for dataset in "${!tasks[@]}"; do
    for task_info in ${tasks[$dataset]}; do
        total_tasks=$((total_tasks + 1))
    done
done

# 开始运行
echo "准备开始运行 RelBench 所有实验，总共 ${total_tasks} 个任务。"
echo "结果将保存在 '${RESULTS_DIR}' 目录下。"
echo "这将会是一个非常漫长的过程..."
echo "=========================================================="

current_task_num=0
# 遍历所有数据集
for dataset in "${!tasks[@]}"; do
    echo ""
    echo "**********************************************************"
    echo "*** 开始处理数据集: ${dataset}"
    echo "**********************************************************"

    # 遍历该数据集下的所有任务
    for task_info in ${tasks[$dataset]}; do
        current_task_num=$((current_task_num + 1))

        # 解析任务名和任务类型
        task_name=${task_info%#*}
        task_type=${task_info#*#}

        echo ""
        echo "----------------------------------------------------------"
        echo "--- [${current_task_num}/${total_tasks}] 正在运行任务: ${task_name} (类型: ${task_type}) ---"
        echo "----------------------------------------------------------"


        if [ "$task_type" == "node" ]; then
            scripts_to_run=("${node_scripts[@]}")
        elif [ "$task_type" == "link" ]; then
            scripts_to_run=("${link_scripts[@]}")
        else
            echo "错误：未知的任务类型 '${task_type}' for task '${task_name}'"
            continue
        fi

        # 运行该任务对应的所有脚本
        for script in "${scripts_to_run[@]}"; do
            model_name=${script%.py}
            output_file="${RESULTS_DIR}/${dataset}_${task_name}_${model_name}.log"
            command="python examples/${script} --dataset ${dataset} --task ${task_name}"

            echo "   -> 正在执行: ${command}"
            echo "      日志文件: ${output_file}"

            # 执行命令并将标准输出和错误输出都重定向到日志文件
            # 使用 & 在后台运行，可以并行处理，但对资源要求高
            # 这里我们使用串行执行，更稳定
            $command > "$output_file" 2>&1

            # 检查上一个命令的退出状态
            if [ $? -eq 0 ]; then
                echo "      ✅ 执行成功."
            else
                echo "      ❌ 执行失败. 请检查日志文件: ${output_file}"
            fi
        done
    done
done

echo ""
echo "=========================================================="
echo "🎉 所有实验已全部运行完毕！"
echo "=========================================================="
