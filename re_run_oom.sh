#!/bin/bash

# ==============================================================================
# RelBench 显存不足(OOM)任务 重试脚本
#
# 功能:
# 1. 集合所有因为 CUDA out of memory 失败的实验命令。
# 2. 方便在更换到更大显存的GPU服务器后进行批量重试。
#
# 注意:
# - 在原硬件上直接运行此脚本很可能会再次失败。
# ==============================================================================

# 为本次运行创建一个带时间戳的结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiment_results/oom_retry_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "OOM 重试任务的结果将保存在: ${RESULTS_DIR}"
echo "=========================================================="

run_experiment() {
    local command_str="$1"
    local log_file="$2"
    echo "--- 正在执行: ${command_str}"
    echo "    日志文件: ${log_file}"
    eval "${command_str}" > "${log_file}" 2>&1
    if [ $? -eq 0 ]; then
        echo "    ✅ 执行成功."
    else
        echo "    ❌ 执行失败. 请检查日志: ${log_file}"
    fi
    echo "----------------------------------------------------------"
}

# --- rel-amazon (所有GNN任务均OOM) ---
run_experiment "python examples/gnn_node.py --dataset rel-amazon --task user-churn" "${RESULTS_DIR}/rel-amazon_user-churn_gnn_node.log"
run_experiment "python examples/gnn_node.py --dataset rel-amazon --task item-churn" "${RESULTS_DIR}/rel-amazon_item-churn_gnn_node.log"
run_experiment "python examples/gnn_node.py --dataset rel-amazon --task user-ltv" "${RESULTS_DIR}/rel-amazon_user-ltv_gnn_node.log"
run_experiment "python examples/gnn_node.py --dataset rel-amazon --task item-ltv" "${RESULTS_DIR}/rel-amazon_item-ltv_gnn_node.log"
run_experiment "python examples/gnn_link.py --dataset rel-amazon --task user-item-purchase" "${RESULTS_DIR}/rel-amazon_user-item-purchase_gnn_link.log"
run_experiment "python examples/idgnn_link.py --dataset rel-amazon --task user-item-purchase" "${RESULTS_DIR}/rel-amazon_user-item-purchase_idgnn_link.log"
run_experiment "python examples/gnn_link.py --dataset rel-amazon --task user-item-rate" "${RESULTS_DIR}/rel-amazon_user-item-rate_gnn_link.log"
run_experiment "python examples/idgnn_link.py --dataset rel-amazon --task user-item-rate" "${RESULTS_DIR}/rel-amazon_user-item-rate_idgnn_link.log"
run_experiment "python examples/gnn_link.py --dataset rel-amazon --task user-item-review" "${RESULTS_DIR}/rel-amazon_user-item-review_gnn_link.log"
run_experiment "python examples/idgnn_link.py --dataset rel-amazon --task user-item-review" "${RESULTS_DIR}/rel-amazon_user-item-review_idgnn_link.log"

# --- rel-avito (部分链接预测任务OOM) ---
run_experiment "python examples/gnn_link.py --dataset rel-avito --task user-ad-visit" "${RESULTS_DIR}/rel-avito_user-ad-visit_gnn_link.log"
run_experiment "python examples/idgnn_link.py --dataset rel-avito --task user-ad-visit" "${RESULTS_DIR}/rel-avito_user-ad-visit_idgnn_link.log"

# --- rel-event (部分节点任务OOM) ---
run_experiment "python examples/gnn_node.py --dataset rel-event --task user-ignore" "${RESULTS_DIR}/rel-event_user-ignore_gnn_node.log"
run_experiment "python examples/gnn_node.py --dataset rel-event --task user-repeat" "${RESULTS_DIR}/rel-event_user-repeat_gnn_node.log"

# --- rel-hm (LightGBM链接预测OOM) ---
run_experiment "python examples/lightgbm_link.py --dataset rel-hm --task user-item-purchase" "${RESULTS_DIR}/rel-hm_user-item-purchase_lightgbm_link.log"

echo "=========================================================="
echo "所有OOM任务已尝试重新运行完毕。"