#!/bin/bash

# ==============================================================================
# RelBench 剩余错误任务 重试脚本
#
# 功能:
# 1. 集合所有剩余的、因为OOM而失败的实验。
# 2. 请在根据下文的解决方案修改了代码或参数后，再运行此脚本。
# ==============================================================================

# 为本次运行创建一个带时间戳的结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiment_results/remaining_retry_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "剩余任务的重试结果将保存在: ${RESULTS_DIR}"
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

# 错误 1: gnn_link 在 avito 数据集上 (优化器显存不足)
# 解决方案: 减小 --channels 参数
# run_experiment "python examples/gnn_link.py --dataset rel-avito --task user-ad-visit --channels 64 --batch_size 128" "${RESULTS_DIR}/rel-avito_user-ad-visit_gnn_link.log"

run_experiment "python examples/idgnn_link.py --dataset rel-trial --task site-sponsor-run --num_layers 4 --batch_size 128" "${RESULTS_DIR}/rel-trial_site-sponsor-run_idgnn_link.log"

# 错误 2: idgnn_link 在 avito 数据集上 (评估矩阵显存不足)
# 解决方案: 减小 --batch_size 参数
# run_experiment "python examples/idgnn_link.py --dataset rel-avito --task user-ad-visit --batch_size 128" "${RESULTS_DIR}/rel-avito_user-ad-visit_idgnn_link.log"

# 错误 3: lightgbm_link 在 hm 数据集上 (数据预处理显存不足)
# 解决方案: 修改 lightgbm_link.py 文件 (详见下文)
# run_experiment "python examples/lightgbm_link.py --dataset rel-hm --task user-item-purchase" "${RESULTS_DIR}/rel-hm_user-item-purchase_lightgbm_link.log"


echo "=========================================================="
echo "所有剩余任务已尝试重新运行完毕。"