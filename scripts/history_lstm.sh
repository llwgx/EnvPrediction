#!/bin/bash

# 创建存储实验结果的主目录
mkdir -p experiments_logs

# 定义实验名称（可修改）
EXPERIMENT_NAME="best_history_long_lstm"

# 定义不同参数的范围
HISTORY_LENGTHS=(3 6 12 24 48 96 192)
PREDICTION_LENGTHS=(3 6 12 24)
HIDDEN_UNITS=(32 64)
TARGET_VARS=("IN_T" "IN_RH" "CO_2")

# 运行实验
for HISTORY in "${HISTORY_LENGTHS[@]}"; do
    for PREDICTION in "${PREDICTION_LENGTHS[@]}"; do
        for UNITS in "${HIDDEN_UNITS[@]}"; do
            for TARGET_VAR in "${TARGET_VARS[@]}"; do
                # 定义实验目录，包含实验名和参数信息
                EXP_DIR="experiments_logs/${EXPERIMENT_NAME}_HL${HISTORY}_PL${PREDICTION}_HU${UNITS}_TV${TARGET_VAR}"
                LOG_FILE="${EXP_DIR}/experiment.log"

                # 创建实验目录
                mkdir -p "$EXP_DIR"

                echo "Running experiment: ${EXPERIMENT_NAME} with HL=${HISTORY}, PL=${PREDICTION}, HU=${UNITS}, TV=${TARGET_VAR}"

                python run.py \
                    --experiment_name "${EXPERIMENT_NAME}" \
                    --file_path './data/FJ.csv' \
                    --history_length "$HISTORY" \
                    --prediction_length "$PREDICTION" \
                    --hidden_units "$UNITS" \
                    --target_var "$TARGET_VAR" > "$LOG_FILE" 2>&1

                echo "Experiment ${EXPERIMENT_NAME} completed. Log saved to $LOG_FILE"
            done
        done
    done
done

