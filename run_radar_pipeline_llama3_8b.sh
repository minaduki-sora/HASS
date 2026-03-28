#!/bin/bash

# Define paths
BASE_MODEL_PATH="../weights/hf/Meta-Llama-3-8B-Instruct"
EA_MODEL_PATH="../weights/hass/HASS-LLaMA3-Instruct-8B"
BENCH_NAME="shareGPT"
TEMP=1.0

# 1. Collect offline data (state sequence and acceptance length distribution)
echo "Step 1: Collecting offline data for RL..."
python -m HASS.ge_data.ge_data_llama3_rb \
    --ea-model-path $EA_MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --bench-name $BENCH_NAME \
    --num-gpus-total 1 \
    --depth 7 \
    --top-k 10 \
    --temperature $TEMP \
    --answer-file output/${BENCH_NAME}/llama3-d7-rb.jsonl \
    --save-dataset data/scores_rb/${BENCH_NAME}-llama3-d7-topk10-t${TEMP}/ \
    --question-begin 0 \
    --question-end 1000

# 2. Collect time log data (for reward function)
echo "Step 2: Collecting time log data..."
python HASS/evaluation/gen_ea_answer_llama3chat_log.py \
    --ea-model-path $EA_MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --bench-name $BENCH_NAME \
    --num-gpus-total 1 \
    --depth 7 \
    --top-k 10 \
    --temperature $TEMP \
    --time-file output/${BENCH_NAME}/llama3/t1d7/time_rl.jsonl \
    --question-begin 0 \
    --question-end 10

# 3. Train the prediction model (Hawkeye/RADAR)
echo "Step 3: Training the prediction model..."
# Note: You might need to adjust the config file in HASS/train/
python HASS/train/train_hawkeye.py \
    --config HASS/train/train_llama3.1_hidden.json

# 4. Test RADAR
echo "Step 4: Testing the prediction model..."
python HASS/evaluation/gen_ea_we_answer_llama3chat.py \
    --ea-model-path $EA_MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --eye-model-path output/${BENCH_NAME}/llama3/t1d7/hawkeye.pt \
    --bench-name $BENCH_NAME \
    --num-gpus-total 1 \
    --depth 7 \
    --top-k 10 \
    --temperature $TEMP
