#!/bin/bash

# System configuration
SYSTEM_NAME="EFC"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="."  # Current directory

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

# Execute each script from src directory
python inference.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}/" || exit 1

python prepare_leaderboard_submission.py --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" || exit 1

#python averitec_evaluate.py \
#    --prediction_file "leaderboard_submission/submission.csv" \
#    --label_file "leaderboard_submission/solution_dev.csv" || exit 1