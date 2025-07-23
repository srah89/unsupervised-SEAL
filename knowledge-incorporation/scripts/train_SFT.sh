#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%A_sft.log
#SBATCH --gres=gpu:1

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
source /venv/main/bin/activate
cd /workspace/SEAL
mkdir -p logs

# -------- User-editable ---------------------------------------------- #
# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Put the (n-1)'th RL checkpoint. This script then trains the n'th checkpoint. The 0'th checkpoint is the base model.
MODEL_NAME="Qwen/Qwen2.5-7B"  # Put the (n-1)'th RL checkpoint. This script then trains the n'th checkpoint. The 0'th checkpoint is the base model.
TRAIN_FILE="knowledge-incorporation/data/synthetic_data/EM_SFT/sft_best1of3_0706_214926.jsonl"  # Path to training data output by src/EM/build_SFT_dataset.py
OUTPUT_DIR="models/iter1"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BATCH_SIZE=1
GRAD_ACC=5
EPOCHS=2
LR=3e-4
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LOG_STEPS=1
# --------------------------------------------------------------------- #

# export NCCL_P2P_DISABLE=1  # fixes hangs on some setups

# Kill vLLM temporarily to free GPU memory
echo "Stopping vLLM to free GPU memory for training..."
pkill -f "vllm serve" || true
sleep 3

echo "Launching SFT run on $(hostname)..."
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    knowledge-incorporation/src/EM/train_SFT.py \
    --train_file "${TRAIN_FILE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --logging_steps ${LOG_STEPS}

echo "Job finished."
