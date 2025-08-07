#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%A_grpo.log
#SBATCH --gres=gpu:1

# -------- Environment ------------------------------------------------ #
source ~/.bashrc
source /venv/main/bin/activate
cd /workspace/SEAL
mkdir -p logs

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="Qwen/Qwen2.5-7B"  # Base model name
TRAIN_FILE="knowledge-incorporation/data/synthetic_data/EM_SFT/sft_best1of5_0806_010450.jsonl"  # Path to training data output by build_SFT_dataset.py
OUTPUT_DIR="models/grpo_iter1"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BATCH_SIZE=2
GRAD_ACC=8
EPOCHS=3
LR=1e-5
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LOG_STEPS=10
GROUP_SIZE=3
PREFERENCE_TEMPERATURE=0.1
BETA=0.0
EPSILON=0.2
SCALE_REWARDS=true

# -------- Wandb Configuration ---------------------------------------- #
WANDB_PROJECT="yoruba-knowledge-incorporation"
WANDB_ENTITY=""  # Leave empty for default entity
WANDB_TAGS=("yoruba" "knowledge-incorporation" "sft-training")
# --------------------------------------------------------------------- #

# Kill vLLM temporarily to free GPU memory
echo "Stopping vLLM to free GPU memory for training..."
pkill -f "vllm serve" || true
sleep 3

echo "Launching SFT training on $(hostname)..."
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
    --logging_steps ${LOG_STEPS} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "sft_$(basename "${TRAIN_FILE%.jsonl}")_$(date +%m%d_%H%M%S)" \
    --wandb_tags "${WANDB_TAGS[@]}"

echo "SFT training completed."
