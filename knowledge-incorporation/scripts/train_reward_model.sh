#!/bin/bash
#SBATCH --job-name=reward_model
#SBATCH --output=logs/%A_reward_model.log
#SBATCH --gres=gpu:1

# -------- Environment ------------------------------------------------ #
source ~/.bashrc
source /venv/main/bin/activate
cd /workspace/SEAL
mkdir -p logs

# -------- User-editable ---------------------------------------------- #
DATA_DIR="knowledge-incorporation/data"
OUTPUT_DIR="knowledge-incorporation/models/reward_model"
REWARD_MODEL_NAME="microsoft/DialoGPT-medium"  # Smaller model for efficiency
GENERATION_MODEL_NAME="microsoft/DialoGPT-medium"  # Model for generating negative examples
NUM_SAMPLES=2000  # Number of preference pairs to generate
BATCH_SIZE=8
LEARNING_RATE=1e-5
NUM_EPOCHS=3
MAX_LENGTH=512
SEED=42

# -------- Wandb Configuration ---------------------------------------- #
WANDB_PROJECT="yoruba-knowledge-incorporation"
WANDB_ENTITY=""  # Leave empty for default entity
WANDB_TAGS=("yoruba" "knowledge-incorporation" "reward-model-training")
# --------------------------------------------------------------------- #

# Kill vLLM temporarily to free GPU memory
echo "Stopping vLLM to free GPU memory for training..."
pkill -f "vllm serve" || true
sleep 3

echo "Launching reward model training on $(hostname)..."
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Reward model: ${REWARD_MODEL_NAME}"
echo "Generation model: ${GENERATION_MODEL_NAME}"
echo "Number of samples: ${NUM_SAMPLES}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the reward model training
python knowledge-incorporation/src/EM/train_reward_model.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --reward_model_name "${REWARD_MODEL_NAME}" \
    --generation_model_name "${GENERATION_MODEL_NAME}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --max_length "${MAX_LENGTH}" \
    --seed "${SEED}"

echo "Reward model training complete!"
echo "Model saved to: ${OUTPUT_DIR}" 