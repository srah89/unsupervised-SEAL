#!/bin/bash
#SBATCH --job-name=reward_model
#SBATCH --output=logs/%A_reward_model.log
#SBATCH --error=logs/%A_reward_model.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# -------- Environment ------------------------------------------------ #
source ~/.bashrc
source /venv/main/bin/activate
cd /workspace/SEAL
mkdir -p logs

# -------- User-editable ---------------------------------------------- #
DATA_DIR="knowledge-incorporation/data"
OUTPUT_DIR="knowledge-incorporation/models/reward_model"
# -------- Model Configuration ---------------------------------------- #
REWARD_MODEL_NAME="bert-base-uncased"  # Excellent for classification and reward modeling
GENERATION_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"  # Model for generating negative examples
NUM_SAMPLES=2000  # Number of preference pairs to generate
BATCH_SIZE=4  # Reduced for better memory management with custom architecture
LEARNING_RATE=5e-6  # Lower learning rate for stable training
NUM_EPOCHS=3
MAX_LENGTH=512
SEED=42

# -------- Wandb Configuration ---------------------------------------- #
WANDB_PROJECT="SEAL-knowledge-incorporation"
WANDB_ENTITY=""  # Leave empty for default entity
WANDB_TAGS=("SEAL" "knowledge-incorporation" "reward-model-training" "qwen" "custom-architecture")

# -------- GPU Memory Management -------------------------------------- #
# Kill vLLM temporarily to free GPU memory
echo "Stopping vLLM to free GPU memory for training..."
pkill -f "vllm serve" || true
sleep 5

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()" || true

# Check GPU status
echo "GPU Status before training:"
nvidia-smi

# --------------------------------------------------------------------- #

echo "Launching reward model training on $(hostname)..."
echo "=================================================="
echo "Configuration:"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Reward model: ${REWARD_MODEL_NAME}"
echo "Generation model: ${GENERATION_MODEL_NAME}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=================================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="${WANDB_ENTITY}"

# Pre-flight checks
echo "Running pre-flight checks..."

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory ${DATA_DIR} does not exist!"
    exit 1
fi

# Check if synthetic data exists
if [ ! -d "${DATA_DIR}/synthetic_data" ]; then
    echo "ERROR: Synthetic data directory ${DATA_DIR}/synthetic_data does not exist!"
    exit 1
fi

# Count available data files
DATA_FILE_COUNT=$(find "${DATA_DIR}/synthetic_data" -name "*.json" | wc -l)
echo "Found ${DATA_FILE_COUNT} JSON data files"

if [ "${DATA_FILE_COUNT}" -eq 0 ]; then
    echo "ERROR: No JSON data files found in ${DATA_DIR}/synthetic_data"
    exit 1
fi

echo "Pre-flight checks passed!"

# Run the reward model training with error handling
echo "Starting reward model training..."
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

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Reward model training completed successfully!"
    echo "Model saved to: ${OUTPUT_DIR}"
    
    # List output files
    echo "Output files:"
    ls -la "${OUTPUT_DIR}/"
    
    # Check model size
    if [ -f "${OUTPUT_DIR}/pytorch_model.bin" ]; then
        MODEL_SIZE=$(du -h "${OUTPUT_DIR}/pytorch_model.bin" | cut -f1)
        echo "Model size: ${MODEL_SIZE}"
    fi
    
    # Display final metrics if available
    if [ -f "${OUTPUT_DIR}/training_info.json" ]; then
        echo "Training summary:"
        cat "${OUTPUT_DIR}/training_info.json"
    fi
    
else
    echo "=================================================="
    echo "ERROR: Reward model training failed!"
    echo "Check the logs for details: logs/${SLURM_JOB_ID}_reward_model.log"
    
    # Display last few lines of the log for quick debugging
    if [ -f "logs/${SLURM_JOB_ID}_reward_model.log" ]; then
        echo "Last 20 lines of log:"
        tail -n 20 "logs/${SLURM_JOB_ID}_reward_model.log"
    fi
    
    exit 1
fi
