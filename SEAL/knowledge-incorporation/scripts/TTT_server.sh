#!/bin/bash
#SBATCH --job-name=server
#SBATCH --output=logs/%A_server.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
source /venv/main/bin/activate
cd /workspace/SEAL
mkdir -p logs

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # HF model name or path to RL checkpoint (e.g. models/iter1)
VLLM_SERVER_GPUS="0"
INNER_LOOP_GPU="0"
PORT=18000
ZMQ_PORT=5555

MAX_SEQ_LENGTH=2048  # Max sequence length for training
EVAL_MAX_TOKENS=64   # Max generated tokens for evaluation completions
EVAL_TEMPERATURE=0.0
EVAL_TOP_P=1.0

MAX_LORA_RANK=32     # Max LoRA rank that will be used
# --------------------------------------------------------------------- #
echo "Launching TTT server on $(hostname)..."

set -a
source .env
set +a

VLLM_HOST="127.0.0.1"
VLLM_API_URL="http://${VLLM_HOST}:${PORT}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Check if vLLM is already running on the port
if curl --silent --fail ${VLLM_API_URL}/health >/dev/null 2>&1; then
    echo "vLLM already running on port ${PORT} - killing and restarting with optimized settings"
    pkill -f "vllm serve" || true
    sleep 3
fi

echo "Launching vLLM on GPUs ${VLLM_SERVER_GPUS} with reduced memory usage"
CUDA_VISIBLE_DEVICES=${VLLM_SERVER_GPUS} vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port ${PORT} \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --enable-lora \
    --max-lora-rank ${MAX_LORA_RANK} \
    --trust-remote-code \
    --enforce-eager \
    --download-dir /workspace/models \
    > "logs/${SLURM_JOB_ID}_vllm_server.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM..."
until curl --silent --fail ${VLLM_API_URL}/health >/dev/null; do sleep 3; done
echo "    vLLM ready at ${VLLM_API_URL}"

echo "Starting Inner Loop server on GPU ${INNER_LOOP_GPU}..."
CUDA_VISIBLE_DEVICES=${INNER_LOOP_GPU} python3 -m knowledge-incorporation.src.inner.TTT_server \
    --vllm_api_url "${VLLM_API_URL}" \
    --model "${MODEL_NAME}" \
    --zmq_port ${ZMQ_PORT} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --eval_max_tokens ${EVAL_MAX_TOKENS} \
    --eval_temperature ${EVAL_TEMPERATURE} \
    --eval_top_p ${EVAL_TOP_P} \
    > logs/${SLURM_JOB_ID}_TTT_server.log 2>&1 &

ZMQ_PID=$!
echo "    Inner Loop Server started with PID ${ZMQ_PID}."
echo "Ready to accept requests on port ${ZMQ_PORT}."

trap "echo 'Shutting down...'; kill ${ZMQ_PID} ${VLLM_PID}" EXIT
wait

echo "Job finished."
