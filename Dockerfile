# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    libzmq3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install additional dependencies that might be needed
RUN pip3 install \
    jupyter \
    notebook \
    ipykernel \
    zmq \
    pyzmq

# Copy the entire project
COPY . .

# Make scripts executable
RUN find knowledge-incorporation/scripts -name "*.sh" -exec chmod +x {} \;
RUN find few-shot -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /workspace/data \
    && mkdir -p /workspace/results \
    && mkdir -p /workspace/knowledge-incorporation/data \
    && mkdir -p /workspace/knowledge-incorporation/results

# Expose common ports used by the project
EXPOSE 8000 8080 8265 11111 11112

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden)
CMD ["bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

