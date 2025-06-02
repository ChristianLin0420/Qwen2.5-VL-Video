FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    vim \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Initialize conda in bash
RUN /opt/conda/bin/conda init bash

# Create conda environment with Python 3.10
RUN conda create -n qwen python=3.10 -y

# Activate conda environment and install packages
SHELL ["conda", "run", "-n", "qwen", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install required packages
RUN pip install transformers==4.50.0 \
    deepspeed==0.16.4 \
    flash_attn==2.7.4.post1 \
    triton==3.0.0 \
    accelerate==1.4.0 \
    torchcodec==0.2 \
    decord \
    wandb

# Install ffmpeg through conda
RUN conda install -c conda-forge ffmpeg -y

# Install TorchCodec
RUN pip install torchcodec

# Install Correct Torch
RUN pip install torch==2.6.0

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Set git lfs
RUN git lfs install

# Make scripts executable
RUN chmod +x /workspace/qwen-vl-finetune/scripts/sft.sh

# Set the default shell for runtime
SHELL ["/bin/bash", "-c"]

# Set conda environment activation in bashrc
RUN echo "conda activate qwen" >> ~/.bashrc

# Default command
CMD ["bash"] 