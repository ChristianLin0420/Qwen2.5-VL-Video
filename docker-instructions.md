# Docker Instructions for Qwen2.5-VL-Video

This guide provides step-by-step instructions for building, running, and pushing the Qwen2.5-VL-Video Docker image.

## Prerequisites

1. Docker installed on your system
2. NVIDIA Docker runtime (for GPU support)
3. Docker Hub account (for pushing images)

## Building the Docker Image

1. Navigate to the project root directory:
```bash
cd /path/to/Qwen2.5-VL-Video
```

2. Build the Docker image:
```bash
docker build -t qwen25-vl-video:latest .
```

For a specific tag:
```bash
docker build -t qwen25-vl-video:v1.0 .
```

## Running the Docker Container

### Basic Run Command
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  qwen25-vl-video:latest
```

### Run with Persistent Storage
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/models:/models \
  -v /path/to/datasets:/datasets \
  qwen25-vl-video:latest
```

### Run with Specific GPU
```bash
docker run --gpus '"device=0"' -it --rm \
  -v $(pwd):/workspace \
  qwen25-vl-video:latest
```

### Run Training Script
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  qwen25-vl-video:latest \
  bash -c "conda activate qwen && cd /workspace/qwen-vl-finetune && bash scripts/sft.sh"
```

### Run with Network Host Mode (for distributed training)
```bash
docker run --gpus all -it --rm \
  --network host \
  -v $(pwd):/workspace \
  -e MASTER_ADDR=127.0.0.1 \
  -e MASTER_PORT=29500 \
  qwen25-vl-video:latest
```

## Pushing to Docker Hub

1. Login to Docker Hub:
```bash
docker login
```

2. Tag your image with your Docker Hub username:
```bash
docker tag qwen25-vl-video:latest YOUR_DOCKERHUB_USERNAME/qwen25-vl-video:latest
```

3. Push the image:
```bash
docker push YOUR_DOCKERHUB_USERNAME/qwen25-vl-video:latest
```

4. Push with multiple tags:
```bash
# Tag with version
docker tag qwen25-vl-video:latest YOUR_DOCKERHUB_USERNAME/qwen25-vl-video:v1.0
docker push YOUR_DOCKERHUB_USERNAME/qwen25-vl-video:v1.0

# Push both latest and version tags
docker push YOUR_DOCKERHUB_USERNAME/qwen25-vl-video:latest
```

## Docker Compose (Optional)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  qwen-vl:
    image: qwen25-vl-video:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MASTER_ADDR=127.0.0.1
      - MASTER_PORT=29500
    volumes:
      - .:/workspace
      - /path/to/models:/models
      - /path/to/datasets:/datasets
    stdin_open: true
    tty: true
    command: bash
```

Run with docker-compose:
```bash
docker-compose up -d
docker-compose exec qwen-vl bash
```

## Useful Docker Commands

### Check running containers
```bash
docker ps
```

### Check logs
```bash
docker logs CONTAINER_ID
```

### Execute command in running container
```bash
docker exec -it CONTAINER_ID bash
```

### Clean up Docker resources
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused resources
docker system prune -a
```

## Environment Variables

The following environment variables can be passed to the container:

- `MASTER_ADDR`: Master address for distributed training (default: 127.0.0.1)
- `MASTER_PORT`: Master port for distributed training (default: random)
- `WORLD_SIZE`: Number of nodes for distributed training
- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use

Example:
```bash
docker run --gpus all -it --rm \
  -e MASTER_ADDR=10.0.0.1 \
  -e MASTER_PORT=29500 \
  -e WORLD_SIZE=2 \
  -v $(pwd):/workspace \
  qwen25-vl-video:latest
```

## Troubleshooting

### GPU not detected
Ensure NVIDIA Docker runtime is installed:
```bash
# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Out of memory errors
Reduce batch size in the training script or use gradient checkpointing.

### Permission issues
Run with user permissions:
```bash
docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  qwen25-vl-video:latest
```

## Notes

- The conda environment `qwen` is automatically activated when you enter the container
- All project files are copied to `/workspace` in the container
- The container uses CUDA 12.4 with cuDNN support
- Flash Attention and other optimizations are pre-installed 