ACCOUNT="edgeai_tao-ptm_image-foundation-model-clip"
PARTITION="interactive_singlenode"
IMAGE="christianlin0420/qwen25-vl-video:latest"
JOB_NAME="qwen_2.5_vl_grpo_assy07"
srun \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --job-name $JOB_NAME \
    --gpus 8 \
    --cpus-per-task=32 \
    --mem-per-cpu=4G \
    --time=4:00:00 \
    --container-image=$IMAGE \
    --container-mounts=$HOME:/root,/lustre/fsw/portfolios/edgeai/users/chrislin/projects/Qwen2.5-VL-Video:/workspace,$HOME/.cache:/root/.cache \
    --pty /bin/bash