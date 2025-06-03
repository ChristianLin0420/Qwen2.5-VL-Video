#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3

#SBATCH --time=04:00:00                 # Adjust time limit as needed

#SBATCH --mem=0                         # all mem avail
#SBATCH --overcommit                    # allows more than one process per CPU
#SBATCH --dependency=singleton
#SBATCH --exclusive                     # exclusive node access

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32 # number of cores

#SBATCH --job-name=qwen25-vl-7b-grpo # customize your job name
#SBATCH --output=./logs/%x-%j/stdout.log      # !!!! Update log NAME Here
#SBATCH --error=./logs/%x-%j/stderr.log      # !!!! Update log NAME Here

#SBATCH --mail-type=BEGIN,END,FAIL # Adjust event types as needed
#SBATCH --mail-user=chrislin@nvidia.com


# Your training script here, if use conda then don't need container-image/ container-mounts arguments
IMAGE="christianlin0420/qwen25-vl-video:latest"

CMD='
conda activate qwen;
pip install datasets;
cd qwen-vl-finetune;
bash scripts/train_grpo.sh
'
# Pass CMD with double quotes
srun \
    --container-image=$IMAGE \
    --container-mounts=$HOME:/root,/lustre/fsw/portfolios/edgeai/users/chrislin/projects/Qwen2.5-VL-Video:/workspace,$HOME/.cache:/root/.cache \
    --container-writable \
    bash -c "$CMD"