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

#SBATCH --job-name=aircraft-nvila-8b-video-image-vt-ft-llm-ft-grpo-8gen-1epcs-no-think-kl-original-example # customize your job name
#SBATCH --output=./logs/%x-%j/stdout.log      # !!!! Update log NAME Here
#SBATCH --error=./logs/%x-%j/stderr.log      # !!!! Update log NAME Here

#SBATCH --mail-type=BEGIN,END,FAIL # Adjust event types as needed
#SBATCH --mail-user=chrislin@nvidia.com


# Your training script here, if use conda then don't need container-image/ container-mounts arguments
IMAGE="nvcr.io/nvidian/iva/nvila-grpo:v1_20250504"

CMD='
cd /workspace/VILA;
([[ "$SLURM_LOCALID" == "0" ]] && echo "installing deps" && pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils) ; ([[ "$SLURM_LOCALID" != "0" ]] && echo "sleeping" && sleep 30) ;
bash scripts/NVILA/fgvc_aircraft_8b_image_grpo.sh --vt ft --llm ft
'
# Pass CMD with double quotes
srun \
    --container-image=$IMAGE \
    --container-mounts=$HOME:/root,/lustre/fsw/portfolios/edgeai/users/chrislin/projects/multi-modality-research/VILA:/workspace/VILA,/lustre/fsw/portfolios/edgeai/users/chrislin/pretrained_weight:/workspace/pretrained_weight,$HOME/.cache:/root/.cache \
    --container-writable \
    bash -c "$CMD"