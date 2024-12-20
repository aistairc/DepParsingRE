#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -cwd
#$ -p -500

source /etc/profile.d/modules.sh
source ~/venv/f-quant/bin/activate
module load gcc/12.2.0
module load python/3.10/3.10.10
module load cuda/11.8/11.8.0
module load cudnn/8.8/8.8.1
module load nccl/2.15/2.15.5-1

export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/aae15163zd/cache/huggingface/models"

#python sample_flan.py
#python check_outputs.py
INDEX=${1}
python prepare_dep_parsed_squad.py --index $INDEX
