#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=20:00:00
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


export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/aae15163zd/cache/huggingface/models"

  #--predict_with_generate
  #--warmup_ratio 0.1
  #--max_steps 90000
  #--gradient_checkpointing 1
  #--deepspeed ./ds_configs/zero_1_config.json
  #--run_name pt-dep-con-all-flan-t5-base
  #--use_sequential_sampler
  #--max_steps 10000
python_cmd="run_pretraining.py
  --report_to wandb
  --run_name pt-wiki-0-10-con-paren
  --model_name_or_path google/t5-small-lm-adapt
  --dataset_name wikipedia-dep
  --cached_data_dirs_file ./caches_flan_dep.txt
  --preprocessing_num_workers 40
  --do_train
  --do_eval
  --num_train_epochs 1
  --optim adafactor
  --learning_rate 5e-04
  --lr_scheduler_type constant
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 1
  --save_strategy steps
  --save_steps 5000
  --logging_strategy steps
  --logging_steps 1000
  --evaluation_strategy steps
  --eval_steps 5000
  --output_dir /scratch/aae15163zd/outputs/pt/con_paren
"
  #--output_dir /scratch/aae15163zd/outputs/instruction-tuning/packed-no-mask-all-flan

# launch on slave nodes
node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
    qrsh -inherit -V -cwd $slave_node \
    eval "torchrun --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` "$python_cmd &
    node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
eval "torchrun --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` "$python_cmd

# finalize
wait
exit 0
