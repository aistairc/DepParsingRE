#!/bin/bash

#$ -l rt_AF=2
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


export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/aae15163zd/cache/huggingface/models"

  #--model_name_or_path google/flan-t5-base
  #--dataset_name cnn_dailymail
  #--dataset_config 3.0.0
  #--model_name_or_path /scratch/aae15163zd/outputs/instruction-tuning/all-flan-t5-base-lma-tf32/checkpoint-60000/
  #--model_name_or_path google/flan-t5-base
  #--model_name_or_path ~/grp/ACL24/outputs/sm/cnndm-t5-large-con-ep3/
  #--source_prefix 'Write summary for this document. '
python_cmd="run_summarization.py
  --report_to wandb
  --run_name sm-cnndm-flan-large
  --model_name_or_path google/flan-t5-large
  --dataset_name cnn_dailymail
  --dataset_config 3.0.0
  --preprocessing_num_workers 40
  --pad_to_max_length
  --do_train
  --bf16
  --num_train_epochs 5
  --learning_rate 2e-04
  --warmup_ratio 0.1
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 1
  --predict_with_generate
  --save_strategy no
  --logging_strategy no
  --logging_steps 1000
  --evaluation_strategy epoch
  --eval_steps 1000
  --output_dir /scratch/aae15163zd/outputs/sum/foo
  --overwrite_output_dir
  --max_source_length 512
  --max_target_length 512
  --num_beams 5
  --deepspeed ds_configs/zero_1_config.json
"

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
