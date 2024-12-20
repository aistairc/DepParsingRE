#!/bin/bash

#$ -l rt_AF=4
#$ -l h_rt=10:00:00
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

  #--dataset_config_name 3.0.0
  #--dataset_name cnn_dailymail
  #--source_prefix 'Summarization|'
  #--source_prefix 'summarize: '
  #--source_prefix 'Please generate the question of following answer and context'
  #--model_name_or_path google/flan-t5-base
  #--model_name_or_path google/t5-base-lm-adapt
  #--model_name_or_path google/t5-small-lm-adapt
  #--model_name_or_path /scratch/aae15163zd/outputs/pt/dep-con-0-10/checkpoint-35000/
  #--model_name_or_path google/t5-v1_1-small
  #--source_prefix 'Write the question of following answer and context. '
  #--model_name_or_path /scratch/aae15163zd/outputs/qg/con-paren-I-ep20/checkpoint-2072/
  #--model_name_or_path /scratch/aae15163zd/outputs/qg/con-paren-I-ep20/checkpoint-2960/
  #--num_train_epochs 10
python_cmd="run_question_generation.py
  --report_to wandb
  --run_name qg-squad-dep-parsing
  --model_name_or_path t5-large
  --lm_type seq2seq
  --dataset_path ~/grp/inputs/squad-parsed-DEP/
  --text_column sentence
  --summary_column dep_prompt
  --preprocessing_num_workers 40
  --pad_to_max_length
  --do_train
  --do_eval
  --bf16
  --max_steps 3000
  --learning_rate 2e-04
  --warmup_ratio 0.1
  --per_device_train_batch_size 2
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps 2
  --predict_with_generate
  --save_strategy steps
  --save_steps 300
  --logging_strategy steps
  --logging_steps 300
  --evaluation_strategy no
  --output_dir ~/grp/ACL24/outputs/eval_dep/t5-large-msl-1024-steps-3k
  --overwrite_output_dir
  --overwrite_cache
  --max_source_length 256
  --max_target_length 1024
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
