#!/bin/bash

#$ -l rt_F=8
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

  #--dataset_config_name 3.0.0
  #--dataset_name cnn_dailymail
  #--source_prefix 'Summarization|'
  #--source_prefix 'summarize: '
  #--source_prefix 'Please generate the question of following answer and context'
  #--model_name_or_path google/flan-t5-base
  #--model_name_or_path google/t5-base-lm-adapt
  #--model_name_or_path /scratch/aae15163zd/outputs/pt/dep-con-0-10/checkpoint-35000/
  #--model_name_or_path google/t5-v1_1-small
  #--source_prefix 'Write the question of following answer and context. '
  #--predict_with_generate
python_cmd="run_question_generation.py
  --report_to wandb
  --run_name qg-squad-pre-dep-prompt
  --model_name_or_path google/t5-v1_1-large
  --lm_type seq2seq
  --dataset_name squad
  --dataset_path /scratch/aae15163zd/inputs/squad-parsed-con-parenthesis-I
  --source_prefix ''
  --text_column input
  --summary_column output
  --preprocessing_num_workers 40
  --pad_to_max_length
  --do_train
  --num_train_epochs 5
  --learning_rate 1e-04
  --warmup_ratio 0.1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 4
  --save_strategy epoch
  --logging_strategy epoch
  --evaluation_strategy epoch
  --output_dir /scratch/aae15163zd/outputs/qg/large-v1_1-squad-con-paren-ep5-noaccum
  --overwrite_output_dir
  --overwrite_cache
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
