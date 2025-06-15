#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct
# train_file=/root/autodl-tmp/linjh/med_report_R1/dataset/processed/iu_xray/data/iu_xray-train.parquet
# val_files=/root/autodl-tmp/linjh/med_report_R1/dataset/processed/iu_xray/data/iu_xray-test.parquet

train_file=/root/autodl-tmp/wh/med_report_R1/assets/disease_samples_train.parquet
val_files=/root/autodl-tmp/wh/med_report_R1/assets/disease_samples_test.parquet

rollout_batch_size=16
global_batch_size=4
max_prompt_length=1024
max_response_length=1024
micro_batch_size_per_device_for_experience=1
micro_batch_size_per_device_for_update=1
n_gpus_per_node=4

SWANLAB_API_KEY=56wOpxTvc2OtM4XgSb3bi
swanlab login --relogin -k ${SWANLAB_API_KEY}
source /etc/network_turbo

project_name=med_report_rl
experiment_name=qwen2_5_vl_3b_grpo_filter_disease


nohup python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${train_file} \
    data.val_files=${val_files} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    data.rollout_batch_size=${rollout_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    worker.actor.global_batch_size=${global_batch_size} \
    worker.actor.micro_batch_size_per_device_for_experience=${micro_batch_size_per_device_for_experience} \
    worker.actor.micro_batch_size_per_device_for_update=${micro_batch_size_per_device_for_update} \
    worker.reward.reward_function=./examples/reward_function/disease_reward.py:compute_score > logs/${project_name}_${experiment_name}.log 2>&1 &
    