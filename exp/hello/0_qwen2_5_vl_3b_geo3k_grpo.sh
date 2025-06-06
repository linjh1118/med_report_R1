#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct
dataset_dir=/root/autodl-tmp/linjh/med_report_R1/dataset/processed/geometry3k

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


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${dataset_dir}@train \
    data.val_files=${dataset_dir}@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    data.rollout_batch_size=${rollout_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    worker.actor.global_batch_size=${global_batch_size} \
    worker.actor.micro_batch_size_per_device_for_experience=${micro_batch_size_per_device_for_experience} \
    worker.actor.micro_batch_size_per_device_for_update=${micro_batch_size_per_device_for_update} \
    