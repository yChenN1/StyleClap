#!/bin/bash
EXP_NAME=$1
CURR_DIR=$(pwd)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache

# 切换到目标执行目录
cd ../src/laion_clap

# python evaluate/eval_linear_probe.py \
#     --save-frequency 50 \
#     --save-top-performance 3 \
#     --save-most-recent \
#     --dataset-type="toy" \
#     --precision="fp32" \
#     --warmup 0 \
#     --batch-size=160 \
#     --lr=1e-4 \
#     --wd=0.1 \
#     --epochs=100 \
#     --workers=8 \
#     --use-bn-sync \
#     --freeze-text \
#     --amodel HTSAT-base \
#     --tmodel roberta \
#     --train-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_filtered/*.parquet' \
#     --val-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval_filtered/*.parquet' \
#     --report-to "wandb" \
#     --wandb-notes "finetune-instruct" \
#     --datasetnames "instruct" \
#     --datasetinfos "train" \
#     --seed 3407 \
#     --remotedata \
#     --logs /mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/${EXP_NAME} \
#     --gather-with-grad \
#     --lp-loss="ce" \
#     --lp-metrics="acc" \
#     --lp-lr=1e-4 \
#     --lp-mlp \
#     --lp-out-ch 28 \
#     --openai-model-cache-dir /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/ \
#     --pretrained="/mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/" \
#     --data-filling "repeatpad" \
#     --data-truncating "rand_trunc" \
#     --optimizer "adam"


export NCCL_CROSS_NIC=1
export OMP_NUM_THREADS=1
# export NCCL_ALGO=^Ring
NUM_TOTAL_GPU=$((ARNOLD_WORKER_NUM*ARNOLD_WORKER_GPU))
NUM_TOTAL_GPU=8
ARNOLD_WORKER_NUM=1
ARNOLD_ID=0

accelerate launch \
  --num_machines $ARNOLD_WORKER_NUM \
  --machine_rank $ARNOLD_ID \
  --num_processes $NUM_TOTAL_GPU \
  --main_process_ip $ARNOLD_WORKER_0_HOST \
  --main_process_port $(echo $ARNOLD_WORKER_0_PORT | cut -d"," -f2) \
  --dynamo_backend "no" \
  --mixed_precision "no" \
    -m evaluate.eval_linear_probe \
    --save-frequency 20 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="toy" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=4 \
    --use-bn-sync \
    --freeze-text \
    --amodel HTSAT-base \
    --tmodel roberta \
    --train-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_filtered/*.parquet' \
    --val-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval_filtered/*.parquet' \
    --report-to "wandb" \
    --wandb-notes "finetune-instruct" \
    --datasetnames "instruct" \
    --datasetinfos "train" \
    --seed 3407 \
    --remotedata \
    --logs /mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/${EXP_NAME} \
    --gather-with-grad \
    --lp-loss="ce" \
    --lp-metrics="acc" \
    --lp-lr=1e-4 \
    --lp-mlp \
    --lp-out-ch 28 \
    --openai-model-cache-dir /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/ \
    --pretrained="/mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --optimizer "adam"