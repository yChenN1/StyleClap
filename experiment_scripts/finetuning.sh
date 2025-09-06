#!/bin/bash
EXP_NAME=$1
CURR_DIR=$(pwd)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache

# 切换到目标执行目录
cd ../src/laion_clap

export NCCL_CROSS_NIC=1
export OMP_NUM_THREADS=1
# export NCCL_ALGO=^Ring
NUM_TOTAL_GPU=$((ARNOLD_WORKER_NUM*ARNOLD_WORKER_GPU))

accelerate launch \
  --num_machines $ARNOLD_WORKER_NUM \
  --machine_rank $ARNOLD_ID \
  --num_processes $NUM_TOTAL_GPU \
  --main_process_ip $ARNOLD_WORKER_0_HOST \
  --main_process_port $(echo $ARNOLD_WORKER_0_PORT | cut -d"," -f2) \
  --dynamo_backend "no" \
  --mixed_precision "no" \
  -m training.main \
  --save-frequency 5 \
  --save-top-performance 3 \
  --save-most-recent \
  --dataset-type="toy" \
  --precision="fp32" \
  --batch-size=96 \
  --lr=1e-4 \
  --wd=0.0 \
  --epochs=45 \
  --no-eval \
  --workers=6 \
  --use-bn-sync \
  --amodel HTSAT-base \
  --tmodel roberta \
  --warmup 1000 \
  --train-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet/*.parquet' \
  --val-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet_eval/*.parquet' \
  --report-to "wandb" \
  --wandb-notes "finetune-instruct" \
  --datasetnames "instruct" \
  --datasetinfos "train" "test" \
  --top-k-checkpoint-select-dataset="Clotho-test" \
  --top-k-checkpoint-select-metric="mAP@10" \
  --openai-model-cache-dir /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/ \
  --logs /mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/${EXP_NAME} \
  --seed 3407 \
  --gather-with-grad \
  --optimizer "adam" \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --pretrained /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt \
  2>&1 | tee ${EXP_NAME}.log

# torchrun --nproc-per-node=8 --master-port=21316 -- \
#   -m training.main \
#   --save-frequency 5 \
#   --save-top-performance 3 \
#   --save-most-recent \
#   --dataset-type="toy" \
#   --precision="fp32" \
#   --batch-size=96 \
#   --lr=1e-4 \
#   --wd=0.0 \
#   --epochs=45 \
#   --no-eval \
#   --workers=6 \
#   --use-bn-sync \
#   --amodel HTSAT-base \
#   --tmodel roberta \
#   --warmup 1000 \
#   --train-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet/*.parquet'\
#   --val-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet_eval/*.parquet' \
#   --report-to "wandb" \
#   --wandb-notes "finetune-instruct" \
#   --datasetnames "instruct" \
#   --datasetinfos "train" "test" \
#   --top-k-checkpoint-select-dataset="Clotho-test" \
#   --top-k-checkpoint-select-metric="mAP@10" \
#   --openai-model-cache-dir /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/ \
#   --logs /mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/${EXP_NAME} \
#   --seed 3407 \
#   --gather-with-grad \
#   --optimizer "adam" \
#   --data-filling "repeatpad" \
#   --data-truncating "rand_trunc" \
#   --pretrained /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt | tee ${EXP_NAME}.log
