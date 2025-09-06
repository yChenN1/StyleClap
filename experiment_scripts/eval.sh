#!/bin/bash

CURR_DIR=$(pwd)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache

# 切换到目标执行目录
cd ../src/laion_clap

torchrun --nproc-per-node=1 --master-port=21316 -- \
-m evaluate.eval_retrieval_main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="toy" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=32 \
    --wd=0.0 \
    --epochs=50 \
    --workers=6 \
    --use-bn-sync \
    --freeze-text \
    --amodel HTSAT-base \
    --tmodel roberta \
    --report-to "tensorboard" \
    --wandb-notes "10.17-freesound-dataset-4#" \
    --datasetnames "instruct" \
    --datasetinfos "train" "test" \
    --seed 3407 \
    --train-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet/*.parquet'\
    --val-data '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech_parquet_eval/chunk_331.parquet' \
    --logs /mnt/bn/tanman-yg/chenqi/code/StyleClap/eval/ \
    --gather-with-grad \
    --openai-model-cache-dir /mnt/fast/nobackup/scratch4weeks/yc01815/transformers_cache \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt
    # --pretrained /mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0809giga/2025_08_09-18_30_41-model_HTSAT-base-lr_5e-05-b_96-j_6-p_fp32/checkpoints/epoch_45.pt