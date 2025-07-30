#!/bin/bash

CURR_DIR=$(pwd)

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
    --report-to "wandb" \
    --wandb-notes "10.17-freesound-dataset-4#" \
    --datasetnames "instruct" \
    --datasetinfos "train" "test" \
    --seed 3407 \
    --train-data '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks/chunk_000.parquet'\
    --val-data '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks/chunk_000.parquet' \
    --logs /mnt/fast/nobackup/scratch4weeks/yc01815/clap/finetuning/0728 \
    --gather-with-grad \
    --openai-model-cache-dir /mnt/fast/nobackup/scratch4weeks/yc01815/transformers_cache \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained /mnt/fast/nobackup/scratch4weeks/yc01815/pretrain_models/0728base_lr1e4/epoch_45.pt