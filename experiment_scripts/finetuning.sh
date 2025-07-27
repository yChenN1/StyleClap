#!/bin/bash

CURR_DIR=$(pwd)

# 切换到目标执行目录
cd ../src/laion_clap

torchrun --nproc-per-node=1 --master-port=21316 -- \
  -m training.main \
  --save-frequency 5 \
  --save-top-performance 3 \
  --save-most-recent \
  --dataset-type="toy" \
  --precision="fp32" \
  --batch-size=32 \
  --lr=1e-5 \
  --wd=0.0 \
  --epochs=45 \
  --no-eval \
  --workers=6 \
  --use-bn-sync \
  --amodel HTSAT-base \
  --tmodel roberta \
  --warmup 50 \
  --train-data '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks/chunk_000.parquet'\
  --val-data '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks_valid/chunk_valid.parquet' \
  --report-to "wandb" \
  --wandb-notes "finetune-instruct" \
  --datasetnames "instruct" \
  --datasetinfos "train" "test" \
  --top-k-checkpoint-select-dataset="Clotho-test" \
  --top-k-checkpoint-select-metric="mAP@10" \
  --openai-model-cache-dir /mnt/fast/nobackup/scratch4weeks/yc01815/transformers_cache \
  --logs /mnt/fast/nobackup/scratch4weeks/yc01815/clap/finetuning/0721 \
  --seed 3407 \
  --gather-with-grad \
  --optimizer "adam" \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --pretrained /mnt/fast/nobackup/scratch4weeks/yc01815/pretrain_models/HTSAT_music_speech_epoch_15_esc_89.25.pt 
