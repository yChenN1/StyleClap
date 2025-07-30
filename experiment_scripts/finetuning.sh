#!/bin/bash
EXP_NAME=$1
CURR_DIR=$(pwd)

# 切换到目标执行目录
cd ../src/laion_clap

torchrun --nproc-per-node=8 --master-port=21316 -- \
  -m training.main \
  --save-frequency 5 \
  --save-top-performance 3 \
  --save-most-recent \
  --dataset-type="toy" \
  --precision="fp32" \
  --batch-size=96 \
  --lr=5e-5 \
  --wd=0.0 \
  --epochs=45 \
  --no-eval \
  --workers=6 \
  --use-bn-sync \
  --amodel HTSAT-base \
  --tmodel roberta \
  --warmup 1000 \
  --train-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset/*.parquet'\
  --val-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval/chunk_valid.parquet' \
  --report-to "tensorboard" \
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
  --pretrained /mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt | tee ${EXP_NAME}.log
