#!/bin/bash

CURR_DIR=$(pwd)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache

# 切换到目标执行目录
cd ../src/laion_clap

python evaluate/eval_linear_probe.py \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="toy" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=1 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=8 \
    --use-bn-sync \
    --freeze-text \
    --amodel HTSAT-base \
    --tmodel roberta \
    --val-data '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval_filtered/*.parquet' \
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
    --optimizer "adam" \
    --resume "/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0903style/2025_09_04-11_55_48-linear_probemodel_HTSAT-base-lr_0.0001-b_96-j_4-p_fp32/checkpoints/pretrain_epoch_15_lp_epoch_latest.pt"