#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from tqdm import tqdm

parquet_files = list(Path('/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval/').glob('*.parquet'))
print(parquet_files)
df_list = [pd.read_parquet(file) for file in parquet_files]
data = pd.concat(df_list, ignore_index=True)
data['name'] = data['src_audio'].apply(lambda x: Path(x['path']).name if isinstance(x, dict) and 'path' in x else None)
data = data.set_index('name')


df = pd.read_csv("/mnt/bn/tanman-yg/chenqi/code/StyleClap/experiment_scripts/tmp_valid.csv")
# categories = ['confusion', 'disgust', 'enunciated', 'laughter', 'sad', 'whisper',
#  'disappointment', 'fear', 'cuteness', 'serenity', 'relief', 'happy',
#  'amazement', 'anger', 'adoration', 'neutral', 'pride', 'distress',
#  'contentment', 'amusement', 'realization', 'guilt', 'desire', 'pain',
#  'interest', 'ecstasy', 'embarrassment', 'singing']
# category_mapping = {category: i for i, category in enumerate(categories)}
# df['emotion_code'] = df["src_emo_norm"].map(category_mapping)

subset = df[df["emotion_code"].notna()]
emotion_mapping = dict(zip(subset['audio_path'], subset['emotion_code']))
data['emotion_code'] = data.index.map(emotion_mapping)
data_filtered = data[data['emotion_code'].notna()]

output_dir = Path("/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval_filtered/")
output_dir.mkdir(exist_ok=True)  # 确保输出目录存在

chunk_size = 2000
total_chunks = (len(data_filtered) + chunk_size - 1) // chunk_size  # 计算总块数

print(f"过滤后数据量: {len(data_filtered)}")
print(f"将分割为 {total_chunks} 个文件，每个最多 {chunk_size} 条数据")

for i in range(total_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(data_filtered))
    
    chunk = data_filtered.iloc[start_idx:end_idx]
    
    # 生成文件名
    output_file = output_dir / f"filtered_chunk_{i+1:03d}_of_{total_chunks:03d}.parquet"
    
    # 保存当前块
    chunk.to_parquet(output_file)
    print(f"已保存: {output_file} (包含 {len(chunk)} 条数据)")

print("所有文件保存完成!")


