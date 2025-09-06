import numpy as np
from src import laion_clap
import torch
import pandas as pd
from torchaudio.transforms import Resample
import random

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def crop_wav(x, crop_size=450000):
    # nealy 1024 after get mel
    if x.shape[0] <= crop_size:
        x = np.concatenate([x, np.array([0.0] * (crop_size - len(x)))], axis=0).astype(np.float32)
        return x
    crop_pos = random.randint(0, len(x) - crop_size - 1)
    return x[crop_pos: crop_pos + crop_size]

def calculate_metrics(preds, gts):
    """
    计算总体准确率和每个类别的准确率
    
    参数:
        preds: 预测结果列表
        gts: 真实标签列表
        
    返回:
        包含总体准确率和每个类别准确率的字典
    """
    # 检查输入列表长度是否一致
    if len(preds) != len(gts):
        raise ValueError("预测列表和真实标签列表长度必须一致")
    
    # 获取所有唯一的类别
    classes = set(gts + preds)
    
    # 初始化计数器
    total = len(gts)
    correct = 0
    class_correct = {cls: 0 for cls in classes}  # 每个类别被正确预测的数量
    class_total = {cls: 0 for cls in classes}    # 每个类别的总样本数
    
    # 统计
    for pred, gt in zip(preds, gts):
        # 统计总体正确数
        if pred == gt:
            correct += 1
            class_correct[gt] += 1
        
        # 统计每个类别的总样本数
        class_total[gt] += 1
    
    # 计算总体准确率
    overall_acc = correct / total if total > 0 else 0
    
    # 计算每个类别的准确率
    class_acc = {}
    for cls in classes:
        # 避免除以零
        class_acc[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
    
    return {
        'overall_accuracy': overall_acc,
        'class_accuracy': class_acc
    }
    
model = laion_clap.CLAP_LP(enable_fusion=False, amodel='HTSAT-base')

model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0903style/2025_09_04-11_55_48-linear_probemodel_HTSAT-base-lr_0.0001-b_96-j_4-p_fp32/checkpoints/pretrain_epoch_15_lp_epoch_latest.pt')
# model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0809giga/2025_08_09-18_30_41-model_HTSAT-base-lr_5e-05-b_96-j_6-p_fp32/checkpoints/epoch_45.pt')
# model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt')


data_file = '/mnt/bn/tanman-yg/chenqi/code/StyleClap/experiment_scripts/tmp_valid.csv'
# data_file = '/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/LLaSA_training/data_instruction/EN_description.csv'
# data = pd.read_csv(data_file).to_dict(orient='records')
base_path = '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval_filtered'
data = pd.read_parquet(base_path).to_dict(orient='records')

gts = []
preds = []
input_type = 'from_feat'
for da in data:
    if input_type == 'from_feat':
        gt = int(da['emotion_code'])
        waveform = torch.tensor(da['src_audio']['array'])
        src_sr = da['src_audio']['sampling_rate']
        if src_sr != 16000:
            waveform = Resample(src_sr, 16000)(waveform)
        waveform = int16_to_float32(waveform.numpy())
        waveform = crop_wav(waveform)
        data_dict = {
                "waveform": torch.tensor(waveform).unsqueeze(0),
            }
        logits, pred = model.forward_with_feature(data_dict)
        preds.append(pred)
        gts.append(gt)
        metrics = calculate_metrics(preds, gts)
        print(metrics)
    elif input_type == 'from_path':
        source_audio = f"{base_path}/{da['audio_path']}"
        logits, pred = model([source_audio])
    

