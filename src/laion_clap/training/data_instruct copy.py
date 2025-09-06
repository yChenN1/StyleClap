import ast
import json
import logging
import math
import os
import random
import h5py
from dataclasses import dataclass
from datasets import load_dataset
import braceexpand
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms
import webdataset as wds
from PIL import Image
from torchaudio.transforms import Resample
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path
import wget
import tempfile
import copy
from torch.nn.utils.rnn import pad_sequence
from contextlib import suppress

from clap_module.utils import get_tar_path_from_dataset_name, dataset_split
from clap_module.utils import load_p, load_class_label
from clap_module import tokenize as clip_tokenizer
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import BartTokenizer

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenizer(text, tmodel="roberta", max_length=77):
    """tokenizer for different models
    tmodel is default to roberta as it is the best model for our task
    max_length is default to 77 from the OpenAI CLIP parameters
    We assume text to be a single string, but it can also be a list of strings
    """
    if tmodel == "transformer":
        return clip_tokenizer(text).squeeze(0)

    elif tmodel == "bert":
        result = bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

    elif tmodel == "roberta":
        result = roberta_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

    elif tmodel == "bart":
        result = bart_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}


def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')


def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def gender_mapping(gender):
    if gender == 'female':
        return 0
    elif gender == 'male':
        return 1
    else:
        raise ValueError(f"Unknown gender: {gender}")

def age_mapping(age):
    if age == 'Elderly':
        return 0
    elif age == 'Middle-aged':
        return 1
    elif age == 'Young Adult':
        return 2
    elif age == 'Teenager':
        return 3
    elif age == 'Child':
        return 4
    else:
        raise ValueError(f"Unknown age: {age}")
    
def speed_mapping(speed):
    if speed == 'slow':
        return 0
    elif speed == 'normal':
        return 1
    elif speed == 'fast':
        return 2
    else:
        raise ValueError(f"Unknown speed: {speed}")

def pitch_mapping(pitch):
    if pitch == 'low':
        return 0
    elif pitch == 'normal':
        return 1
    elif pitch == 'high':
        return 2
    else:
        raise ValueError(f"Unknown pitch: {pitch}")

def energy_mapping(energy):
    if energy == 'low':
        return 0
    elif energy == 'normal':
        return 1
    elif energy == 'high':
        return 2
    else:
        raise ValueError(f"Unknown energy: {energy}")
    
def emotion_mapping(emotion):
    if emotion == 'happy':
        return 0
    elif emotion == 'sad':
        return 1
    elif emotion == 'angry':
        return 2
    elif emotion == 'fearful':
        return 3
    elif emotion == 'disgusted':
        return 4
    elif emotion == 'surprised':
        return 5
    elif emotion == 'neutral' or emotion == 'natural':
        return 6
    else:
        raise ValueError(f"Unknown emotion: {emotion}")
    

# For Toy Dataset
class ToyDataset(Dataset):
    def __init__(self, data, model_cfg):
        """Toy Dataset for testing the audioset input with text labels
        Parameters
        ----------
            index_path: str
                the link to the h5 file of each audio
            idc: str
                the link to the npy file, the number of samples in each class
            config: dict
                the audio cfg file
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.data = data
        self.class_label = pd.read_csv("/mnt/bn/tanman-yg/chenqi/datas/gigaspeech/EN_labels.csv")
        self.class_label.set_index("Key", inplace=True)

        self.audio_cfg = model_cfg["audio_cfg"]
        # self.eval_mode = eval_mode

        logging.info("total dataset size: %d" % (len(self.data)))

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, frame_num - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis=0)
        return new_sample

    def generate_queue(self):
        self.queue = []
        while len(self.queue) < self.total_size:
            class_set = [*range(self.classes_num)]
            random.shuffle(class_set)
            self.queue += [
                self.ipc[d][random.randint(0, len(self.ipc[d]) - 1)] for d in class_set
            ]
        self.queue = self.queue[: self.total_size]

        logging.info("queue regenerated:%s" % (self.queue[-5:]))

    def crop_wav(self, x, crop_size=450000):
        # nealy 1024 after get mel
        if x.shape[0] <= crop_size:
            return x
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos: crop_pos + crop_size]
    
    def pad_mel(self, tensor_list, dim=1, pad_value=0.0):
        # 计算第1维的最大长度
        max_len = max(t.shape[dim] for t in tensor_list)
        padded = [F.pad(t, (0, 0, 0, max_len - t.shape[dim]), value=pad_value) for t in tensor_list]
        return torch.stack(padded)

    def prompt_text(self, target):
        text = tokenizer(target)
        return text

    def __getitem__(self, index):
        """Load waveform, text, and target of an audio clip

        Parameters
        ----------
            index: int
                the index number
        Return
        ------
            output: dict {
                "hdf5_path": str,
                "index_in_hdf5": int,
                "audio_name": str,
                "waveform": list (audio_length,),
                "target": list (class_num, ),
                "text": torch.tensor (context_length,)
            }
                the output dictionary
        """
        s_index = self.data[index]
        waveform = torch.tensor(s_index['src_audio']['array'])
        src_sr = s_index['src_audio']['sampling_rate']
        text = s_index['src_caption']
        audio_name = Path(s_index['src_audio']['path']).name
        target_sr = 16000
        if src_sr != target_sr:
            waveform = Resample(src_sr, target_sr)(waveform)
        waveform = int16_to_float32(waveform.numpy())
        # __import__('remote_pdb').set_trace()
        waveform = self.crop_wav(waveform)
        mel_spec = get_mel(torch.from_numpy(waveform), self.audio_cfg)[None, :, :]
        if mel_spec.shape[2] > 1024:
            print(waveform.shape, mel_spec.shape)
        mel_spec = torch.cat([mel_spec, mel_spec.clone(), mel_spec.clone(), mel_spec.clone()], dim=0).cpu().numpy()
        longer = random.choice([True, False])
        if longer == False:
            mel_spec[1:, :, :] = 0.0
        text = self.prompt_text(text)

        ### cls_label
        if self.class_label is not None:
            meta_info = self.class_label.loc[f'gigaspeech_{audio_name[:-4]}']
            gender = gender_mapping(meta_info['Gender'])
            age = age_mapping(meta_info['Age'])
            speed = speed_mapping(meta_info['Speed'])
            pitch = pitch_mapping(meta_info['Pitch'])
            energy = energy_mapping(meta_info['Energy'])
            emotion = emotion_mapping(meta_info['Emotion'])
            class_label = torch.tensor([gender, age, speed, pitch, energy, emotion])
            data_dict = {
                "waveform": waveform,
                "text": text,
                "longer": longer,
                "mel_fusion": mel_spec,
                "audio_name": audio_name,
                "caption": s_index['src_caption'],
                "class_label": class_label
            }
        else:
            data_dict = {
                "waveform": waveform,
                "text": text,
                "longer": longer,
                "mel_fusion": mel_spec,
                "audio_name": audio_name,
                "caption": s_index['src_caption']
            }
        return data_dict

    def collate_fn(self, batch):
        waveforms = [torch.tensor(item["waveform"]) for item in batch]
        texts = [item["text"] for item in batch]
        longers = [item["longer"] for item in batch]
        mel_fusions = [torch.tensor(item["mel_fusion"]) for item in batch]
        audio_name = [item["audio_name"] for item in batch]
        caption = [item["caption"] for item in batch]
        
        waveforms_padded = pad_sequence(waveforms, batch_first=True)  # shape: (B, max_len)
        mel_fusions_stacked = self.pad_mel(mel_fusions)
        input_ids = torch.stack([item["input_ids"] for item in texts])
        attention_mask = torch.stack([item["attention_mask"] for item in texts])
        text = {"input_ids": input_ids, "attention_mask": attention_mask}

        if 'class_label' in batch[0]:
            class_label = [item["class_label"] for item in batch]
            class_label = torch.stack(class_label)
            return {
                "waveform": waveforms_padded,             # (B, T_wav)
                "text": text,                     # (B, T_text)
                "longer": torch.tensor(longers),          # (B,)
                "mel_fusion": mel_fusions_stacked,        # (B, 4, mel_bins, T)
                "audio_name": audio_name,
                "caption": caption,
                "class_label": class_label
            }
        else:
            return {
                "waveform": waveforms_padded,             # (B, T_wav)
                "text": text,                     # (B, T_text)
                "longer": torch.tensor(longers),          # (B,)
                "mel_fusion": mel_fusions_stacked,        # (B, 4, mel_bins, T)
                "audio_name": audio_name,
                "caption": caption
            }

    def __len__(self):
        return len(self.data)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg['mel_bins'],
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    ).to(audio_data.device)
    
    mel = mel_tf(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=audio_cfg['mel_bins'],
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)

def get_toy_dataset(args, model_cfg, data_split, is_train):
    dataset = ToyDataset(data_split, model_cfg)

    num_samples = len(dataset)
    if is_train:
        sampler = (
            DistributedSampler(dataset, shuffle=True)
            if args.distributed and is_train
            else None
        )
    else:
        sampler = (
            DistributedSampler(dataset, shuffle=False)
            if args.distributed and is_train
            else None
        ) 

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, model_cfg):
    data = {}
    from torch.utils.data import Subset
    # Instantiate custom dataset (pass in tokenizer for prompt construction and tokenization)
    if args.train_data:
        data_split = load_dataset(
            'parquet',
            data_files={
                'train': [
                    args.train_data,
                ]
            },
            split='train',
        )
        train_dataset = get_toy_dataset(args, model_cfg, data_split, is_train=True)
        data["train"] = train_dataset

    if args.val_data:
        test_data_split = load_dataset(
            'parquet',
            data_files={
                'train': [
                    args.val_data,
                ]
            },
            split='train',
        )
        test_dataset = get_toy_dataset(args, model_cfg, test_data_split, is_train=False)
        # test_dataset = Subset(test_dataset, list(range(500)))
        data["val"] =test_dataset

    return data
