# import numpy as np
# import librosa
# import torch
# import laion_clap

# # quantization
# def int16_to_float32(x):
#     return (x / 32767.0).astype('float32')


# def float32_to_int16(x):
#     x = np.clip(x, a_min=-1., a_max=1.)
#     return (x * 32767.).astype('int16')

# model = laion_clap.CLAP_Module(enable_fusion=False)
# model.load_ckpt() # download the default pretrained checkpoint.

# # Directly get audio embeddings from audio files
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Directly get audio embeddings from audio files, but return torch tensor
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get text embedings from texts:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data)
# print(text_embed)
# print(text_embed.shape)

# # Get text embedings from texts, but return torch tensor:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data, use_tensor=True)
# print(text_embed)
# print(text_embed.shape)


import numpy as np
import librosa
import torch
from src import laion_clap
import pandas as pd

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

model = laion_clap.CLAP_LP(enable_fusion=False, amodel='HTSAT-base')

model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0903style/2025_09_04-11_55_48-linear_probemodel_HTSAT-base-lr_0.0001-b_96-j_4-p_fp32/checkpoints/pretrain_epoch_15_lp_epoch_latest.pt')
# model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0809giga/2025_08_09-18_30_41-model_HTSAT-base-lr_5e-05-b_96-j_6-p_fp32/checkpoints/epoch_45.pt')
# model.load_ckpt('/mnt/bn/tanman-yg/chenqi/code/StyleClap/pretrained/music_speech_audioset_epoch_15_esc_89.98.pt')


data_file = '/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/LLaSA_training/data_instruction/EN_description.csv'
data = pd.read_csv(data_file).to_dict(orient='records')
base_path = '/mnt/bn/tanman-yg/chenqi/datas/gigaspeech/merged_chunks'

for da in data:
    __import__('pdb').set_trace()
    source_audio = f"{base_path}/{da['Key'].replace('gigaspeech_', '') + '.wav'}"
    # source_audio = f"{base_path}/{da['audio_path']}"
    # source_caption = da['caption']
    # target_audio = da['gen_speech_path']
    source_caption = da['Des']
    # audio_embed = model.get_audio_embedding_from_filelist(x = [source_audio], use_tensor=False)
    
    audio_data, _ = librosa.load(source_audio, sr=48000) # sample rate should be 48000
    audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
    audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)

    text_embed = model.get_text_embedding([source_caption])
    # text_embed = model.get_text_embedding([data[0]['Des'], data[1]['Des'], data[2]['Des'], data[3]['Des'], data[4]['Des'], data[5]['Des'], data[6]['Des'], data[7]['Des'], data[8]['Des'], data[9]['Des']])
    print(torch.cosine_similarity(torch.tensor(audio_embed), torch.tensor(text_embed), dim=1))


assert False