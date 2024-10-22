#encoding:utf-8
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torchaudio

def gan_list_from_dir(dir_path, shuffix='.wav'):
    lst_dir = os.path.dirname(dir_path)
    lst_name = os.path.basename(dir_path).split('.')[0] + '.lst'
    lst_path = os.path.join(lst_dir, lst_name)
    print(lst_path)
    with open(lst_path, 'w') as f:
        for root, folder_list, file_list in os.walk(dir_path):
            for file_ in file_list:
                if not file_.endswith(shuffix):
                    continue
                wav_path = os.path.join(root, file_)
                f.write(wav_path + '\n')

def load_wav(wav_path, target_sr=16000):
    if isinstance(wav_path, np.ndarray):
        wav_path = torch.from_numpy(wav_path)
    clean, sr = torchaudio.load(wav_path)  # [channel, time]
    clean = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(clean)
    return clean

def get_wavlist(wavlist):
    if wavlist is None:
        return None
    wavlist = wavlist.strip().split(',')

    path_list = []
    for lst_path in wavlist:
        f = open(lst_path, "r")
        lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
        for line in lines:
            path_list.append(line.strip('\n'))
    return sorted(path_list)


if __name__ == '__main__':
    wav_path_list = r'D:\Anpj\100_data\Demand\DS_10283_2791\noisy_trainset_56spk_wav'
    gan_list_from_dir(wav_path_list)