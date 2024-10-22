#encoding:utf-8
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio

from task.ainr.dataset.utils import load_wav, get_wavlist

rand_state = np.random

class AudioDataset(Dataset):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.sr   = args['sr']
        self.data_len = args['data_len'] * args['sr']

        self.noisy_list  = get_wavlist(self.args['noisy_list'])
        self.clean_list  = get_wavlist(self.args['clean_list'])
        self.noise_list  = get_wavlist(self.args['noise_list'])
        self.babble_list = get_wavlist(self.args['babble_list'])
        self.rir_list    = get_wavlist(self.args['rir_list'])


    def __getitem__(self, index) -> Tuple[str, Tensor, int]:
        if self.noisy_list is not None: # 数据已经混好
            i = rand_state.randint(0, len(self.clean_list))
            clean = load_wav(self.clean_list[i], target_sr=self.sr) # [channel, time]
            noisy = load_wav(self.noisy_list[i], target_sr=self.sr) # [channel, time]
            assert clean.shape == noisy.shape

            c, t = clean.shape
            if t > self.data_len:
                j = rand_state.randint(0, t - self.data_len)
                clean = clean[:, j:j+self.data_len]
                noisy = noisy[:, j:j + self.data_len]
            else:
                pad = torch.zeros(c, self.data_len - t)
                clean = torch.cat((clean, pad), dim=-1)
                noisy = torch.cat((noisy, pad), dim=-1)

            return clean.to(self.device), noisy.to(self.device)

    def __len__(self):
        return len(self.clean_list)