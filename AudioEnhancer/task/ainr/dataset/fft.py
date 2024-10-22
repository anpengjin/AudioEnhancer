#encoding:utf-8


import os

import torch
import torch.nn as nn
import torchaudio

class STFT(nn.Module):
    def __init__(self, args, device):
        super(STFT, self).__init__()
        self.sr = args['sr']
        self.win_length = args['win_length']
        self.hop_length = args['hop_length']
        self.n_fft = args['n_fft']

        self.window = torch.hann_window(self.win_length).to(device)

    def fft(self, input):
        # input(tensor): [b, c, t]
        b, c, t = input.shape
        input_ = input.reshape(b * c, t)
        freq_ = torch.stft(input=input_, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window)
        freq = freq_.reshape(b, c, freq_.shape[1], freq_.shape[2], freq_.shape[3])
        return freq.permute(0, 1, 3, 2, 4)

    def ifft(self, input):
        # input(tensor): [b, c, f, t, 2]
        b, c, f, t, _ = input.shape
        input_ = input.reshape(b * c, f, t, 2)
        freq_ = torch.istft(input=input_, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window) # [b, c, t]
        out = freq_.reshape(b, c, freq_.shape[1])
        return out
