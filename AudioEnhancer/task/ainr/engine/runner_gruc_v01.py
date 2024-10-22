#encoding:utf-8
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

from task.ainr.dataset.data import AudioDataset
from task.ainr.dataset.fft import STFT
from task.ainr.dataset.utils import get_wavlist, load_wav
from net.gruc.gruc import GRUC

from utils.loss.loss import SpectralLoss

from utils.utils import load_yaml

class Runner(nn.Module):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = load_yaml(args.config)
        self.savedir = args.savedir
        self.device = 'cuda:0'
        self.print_step = 1000
        self.start_epoch = 0

        # model ==================================================================
        self.init_ckpt = args.init_ckpt
        self.model = self.get_model()

        # dataset ==================================================================
        self.dataset = AudioDataset(self.args, self.device)
        self.train_data_len = int(len(self.dataset) * 0.8)
        self.val_data_len = len(self.dataset) - self.train_data_len
        train_dataset, validate_dataset = torch.utils.data.random_split(self.dataset,[self.train_data_len, self.val_data_len])
        self.train_dataset = DataLoader(train_dataset,
                                  batch_size=self.args['batch_size'],
                                  num_workers=self.args['num_workers'],
                                  pin_memory=False,
                                  shuffle=True)
        self.val_dataset = DataLoader(validate_dataset,
                                  batch_size=self.args['batch_size'],
                                  num_workers=self.args['num_workers'],
                                  pin_memory=False)


        self.fft_fun = STFT(self.args, self.device)

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.mse_loss = nn.MSELoss()
        self.spec_loss = SpectralLoss()

    def get_model(self):
        model = GRUC().to(self.device)
        if self.init_ckpt is not None and os.path.exists(self.init_ckpt):
            checkpoint = torch.load(self.init_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']

        return {
            'model': model,
        }

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.model['model'].parameters(), lr=float(self.args['lr']), betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer

    def get_scheduler(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=1)
        return scheduler

    def adjust_learning_rate(self, gamma=0.99):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma

    def run_epoch(self, epoch, dataset, train_flag=True):
        sum_loss = 0
        dataset_len = len(dataset)

        for idx, (clean, noisy) in enumerate(dataset): # clean: [b, c, t] 时域
            noisy_freq = self.fft_fun.fft(noisy) # noisy_freq.shape torch.Size([2, 1, 501, 257, 2])
            clean_freq = self.fft_fun.fft(clean) # clean_freq.shape torch.Size([2, 1, 501, 257, 2])

            spec_out = self.model['model'](noisy_freq)  # [B, C, T, F, 2]

            # Cacl loss
            loss = self.spec_loss(spec_out, clean_freq)

            if train_flag:
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            sum_loss += loss.item()
            if idx > 0 and idx % self.print_step == 0:
                print(
                    f"epoch={epoch} | idx={idx}/{dataset_len} | lr {self.optimizer.param_groups[0]['lr']:.6f}| loss={sum_loss/(idx + 1):.3f} |"
                )
        return sum_loss / dataset_len

    def train(self):
        print('train start ==========================')

        best_loss = np.Inf
        for cur_epoch in range(self.start_epoch, self.args['epochs']):
            self.model['model'].train()
            train_loss = self.run_epoch(cur_epoch, self.train_dataset, train_flag=True)

            self.model['model'].eval()
            val_loss = self.run_epoch(cur_epoch, self.val_dataset, train_flag=False)

            pt_path = os.path.join(self.savedir, f'{cur_epoch}.pt')
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict': self.model['model'].state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
            }, pt_path)
            print('pt_path', pt_path)
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"Accepted | epoch={cur_epoch}/{self.args['epochs']} | best_loss={best_loss}\n")
            else:
                print(f"Rejected | epoch={cur_epoch}/{self.args['epochs']} | best_loss={best_loss}\n")
                # 降低学习率
                self.adjust_learning_rate()

    def infer(self):
        print('infer start ==========================')
        infer_save_dir = os.path.join(self.args['infer_save_dir'], f'epoch{str(self.start_epoch)}')
        os.makedirs(infer_save_dir, exist_ok=True)

        infer_data_len = self.args['sr'] * self.args['infer_data_len']
        wav_list = get_wavlist(self.args['infer_lst'])

        self.model.eval()
        for idx, wav_path in enumerate(wav_list):
            print(idx, wav_path)
            if idx > 100:
                break
            base_name = f'{os.path.basename(wav_path)[:-4]}_epoch{self.start_epoch}_dfnet.wav'

            noisy_origin = load_wav(wav_path, target_sr=self.args['sr']) # noisy torch.Size([1, 51403])
            noisy_origin = noisy_origin.to(self.device)
            # print('noisy', noisy.shape)
            if noisy_origin.shape[0] > 1:
                noisy_origin = noisy_origin[1, :].unsqueeze(0)

            pad_len = infer_data_len - noisy_origin.shape[1] % infer_data_len
            noisy = torch.cat((noisy_origin, torch.zeros(1, pad_len).to(self.device)), dim=-1) # [1, t]

            pred_list = []
            block = noisy.shape[1] // infer_data_len
            for i in range(block):
                noisy_t = noisy[:, i * infer_data_len: (i + 1) * infer_data_len]
                noisy_t = noisy_t.unsqueeze(0)  # torch.Size([1, 1, 80000])
                # print('noisy_t', noisy_t.shape)

                noisy_freq = self.fft_fun.fft(noisy_t)         # torch.Size([2, 1, 257, 501, 2])
                noisy_freq = noisy_freq.permute(0, 1, 3, 2, 4) # torch.Size([2, 1, 501, 257, 2])

                out = self.model(noisy_freq)
                pred = out[0] * noisy_freq      # torch.Size([2, 1, 501, 257, 2])
                pred = pred.permute(0, 1, 3, 2, 4)

                pred_t = self.fft_fun.ifft(pred) # torch.Size([2, 1, 257, 501, 2]) -> torch.Size([1, 1, 80000])
                pred_t = pred_t.squeeze(0)
                pred_list.append(pred_t)

            clean_hat = torch.cat(pred_list, dim=-1)
            clean_hat = clean_hat[:, :-pad_len] # [1, t]
            # print('clean_hat.shape', clean_hat.shape)

            save_path = os.path.join(infer_save_dir, base_name)
            print(save_path)
            sf.write(save_path, clean_hat.T.detach().cpu().numpy(), self.args['sr'])
            sf.write(os.path.join(infer_save_dir, os.path.basename(wav_path)), noisy_origin.T.detach().cpu().numpy(), self.args['sr'])

            # 得到干净的序列并保存
            clean_wav_path = wav_path.replace('noisy', 'clean')
            clean_wav_path2 = os.path.join(infer_save_dir, os.path.basename(clean_wav_path))[:-4] + '_clean.wav'
            clean_origin = load_wav(clean_wav_path, target_sr=self.args['sr'])
            sf.write(clean_wav_path2, clean_origin.T.detach().cpu().numpy(), self.args['sr'])







