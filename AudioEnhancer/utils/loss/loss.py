import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import angle

class SpectralLoss(nn.Module):
    def __init__(self,
                 gamma=0.3,
                 factor_magnitude=1000,
                 factor_complex=1000,
                 factor_under=1):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        '''
        :param input:  [B, 1, T, F, 2]
        :param target: [B, 1, T, F, 2]
        '''
        input = torch.complex(input[..., 0],input[..., 1])
        target = torch.complex(target[..., 0],target[..., 1])
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)

        tmp = (input_abs - target_abs).pow(2)  # noise power special
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m

        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss
