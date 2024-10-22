#encoding:utf-8

from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn


class GRUC(nn.Module):
    def __init__(self, num_layer=3, hidden=320):
        super(GRUC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(),
        )
        self.linear1 = nn.Linear(in_features=257 * 2, out_features=320)
        self.linear2 = nn.Linear(in_features=320, out_features=257)
        self.gruc = nn.GRU(input_size=320, hidden_size=320, num_layers=3)

    def forward(self, spec: Tensor,):
        b, c, t, f, _ = spec.shape
        spec2 = torch.view_as_complex(spec) # [b, 1, t, f]
        spec_mag = torch.abs(spec2)
        x = self.conv1(spec_mag)   # [b, 2, t, f]
        x = x.permute(0, 2, 3, 1)  # [b, t, f, 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.linear1(x) # [b, t, 320]
        x, h = self.gruc(x)
        x = self.linear2(x)
        x = torch.tanh(x)  # [B, T, 257]
        mask = x.unsqueeze(1).unsqueeze(-1)
        out = spec * mask
        return out





if __name__ == '__main__':
    input = torch.randn(4, 1, 100, 257, 2)  # [B, 1, T, F, 2]

    model = GRUC()
    mask = model(input)
    print(mask.shape)  # torch.Size([4, 1, 100, 257, 2])

    from thop import profile
    macs, params = profile(model, inputs=(input,), verbose=False)
    print(f"macs = {macs / 1e9}G")
    print(f"params = {params / 1e6}M")

    # from libdf import unit_norm
    # spec_feat = torch.as_tensor(unit_norm(torch.view_as_complex(input).squeeze(1).numpy()[..., :96], 1.0))
    # print(spec_feat.shape)
    #
    # widths = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5,
    #                    6, 6, 7, 9, 9, 10, 11, 13, 14, 16, 18, 20, 22, 25, 29], dtype=np.uint64)
    # spec, erb_feat, spec_feat = df_features(input, widths)
    # print(spec.shape, erb_feat.shape, spec_feat.shape)

