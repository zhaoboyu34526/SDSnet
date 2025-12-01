#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   learnable_sam.py
    @Time    :   2023/08/15 10:41:31
    @Author  :   12718 
    @Version :   1.0
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None, no_transpose=True, dim=None) -> None:
        """
            One type of adapter introduced in 
            "Learnable Ophthalmology SAM"<https://arxiv.org/abs/2304.13425>
        Args:
            blk (Union[nn.Module, Block]): The Vision Transformer Block
            reduction (int, optional): The reduction rate. Defaults to 4.
            cls_token (bool, optional): Whether use the class token in the block. Defaults to False.
            reshape (bool, optional): Whether needs to reshape. Defaults to False.
            seq_size ([type], optional): The length of token. Defaults to None.
            no_transpose (bool, optional): Whether need to transpose the output. Defaults to False.
            dim ([type], optional): The dimension of the input. Defaults to None.
        """
        super(PromptGen, self).__init__()
        self.block = blk
        dim = dim or blk.dim
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )
        self.no_transpose = no_transpose
        self.cls_token = cls_token
        if self.cls_token:
            self.prompt_cls = nn.Sequential(
                nn.Linear(dim, prompt_dim),
                nn.LayerNorm(prompt_dim),
                nn.GELU(),
                nn.Linear(prompt_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
        self.reshape = reshape
        self.seq_size = seq_size
    
    def forward(self, x:Tensor) -> Tensor:
        """AI is creating summary for forward

        Args:
            x (Tensor): The input features.

        Returns:
            Tensor: The features extracted by the block
        """
        if self.cls_token:
            tokens = x[:,1:]
            cls_token = x[:, 0].unsqueeze(1)
            # cls_token = self.prompt_cls(cls_token)
            bs, seq_len, dim = tokens.size()
            if self.reshape:
                tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
            prompt = self.prompt_learn(tokens)
            promped = tokens + prompt
            if self.reshape:
                promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
            promped = torch.cat([cls_token, promped], dim=1)
        else:
            if self.no_transpose:
                prompt = self.prompt_learn(x)
            else:
                prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            promped = x + prompt
        net = self.block(promped)
        return net

class MSConv2d(nn.Module):
    def __init__(self, in_ch, groups=4):
        """

        Args:
            in_ch (int): number of channels for input
            groups (int, Optional): Number of groups, Defatults to 4.
        """
        super(MSConv2d, self).__init__()
        assert in_ch % groups == 0
        group_ch = in_ch // groups
        self.group_ch = group_ch
        self.conv = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1, 0)
        ])
        for i in range(1, groups):
            self.conv.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, bias=False)
            )
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        groups = torch.split(x, self.group_ch, dim=1)
        features = []
        for i, group in enumerate(groups):
            features.append(self.conv[i](group))
        features = torch.cat(features, dim=1)
        features = self.bn(features)
        features += x
        features = self.relu(features)
        return features

class FFTPrompt(nn.Module):
    def __init__(self, rate=0.25, prompt_type="highpass") -> None:
        super(FFTPrompt, self).__init__()
        assert prompt_type in ["highpass", "lowpass"], "The prompt type must in " \
        "['highpass', 'lowpass'], but got {}".format(prompt_type)
        self.rate = rate
        self.prompt_type = prompt_type
    
    def forward(self, x):
        fft = torch.fft.fft2(x, norm="forward")
        fft = torch.fft.fftshift(fft)
        h, w = x.shape[2:]
        radio = int((h*w*self.rate)**.5 // 2)
        mask = torch.zeros_like(x)
        c_h, c_w = h // 2, w // 2
        mask[:, :, c_h-radio:c_h+radio, c_w-radio:c_w+radio] = 0
        if self.prompt_type == "highpass":
            fft = fft*(1-mask)
        else:
            fft = fft * mask
        real, imag = fft.real, fft.imag
        shift = torch.fft.fftshift(torch.complex(real, imag))
        inv = torch.fft.ifft2(shift, norm="forward")
        inv = inv.real
        return torch.abs(inv)

class PromptSAM(nn.Module):
    def __init__(self, in_ch, image_encoder, num_classes=8, upsample_times=3, groups=4, 
                 prompt_input=False, prompt_type="fft", fft_type="highpass", freq_num=0.25, ms=False) -> None:
        super(PromptSAM, self).__init__()
        #load same from the pretrained model
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        out_dim = self.image_encoder.neck[0].out_channels
        self.img_size = self.image_encoder.img_size
        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    # nn.Conv2d(dim, dim // 2, 1, 1, 0),
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )
        
        if prompt_input:
            if prompt_type == "fft":
                self.prompt_input = FFTPrompt(rate=freq_num, prompt_type=fft_type)
        else:
            self.prompt_input = nn.Identity()

    def upscale(self, x, times=2):
        for i in range(times):
            # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x
    
    def forward(self, x):
        x = self.channel_conv(x)
        x = self.preprocess(x) 
        x = self.prompt_input(x)
        out = self.image_encoder(x)
        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        seg_out = self.decoder(out)
        return seg_out

if __name__ == "__main__":
    decoder = PromptSAM(model_name='vit_t')
    image_embeddings = torch.randn(4, 4, 256, 256)
    mask_predictions = decoder(image_embeddings)