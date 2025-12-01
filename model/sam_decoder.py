# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import sys
sys.path.append(".")
from math import log2
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from einops import rearrange, repeat
from loss.lossfunction import MMDLoss
from model.sam.transformer import TwoWayTransformer
from model.sam.common import LayerNorm2d
from kornia.filters import filter2d
from vector_quantize_pytorch import VectorQuantize
from functools import partial

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        # embed = F.normalize(embed, p=2, dim=0)
        self.register_buffer("embed", embed) # [D, M]
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        self.embed = F.normalize(self.embed, p=2, dim=0)
        flatten = input.reshape(-1, self.dim) # [N, D] D为向量维度，N为向量个数
        # 平方欧氏距离
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # 将embed_ind转换为one-hot编码，对应位置为1，其余位置为0， [N, M]
        embed_ind = embed_ind.view(*input.shape[:-1])
        # N, H, W, B = input.size()
        # embed_ind = embed_ind.view(N, H, W, -1)
        quantize = self.embed_code(embed_ind) # 构建映射关系
        _, furembed_ind = (-dist).min(1)
        furembed_ind = furembed_ind.view(*input.shape[:-1])
        furquantize = self.embed_code(furembed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0) #计算每个码本向量被选中的次数 (1, M)，即每个码本向量的使用频率
            embed_sum = flatten.transpose(0, 1) @ embed_onehot # 考虑频率的加权输入特征和，按照使用频率加权码本向量 (D, M) 

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)
            # EMA更新码本向量的使用频率，存储在cluster_size中，是旧的使用频率×decay+新的使用频率×(1-decay)
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            # 按照使用频率加权码本向量，存储在embed_avg中，是旧的使用频率×decay+新的使用频率×(1-decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        diff = diff + (furquantize.detach() - input).pow(2).mean()
        furquantize = input + (furquantize - input).detach()

        mix_quantize = (quantize + furquantize) / 2

        return mix_quantize, diff

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Quantize_proto(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.rand(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, istd=False):
        
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
            if istd:
                self.embed.data = torch.abs(self.embed.data)
        e_latent_loss = F.mse_loss(quantize.detach(), input)
        q_latent_loss = F.mse_loss(quantize, input.detach())
        diff = q_latent_loss + 1*e_latent_loss
        # diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, self.embed.data, diff

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class SegmentAnythingDecoder(nn.Module):
    def __init__(self, in_channels=256, nums=5):
        super(SegmentAnythingDecoder, self).__init__()
        # 扩展通道数以匹配类别数，同时减小特征图的空间尺寸
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, nums, kernel_size=3, padding=1)

        # 上采样层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64 -> 128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128 -> 256
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256 -> 512

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.up1(x)
        x = F.relu(self.conv2(x))
        x = self.up2(x)
        x = F.relu(self.conv3(x))
        x = self.up3(x)
        x = self.conv4(x)  # 此处不应用ReLU，因为这是最终的分类层
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 8,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes

        self.iou_token = nn.Embedding(num_classes, transformer_dim)
        #self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(num_classes*self.num_mask_tokens, transformer_dim)
        #self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks_1(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # masks = masks[:,int(torch.argmax(iou_pred.sum(0))),:,:] # 选择iou最大的mask 
        max_pred, max_index = torch.max(iou_pred, dim=1)
        batch_indices = torch.arange(masks.size(0)) 
        masks = masks[batch_indices, max_index]   
        iou_pred = iou_pred[batch_indices, max_index]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.squeeze(1)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_classes), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def predict_masks_1(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim) # 8*51*256
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1) # 8*52*256
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1) # 8*52*256

        # Expand per-image data in batch direction to be per-mask
        src =  torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # 8*256*64*64
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # 8*256*64*64

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens) # 8*52*256, 8*4096*256
        iou_token_out = hs[:, 0, :] # 8*256
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # 8*51*256

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MaskDecoder1(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 8,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_multimask_outputs = num_multimask_outputs
        
        self.iou_token = nn.Embedding(num_classes, transformer_dim)
        # self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(num_classes*self.num_mask_tokens, transformer_dim)
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # masks = masks[:,int(torch.argmax(iou_pred.sum(0))), :, :]
        # select the mask with the highest iou
        max_pred, max_index = torch.max(iou_pred, dim=1)
        batch_indices = torch.arange(masks.size(0)) 
        masks = masks[batch_indices, max_index]
        iou_pred = iou_pred[batch_indices, max_index]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Psparse_prompt_embeddingsredicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim)
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1)
        sparse_prompt_embeddings = sparse_prompt_embeddings.repeat(self.num_classes, 1, 1) # 8*52*256
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = output_tokens

        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    def predict_masks_1(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim)
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # Expand per-image data in batch direction to be per-mask
        src =  torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def exists(val):
    return val is not None

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            if i != depth - 1:
                layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])
            else:
                layers.extend([EqualLinear(emb, emb//2, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.max_grad_norm = 10
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)
        self.conv1x1 = nn.Conv2d(input_channels, filters, 1, 1, 0, bias=False)
        self.activation = leaky_relu()
        # self.dropout = nn.Dropout2d(0.1)
    
    def exists(self, val):
        return val is not None

    def forward(self, x, istyle, noise=False):
        if self.exists(self.upsample):
            x = self.upsample(x)
        if noise:
            inoise = torch.FloatTensor(x.shape[0], 1).uniform_(0., 0.01).to(x.device)
        else:
            inoise = torch.zeros(x.shape[0], 1).to(x.device)
        x_ = self.conv1x1(x)
        inoise = inoise.unsqueeze(2).unsqueeze(3)
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)
        # x_ = self.dropout(x_)
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x_ + x + noise2)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

def dual_contrastive_loss(real_logits, fake_logits):
        device = real_logits.device
        real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

        def loss_half(t1, t2):
            t1 = rearrange(t1, 'i -> i ()')
            t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
            t = torch.cat((t1, t2), dim = -1)
            return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

        return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

def gradient_penalty(images, output, weight = 10):
        batch_size = images.shape[0]
        gradients = torch.autograd.torch_grad(outputs=output, inputs=images,
                            grad_outputs=torch.ones(output.size(), device=images.device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.reshape(batch_size, -1)
        return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 32 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1, nn.ReLU())

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, q_block) in zip(self.blocks, self.quantize_blocks):
            x = block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = F.normalize(self.to_logit(x))
        return x.squeeze(), quantize_loss

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b n d, b n e -> b d e', k, v)
        out = torch.einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class StyleMaskDecoder1(nn.Module):
    def __init__(self, transformer_dim: int, transformer: nn.Module, num_classes: int):
        super(StyleMaskDecoder1, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_blocks = 3
        # Embedding for mask tokens
        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)
        self.mapping = StyleVectorizer(transformer_dim*2, 4)
        
        filters = transformer_dim//2
        input_channels = transformer_dim
        self.blocks = nn.ModuleList()
        self.quantize = Quantize(256, 2048)
        # for num in range(self.num_blocks):
        #     if num != self.num_blocks-1:
        #         block = GeneratorBlock(transformer_dim, input_channels, filters)
        #         self.blocks.append(block)
        #         input_channels = filters
        #         filters = max(filters//2, 1)
        #     else:
        #         block = GeneratorBlock(transformer_dim, input_channels, 1, upsample=False)
        #         self.blocks.append(block)
        self.D = Discriminator(image_size=512)
        for num in range(self.num_blocks):
            block = GeneratorBlock(transformer_dim, input_channels, filters)
            self.blocks.append(block)
            input_channels = filters
            filters = max(filters//2, 1)

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        # MLP for processing transformer's output
        self.mask_prediction_mlp = MLP(input_dim=transformer_dim, hidden_dim=transformer_dim // 2, output_dim=1, num_layers=3, sigmoid_output=True)

    def noise(n, latent_dim, device):
        return torch.randn(n, latent_dim).cuda(device)

    def image_noise(n, im_size, device):
        return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)
    
    def ad_loss(self, fake_output, real_output, rel_disc_loss=True):
        real_output_loss = real_output
        fake_output_loss = fake_output
        if rel_disc_loss:
            real_output_loss = real_output_loss - fake_output.mean()
            fake_output_loss = fake_output_loss - real_output.mean()
        divergence = dual_contrastive_loss(real_output_loss, fake_output_loss)
        return divergence
    
    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, multimask_output=True) -> torch.Tensor:
        b, c, h, w = image_embeddings.shape
        # Prepare mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0)        
        
        # Transformer processing
        hs, src = self.transformer(image_embeddings, image_pe, mask_tokens)
        # Process transformer output with MLP for mask prediction
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # mapping style 
        global_avg_pool = F.adaptive_avg_pool2d(src, (1, 1))
        global_avg_pool = global_avg_pool.view(src.size(0), -1)
        global_max_pool = F.adaptive_max_pool2d(src, (1, 1))
        global_max_pool = global_max_pool.view(src.size(0), -1)
        style_org = torch.cat([global_avg_pool, global_max_pool], dim=1)

        style_noise_scale = style_org.std() * torch.rand(1).to(style_org.device)
        noise = torch.randn_like(style_org) * style_noise_scale
        style = style_org + noise
        style_org = self.mapping(style_org)
        style = self.mapping(style)

        style, diff = self.quantize(style)

        for i, block in enumerate(self.blocks):
            if i == 0:
                src_gen = block(src, style, noise=True)
                src_org = block(src, style_org, noise=False)
            else:
                src_gen = block(src_gen, style, noise=True)
                src_org = block(src_org, style_org, noise=False)

        b, c, h, w = src_gen.shape
        hs = self.output_hypernetworks_mlps(hs.squeeze(0)).unsqueeze(1)
        masks = (hs @ src_gen.view(b, c, h * w)).view(b, -1, h, w).squeeze(0)
        masks_org = (hs @ src_org.view(b, c, h * w)).view(b, -1, h, w).squeeze(0)
        
        return masks, masks_org, diff, src_gen, src_org, style

class StyleMaskDecoder(nn.Module):
    def __init__(self, transformer_dim: int, transformer: nn.Module, num_classes: int):
        super(StyleMaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_blocks = 3
        self.concentration_coeff = 1.1
        self.ema_decay = 0.95
        self.num_proto = 8192
        self.batch_size = 4

        # Embedding for mask tokens
        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)
        self.mapping = StyleVectorizer(transformer_dim*2, 4)
        # self.mapping_gen = StyleVectorizer(transformer_dim*2, 4)
        self.concentration = torch.tensor([self.concentration_coeff]*self.num_proto)
        self.concentration_mean = torch.tensor([self.concentration_coeff]*self.batch_size)
        self.concentration_std = torch.tensor([self.concentration_coeff]*self.batch_size)

        self._dirichlet = torch.distributions.dirichlet.Dirichlet(concentration=self.concentration)
        self._dirichlet_mean = torch.distributions.dirichlet.Dirichlet(concentration=self.concentration_mean)
        self._dirichlet_std = torch.distributions.dirichlet.Dirichlet(concentration=self.concentration_std)
        filters = transformer_dim//2
        input_channels = transformer_dim
        self.blocks = nn.ModuleList()
        self.blocks_gen = nn.ModuleList()
        # self.quantize = Quantize(256, 20480)
        self.quantize_mean = Quantize_proto(256, self.num_proto)
        self.quantize_std = Quantize_proto(256, self.num_proto)
        
        attn_and_ff = lambda chan: nn.Sequential(*[
            Residual(PreNorm(chan, LinearAttention(chan))),
            Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
        ])

        self.attns = nn.ModuleList([])
        self.attns_gen = nn.ModuleList([])

        self.prolayers = nn.Sequential(
            EqualLinear(transformer_dim, transformer_dim, lr_mul=1.0),  # 假设 emb 是输入和输出维度的参数
            nn.LeakyReLU()
        )
        
        self.MMDLoss = MMDLoss()
        for num in range(self.num_blocks):
            block = GeneratorBlock(transformer_dim, input_channels, filters)
            block_gen = GeneratorBlock(transformer_dim, input_channels, filters)
            attn_fn = attn_and_ff(input_channels)
            attn_fn_gen = attn_and_ff(input_channels)
            self.blocks.append(block)
            self.blocks_gen.append(block_gen)
            self.attns.append(attn_fn)
            self.attns_gen.append(attn_fn_gen)
            input_channels = filters
            filters = max(filters//2, 1)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            LayerNorm2d(transformer_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            LayerNorm2d(transformer_dim // 8),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.output_hypernetworks_mlps_style = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        # MLP for processing transformer's output
        self.mask_prediction_mlp = MLP(input_dim=transformer_dim, hidden_dim=transformer_dim // 2, output_dim=1, num_layers=3, sigmoid_output=True)

    def update_ema(self, ema_buffer, value, decay):
        # Update the EMA buffer with a new value
        ema_buffer *= decay
        ema_buffer += (1 - decay) * value
        return ema_buffer
    
    def style_samplegen(self, x):
        B, C, H, W = x.size()
        x_mean = x.mean(dim=[2,3], keepdim=True) # B,C,1,1
        x_std = x.std(dim=[2,3], keepdim=True) + 1e-7 # B,C,1,1
        x_norm = ((x - x_mean) / x_std).detach()
        # x_mean, x_std = x_mean.detach(), x_std.detach() 
 
        # Sample combine weights from the Dirichlet distribution
        combine_weights = self._dirichlet.sample((B,)).to(x.device)  # B,C
        combine_weights = combine_weights.detach()
        if B == self.batch_size:
            combine_weights_mean = self._dirichlet_mean.sample((B,)).to(x.device)
            combine_weights_mean = combine_weights_mean.detach()
            combine_weights_std = self._dirichlet_std.sample((B,)).to(x.device)
            combine_weights_std = combine_weights_std.detach()
        else:
            dirichlet_mean = torch.tensor([self.concentration_coeff]*B)
            dirichlet_std = torch.tensor([self.concentration_coeff]*B)
            combine_weights_mean = torch.distributions.dirichlet.Dirichlet(concentration=dirichlet_mean).sample((B,)).to(x.device)
            combine_weights_mean = combine_weights_mean.detach()
            combine_weights_std = torch.distributions.dirichlet.Dirichlet(concentration=dirichlet_std).sample((B,)).to(x.device)
            combine_weights_std = combine_weights_std.detach()

        x_mean = combine_weights_mean @ x_mean.squeeze(-1).squeeze(-1) # B,C
        x_std = combine_weights_std @ x_std.squeeze(-1).squeeze(-1)
      
        x_mean, proto_mean, diff_mean = self.quantize_mean(x_mean, istd=False)
        x_std, proto_std, diff_std = self.quantize_std(x_std, istd=True)

        new_mean = combine_weights @ proto_mean.permute(1, 0) # B,C
        new_std = combine_weights @ proto_std.permute(1, 0)
 
        x_new = x_norm * new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)
        x_new = (x_new - x).detach() + x
        x_new2 = x_norm * x_std.unsqueeze(-1).unsqueeze(-1) + x_mean.unsqueeze(-1).unsqueeze(-1)
        x_newmix = (x_new + x_new2)/2
        return x_newmix, diff_mean+diff_std
    
    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, multimask_output=True) -> torch.Tensor:
        b, c, h, w = image_embeddings.shape
        # Prepare mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0)        
        diff_sum = 0.
        # Transformer processing
        hs, src = self.transformer(image_embeddings, image_pe, mask_tokens)
        # Process transformer output with MLP for mask prediction
        src = src.transpose(1, 2).view(b, c, h, w)
        
        for i, block in enumerate(self.blocks):
            x_new, diff = self.style_samplegen(src)

            global_avg_pool = F.adaptive_avg_pool2d(x_new, (1, 1))
            global_avg_pool = global_avg_pool.view(x_new.size(0), -1)
            global_max_pool = F.adaptive_max_pool2d(x_new, (1, 1))
            global_max_pool = global_max_pool.view(x_new.size(0), -1)
            style = torch.cat([global_avg_pool, global_max_pool], dim=1)

            style_mix = self.mapping(style)
            style_mix = self.prolayers(style_mix)
            # style_mix, diff = self.quantize(style_mix)
            if i == 0:
                # src_new = attn(src)
                src_new = block(src, style_mix, noise=False)
            else:
                # src_new = attn(src_new)
                src_new = block(src_new, style_mix, noise=False)
            diff_sum += diff
        
        diff_sum = diff_sum/len(self.blocks)
        b, c, h, w = src_new.shape
        upscaled_embedding = self.output_upscaling(src)
        # upscaled_embedding = F.interpolate(upscaled_embedding, (h, w), mode="bilinear",align_corners=False)
        hs_org = self.output_hypernetworks_mlps(hs.squeeze(0))
        hs_style = self.output_hypernetworks_mlps_style(hs.squeeze(0))
        masks = (hs_style @ src_new.view(b, c, h * w)).view(b, -1, h, w)
        masks_org = (hs_org @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks, masks_org, diff_sum

if __name__ == "__main__":
    #   D = Discriminator(image_size=512)
    #   image_embeddings = torch.randn(1, 32, 512, 512)
    #   x1, x2 = D(image_embeddings)
    decoder = StyleMaskDecoder(256, transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ), num_classes=8)
    image_embeddings = torch.randn(2, 256, 64, 64)
    image_pe = torch.randn(2, 256, 64, 64)
    mask_predictions = decoder(image_embeddings, image_pe)
    