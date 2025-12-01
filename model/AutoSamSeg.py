from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sam.prompt_encoder import PositionEmbeddingRandom


class AutoSamSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size=1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Conv2d(4, 3, 1)
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.pe_layer = PositionEmbeddingRandom(128)

    def forward(self, x):
        # x = self.conv(x)
        x = x[:, 1:, :, :]
        original_size = x.shape[-1]
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        image_embedding = self.image_encoder(x) # [B, 256, 64, 64]
        img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
        # 位置信息和批次无关，所以只需要一个位置编码
        masks, iou_pred = self.mask_decoder(image_embeddings=image_embedding.unsqueeze(1),
                                           image_pe=img_pe, multimask_output=True)
       
        if masks.shape[-1] != original_size:
            masks = F.interpolate(
                masks,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        return masks, iou_pred

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out
