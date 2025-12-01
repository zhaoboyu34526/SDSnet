# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union
from .tiny_vit_sam import TinyViT
from .image_encoder import ImageEncoderViT
from model.sam_decoder import MaskDecoder1, MaskDecoder, SegmentAnythingDecoder, StyleMaskDecoder
from .prompt_encoder import PromptEncoder
from model.sam.prompt_encoder import PositionEmbeddingRandom
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np
from copy import deepcopy
from util.util import promt_generate

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
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

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # if "point_coords" in image_record:
            #     points = (image_record["point_coords"], image_record["point_labels"])
            # else:
            #     points = None
            # points = None
            # sparse_embeddings, dense_embeddings = self.prompt_encoder(
            #     points=points,
            #     boxes=image_record.get("boxes", None),
            #     masks=image_record.get("mask_inputs", None),
            # )
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=curr_embedding.unsqueeze(0),
                                           image_pe=img_pe, multimask_output=True)
            # masks = self.postprocess_masks(
            #     low_res_masks,
            #     # input_size=image_record["image"].shape[-2:],
            #     original_size=original_size,
            # )
            outputs.append(low_res_masks)
            # masks = masks > self.mask_threshold
            # outputs.append(
            #     {
            #         "masks": masks,
            #         "iou_predictions": iou_predictions,
            #         "low_res_logits": low_res_masks,
            #     }
            # )
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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

class mobilesam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        mask_decoder: SegmentAnythingDecoder,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device
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
    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        outputs = self.mask_decoder(image_embeddings)
      
        return outputs


class samadaptor(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        mask_decoder: MaskDecoder1,
        img_size=1024,
    ) -> None:
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(128)
        self.no_mask_embed = nn.Embedding(1, 256)

    @property
    def device(self) -> Any:
        return self.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        bs, C, H, W = image_embeddings.size()
        outputs = []
        sparse_embeddings = torch.empty((1, 1, C), device=batched_input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, H, W)
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=curr_embedding.unsqueeze(0),
                                           image_pe=img_pe, sparse_prompt_embeddings=sparse_embeddings,
                                            dense_prompt_embeddings=dense_embeddings,
                                            multimask_output=True,
                                        )
            outputs.append(low_res_masks)
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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

class Sam_test(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder1,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        prompt,
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        H, W = batched_input.size()[-2:]
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(prompt, image_embeddings):
            # Transform input prompts
            if "point_coords" in image_record:
                point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                point_coords = self.transform.apply_coords(point_coords, (H, W))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                points = (coords_torch, labels_torch)
            if "boxes" in image_record:
                box = image_record["boxes"]
                box = self.transform.apply_boxes(box, (H, W))
                box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                box = box[None, :]
            if "mask_inputs" in image_record:
                masks = image_record["mask_inputs"]
                masks = torch.as_tensor(masks, dtype=torch.float, device=batched_input.device)
                masks = masks[None, :, :, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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
    
class Sam_muti_deco(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder_1: MaskDecoder,
        mask_decoder_2: MaskDecoder1,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder_1 = mask_decoder_1
        self.mask_decoder_2 = mask_decoder_2
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        args: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        B, _, H, W = batched_input.size()
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        
        outputs0 = []
        outputs = []

        # 无提示decoder
        for curr_embedding in zip(image_embeddings):
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder_1(image_embeddings=curr_embedding[0].unsqueeze(0),
                                          image_pe=img_pe, multimask_output=True)
            outputs0.append(low_res_masks)
        outputs0 = torch.stack(outputs0, dim=0)
        outputs0 = self.postprocess_masks(
            outputs0,
            original_size=original_size,
        )       
        # tgt_out_maxvalue, tgt_out = torch.max(outputs0, dim=1)
        # for i in range(args.n_class):
        #     tgt_out[(tgt_out_maxvalue < args.threshold) * (tgt_out == i)] = 255

        tgt_out_maxvalue, _ = torch.max(outputs0, dim=1)
        flattened_maxvalues = tgt_out_maxvalue.view(B, -1)
        sorted_values, _ = torch.sort(flattened_maxvalues, descending=True, dim=1)
        num_elements = flattened_maxvalues.shape[1]
        threshold_idx = int(num_elements * (1-args.threshold))
        thresholds = sorted_values[:, threshold_idx].view(-1, 1, 1)
        tgt_out = torch.where(tgt_out_maxvalue < thresholds, torch.full_like(tgt_out_maxvalue, 255), torch.full_like(tgt_out_maxvalue, 0))

        prompt = promt_generate(tgt_out, is_train=False)
        # 有提示decoder
        for image_record, curr_embedding in zip(prompt, image_embeddings):
            # Transform input prompts
            if "point_coords" in image_record:
                point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                point_coords = self.transform.apply_coords(point_coords, (H, W))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                points = (coords_torch, labels_torch)
            if "boxes" in image_record:
                box = image_record["boxes"]
                box = self.transform.apply_boxes(box, (H, W))
                box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                box = box[None, :]
            if "mask_inputs" in image_record:
                masks = image_record["mask_inputs"]
                masks = torch.as_tensor(masks, dtype=torch.float, device=batched_input.device)
                masks = masks[None, :, :, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder_2(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs0, outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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
    

class Samvq_muti_deco(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder_1: MaskDecoder,
        mask_decoder_2: MaskDecoder1,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder_1 = mask_decoder_1
        self.mask_decoder_2 = mask_decoder_2
        self.img_size = img_size
        self.quantize_t = Quantize(256, 512)
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        args: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        B, _, H, W = batched_input.size()
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        quant_t = image_embeddings.permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        outputs0 = []
        outputs = []

        # 无提示decoder
        for curr_embedding in zip(image_embeddings):
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder_1(image_embeddings=curr_embedding[0].unsqueeze(0),
                                          image_pe=img_pe, multimask_output=True)
            outputs0.append(low_res_masks)
        outputs0 = torch.stack(outputs0, dim=0)
        outputs0 = self.postprocess_masks(
            outputs0,
            original_size=original_size,
        )       
        # tgt_out_maxvalue, tgt_out = torch.max(outputs0, dim=1)
        # for i in range(args.n_class):
        #     tgt_out[(tgt_out_maxvalue < args.threshold) * (tgt_out == i)] = 255

        tgt_out_maxvalue, _ = torch.max(outputs0, dim=1)
        flattened_maxvalues = tgt_out_maxvalue.view(B, -1)
        sorted_values, _ = torch.sort(flattened_maxvalues, descending=True, dim=1)
        num_elements = flattened_maxvalues.shape[1]
        threshold_idx = int(num_elements * (1-args.threshold))
        thresholds = sorted_values[:, threshold_idx].view(-1, 1, 1)
        tgt_out = torch.where(tgt_out_maxvalue < thresholds, torch.full_like(tgt_out_maxvalue, 255), torch.full_like(tgt_out_maxvalue, 0))

        prompt = promt_generate(tgt_out, is_train=False)
        # 有提示decoder
        for image_record, curr_embedding in zip(prompt, image_embeddings):
            # Transform input prompts
            if "point_coords" in image_record:
                point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                point_coords = self.transform.apply_coords(point_coords, (H, W))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                points = (coords_torch, labels_torch)
            if "boxes" in image_record:
                box = image_record["boxes"]
                box = self.transform.apply_boxes(box, (H, W))
                box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                box = box[None, :]
            if "mask_inputs" in image_record:
                masks = image_record["mask_inputs"]
                masks = torch.as_tensor(masks, dtype=torch.float, device=batched_input.device)
                masks = masks[None, :, :, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder_2(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs0, outputs, diff_t

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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

class Sam_muti_newdeco(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder_1: MaskDecoder,
        mask_decoder_2: MaskDecoder1,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder_1 = mask_decoder_1
        self.mask_decoder_2 = mask_decoder_2
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        args: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        B, _, H, W = batched_input.size()
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        
        outputs0 = []
        outputs = []

        # 无提示decoder
        for curr_embedding in zip(image_embeddings):
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder_1(image_embeddings=curr_embedding[0].unsqueeze(0),
                                          image_pe=img_pe, multimask_output=True)
            outputs0.append(low_res_masks)
        outputs0 = torch.stack(outputs0, dim=0)
        outputs0 = self.postprocess_masks(
            outputs0,
            original_size=original_size,
        )       
        # tgt_out_maxvalue, tgt_out = torch.max(outputs0, dim=1)
        # for i in range(args.n_class):
        #     tgt_out[(tgt_out_maxvalue < args.threshold) * (tgt_out == i)] = 255
        if self.training:
            tgt_out_maxvalue, _ = torch.max(outputs0, dim=1)
            flattened_maxvalues = tgt_out_maxvalue.view(B, -1)
            sorted_values, _ = torch.sort(flattened_maxvalues, descending=True, dim=1)
            num_elements = flattened_maxvalues.shape[1]
            threshold_idx = int(num_elements * (1-args.threshold))
            thresholds = sorted_values[:, threshold_idx].view(-1, 1, 1)
            tgt_out = torch.where(tgt_out_maxvalue < thresholds, torch.full_like(tgt_out_maxvalue, 255), torch.full_like(tgt_out_maxvalue, 0))

            prompt = promt_generate(tgt_out, is_clu=True)
            # 有提示decoder
            for image_record, curr_embedding in zip(prompt, image_embeddings):
                # Transform input prompts
                if "point_coords" in image_record:
                    point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                    point_coords = self.transform.apply_coords(point_coords, (H, W))
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    points = (coords_torch, labels_torch)
                if "boxes" in image_record:
                    box = image_record["boxes"]
                    box = self.transform.apply_boxes(box, (H, W))
                    box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                    box = box[None, :]
                if "mask_inputs" in image_record:
                    masks = image_record["mask_inputs"]
                    masks = torch.as_tensor(masks, dtype=torch.float, device=batched_input.device)
                    masks = masks[None, :, :, :]
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                low_res_masks, iou_predictions = self.mask_decoder_2(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                outputs.append(low_res_masks)
            outputs = torch.stack(outputs, dim=0)
            outputs = self.postprocess_masks(
                outputs,
                original_size=original_size,
            )
            return outputs0, outputs
        return outputs0
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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


class Sam_muti_styledeco(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder_1: StyleMaskDecoder,
        mask_decoder_2: MaskDecoder1,
        img_size=1024,
    ) -> None:
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder_1 = mask_decoder_1
        self.mask_decoder_2 = mask_decoder_2
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        labels: torch.Tensor,
        multimask_output: bool,
        args: Any      
    ) -> List[Dict[str, torch.Tensor]]:
        B, _, H, W = batched_input.size()
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        
        outputs0 = []
        outputs1 = []
        outputs = []
        # 无提示decoder
        img_pe = self.pe_layer([64, 64]).unsqueeze(0).repeat(B, 1, 1, 1)
        outputs0, outputs1, diffloss = self.mask_decoder_1(image_embeddings=image_embeddings,
                                          image_pe=img_pe, multimask_output=True)
        outputs0 = self.postprocess_masks(
            outputs0,
            original_size=original_size,
        )   
        outputs1 = self.postprocess_masks(
            outputs1,
            original_size=original_size,
        )
        diffloss = diffloss / B

        consistency_loss = self.consis_loss(outputs0, outputs1)
        if self.training:
            tgt_out_maxvalue, _ = torch.max(outputs1, dim=1)
            flattened_maxvalues = tgt_out_maxvalue.view(B, -1)
            sorted_values, _ = torch.sort(flattened_maxvalues, descending=True, dim=1)
            num_elements = flattened_maxvalues.shape[1]
            threshold_idx = int(num_elements * (1-args.threshold))
            thresholds = sorted_values[:, threshold_idx].view(-1, 1, 1)
            tgt_out = torch.where(tgt_out_maxvalue < thresholds, torch.full_like(tgt_out_maxvalue, 255), torch.full_like(tgt_out_maxvalue, 0))
            # True_mask = (outputs1.argmax(1) != labels).unsqueeze(1)
            # masks = torch.cat((True_mask.long().float().detach(),outputs1.detach()),dim=1)

            prompt = promt_generate(tgt_out, is_clu=True)
            # 有提示decoder
            for i, (image_record, curr_embedding) in enumerate(zip(prompt, image_embeddings)):
                # Transform input prompts
                if "point_coords" in image_record:
                    point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                    point_coords = self.transform.apply_coords(point_coords, (H, W))
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    points = (coords_torch, labels_torch)
                if "boxes" in image_record:
                    box = image_record["boxes"]
                    box = self.transform.apply_boxes(box, (H, W))
                    box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                    box = box[None, :]
                # if "mask_inputs" in image_record:
                # mask_inputs = masks[i].unsqueeze(0)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=None,
                )
                
                low_res_masks, iou_predictions = self.mask_decoder_2(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                outputs.append(low_res_masks)
            outputs = torch.stack(outputs, dim=0)
            outputs = self.postprocess_masks(
                outputs,
                original_size=original_size,
            )
            return outputs0, outputs1, outputs, diffloss, consistency_loss
        return outputs0, outputs1
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

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
    def consis_loss(self, aug_prob, im_prob):
        aug_prob = F.softmax(aug_prob, dim=1)
        im_prob = F.softmax(im_prob, dim=1)
        aug_prob = aug_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
        im_prob = im_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
        
        # 添加epsilon防止log(0)
        aug_prob = torch.clamp(aug_prob, 1e-7, 1.0)
        im_prob = torch.clamp(im_prob, 1e-7, 1.0)
        
        # 标准双向KL散度
        consistency_loss = (F.kl_div(aug_prob.log(), im_prob, reduction='batchmean') +
                          F.kl_div(im_prob.log(), aug_prob, reduction='batchmean')) / 2.0
        return consistency_loss
    # def consis_loss(self, aug_prob, im_prob):
    #     aug_prob = F.softmax(aug_prob, dim=1)
    #     im_prob = F.softmax(im_prob, dim=1)
    #     aug_prob = aug_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
    #     im_prob = im_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
    #     p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
    #     consistency_loss = 1* (
    #                         F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
    #                         F.kl_div(p_mixture, im_prob, reduction='batchmean') 
    #                         ) / 2.
    #     return consistency_loss
    