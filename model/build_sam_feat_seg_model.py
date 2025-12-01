import torch

from functools import partial

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN, SegDecoderLinear
from .AutoSamSeg import AutoSamSeg
from .sam_decoder import MaskDecoder
from model.sam import ImageEncoderViT, TwoWayTransformer
from model.sam.tiny_vit_sam import TinyViT


def _build_feat_seg_model(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    sam_seg = SamFeatSeg(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        #seg_decoder=SegDecoderLinear(num_classes=num_classes),
        seg_decoder=SegDecoderCNN(num_classes=num_classes, num_depth=4)
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = []
        for k in state_dict.keys():
            if k in sam_seg.state_dict().keys():
                loaded_keys.append(k)
        sam_seg.load_state_dict(state_dict, strict=False)
        print("loaded keys:", loaded_keys)

    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=14, checkpoint=None):
    return _build_feat_seg_model(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=14, checkpoint=None):
    return _build_feat_seg_model(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=14, checkpoint=None):
    return _build_feat_seg_model(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

def build_sam_vit_t_cnn(num_classes=14, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = AutoSamSeg(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            seg_decoder=SegDecoderCNN(num_classes=num_classes, num_depth=4)
            # pixel_mean=[123.675, 116.28, 103.53],
            # pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    # if checkpoint is not None:
    #     with open(checkpoint, "rb") as f:
    #         state_dict = torch.load(f)
    #     mobile_sam.load_state_dict(state_dict)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = {}
        for k in state_dict.keys():
            if k in mobile_sam.state_dict().keys() and 'iou'not in k and "mask_tokens" not in k:
                loaded_keys[k] = state_dict[k]
        mobile_sam.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())
    return mobile_sam

sam_feat_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
    "vit_t": build_sam_vit_t_cnn,
}

