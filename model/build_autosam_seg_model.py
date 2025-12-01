import torch

from functools import partial

from model.promptsam import PromptSAM
from model.sam.sam import mobilesam

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN
from .AutoSamSeg import AutoSamSeg
from .sam_decoder import MaskDecoder, MaskDecoder1, SegmentAnythingDecoder, StyleMaskDecoder
from model.sam import ImageEncoderViT, TwoWayTransformer, Sam, PromptEncoder, Sam_test, Sam_muti_deco, Samvq_muti_deco, Sam_muti_newdeco, Sam_muti_styledeco, samadaptor
from model.sam.tiny_vit_sam import TinyViT


def _build_sam_seg_model(
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
    sam_seg = AutoSamSeg(
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
        seg_decoder=MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        ),
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = {}
        for k in state_dict.keys():
            if k in sam_seg.state_dict().keys() and 'iou'not in k and "mask_tokens" not in k:
                loaded_keys[k] = state_dict[k]
        sam_seg.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())

    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

def build_sam_vit_t(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
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

def build_sam_vit_test(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam_test(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder1(
                    num_multimask_outputs=5,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
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

def build_sam_vit_test_decoder(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam_muti_deco(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder_1=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
            mask_decoder_2=MaskDecoder1(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
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

def build_vqsam_vit_test(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Samvq_muti_deco(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder_1=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
            mask_decoder_2=MaskDecoder1(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
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

def build_sam_vit_test_newdecoder(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam_muti_newdeco(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder_1=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
            mask_decoder_2=MaskDecoder1(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
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

def build_sam_vit_test_styledecoder(in_chans, num_classes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam_muti_styledeco(
            in_ch=in_chans,
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
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder_1=StyleMaskDecoder(
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                num_classes=num_classes
            ),
            mask_decoder_2=MaskDecoder1(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = {}
        for k in state_dict.keys():
            if k in mobile_sam.state_dict().keys() and 'iou'not in k and "mask_tokens" not in k and "mask_downscaling" not in k:
                loaded_keys[k] = state_dict[k]
        mobile_sam.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())
    return mobile_sam

def build_promptsam_vit_t(in_chans, num_classes, checkpoint=None):
    mobile_sam = PromptSAM(
            in_ch=in_chans,
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
            num_classes=num_classes,
            upsample_times = 3
        )

    mobile_sam.eval()
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

def build_mobliesam_vit_t(in_chans, num_classes, checkpoint=None):
    mobile_sam = mobilesam(
            in_ch=in_chans,
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
            mask_decoder=SegmentAnythingDecoder(
                in_channels=256, 
                nums=num_classes
            ),
        )

    mobile_sam.eval()
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


def build_samadaptor_vit_s(in_chans, num_classes, checkpoint=None):
    sam_adaptor = samadaptor(
            in_ch=in_chans,
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
            mask_decoder=MaskDecoder1(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes
            ),
        )

    sam_adaptor.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)['model']

        loaded_keys = {}
        for k in state_dict.keys():
            if k in sam_adaptor.state_dict().keys() and 'iou'not in k and "mask_tokens" not in k:
                loaded_keys[k] = state_dict[k]
        sam_adaptor.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())
    return sam_adaptor

# def build_sam_vit_t(num_classes, checkpoint=None):
#     prompt_embed_dim = 256
#     image_size = 1024
#     vit_patch_size = 16
#     mobile_sam = AutoSamSeg(
#             image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
#                 embed_dims=[64, 128, 160, 320],
#                 depths=[2, 2, 6, 2],
#                 num_heads=[2, 4, 5, 10],
#                 window_sizes=[7, 7, 14, 7],
#                 mlp_ratio=4.,
#                 drop_rate=0.,
#                 drop_path_rate=0.0,
#                 use_checkpoint=False,
#                 mbconv_expand_ratio=4.0,
#                 local_conv_size=3,
#                 layer_lr_decay=0.8
#             ),
#             seg_decoder=MaskDecoder(
#                     num_multimask_outputs=3,
#                     transformer=TwoWayTransformer(
#                     depth=2,
#                     embedding_dim=prompt_embed_dim,
#                     mlp_dim=2048,
#                     num_heads=8,
#                 ),
#                 transformer_dim=prompt_embed_dim,
#                 iou_head_depth=3,
#                 iou_head_hidden_dim=256,
#                 num_classes=num_classes
#             ),
#         )

#     mobile_sam.eval()
#     if checkpoint is not None:
#         with open(checkpoint, "rb") as f:
#             state_dict = torch.load(f)

#         loaded_keys = {}
#         for k in state_dict.keys():
#             if k in mobile_sam.state_dict().keys() and 'iou'not in k and "mask_tokens" not in k:
#                 loaded_keys[k] = state_dict[k]
#         mobile_sam.load_state_dict(loaded_keys, strict=False)
#         print("loaded keys:", loaded_keys.keys())
#     return mobile_sam

sam_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
    "vit_t": build_sam_vit_t,
    "vit_promptsam": build_promptsam_vit_t,
    "vit_mobliesam": build_mobliesam_vit_t,
    "vit_samadaptor": build_samadaptor_vit_s,
    "vit_test": build_sam_vit_test,
    "vit_muti_decoder": build_sam_vit_test_decoder,
    "vit_vq_muti_decoder": build_vqsam_vit_test,
    "vit_new_muti_decoder": build_sam_vit_test_newdecoder,
    "vit_style_muti_decoder": build_sam_vit_test_styledecoder,
}

