
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

from .resnet import resnet34, ResNet, BasicBlock
from .unet_2d_blocks import (
    CrossAttnDownBlock2D_local,
    CrossAttnUpBlock2D_local,
    DownBlock2D,
    UNetMidBlock2DCrossAttn_local,
    UpBlock2D,
    get_down_block,
    get_up_block,
)
import copy
import torch
from torch import Tensor
#from maskrcnn_benchmark.layers import ROIAlign
#from .crnn import CRNN_2
import math
from models_mae_text import mae_vit_base_patch16_dec512d8b

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor

class PatchEncoder_v2(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self):
        super().__init__()
        self.down_blocks_patch = nn.ModuleList([])
        block_out_channels = (320, 640, 1280, 1280)
        block_out_channels_condition = (320, 640, 768, 768)
        down_sampling_layers = [0, 1, 2]
        output_channel = block_out_channels[0]
        self.conv_in_patch = nn.Conv2d(3,
                                 block_out_channels[0],
                                 kernel_size=3,
                                 padding=(1, 1))

        #self.conv_in_patch.gradient_checkpointing = False
        self.conv_in_patch_2 = nn.Conv2d(block_out_channels[0] * 2,
                                 block_out_channels[0],
                                 kernel_size=3,
                                 padding=(1, 1))
        #self.conv_in_patch_2.gradient_checkpointing = False
        for i in range(len(block_out_channels_condition)):
            input_channel = output_channel
            output_channel = block_out_channels_condition[i]
            if i in down_sampling_layers:
                is_final_block = False
            else:
                is_final_block = True
            blocks_time_embed_dim = block_out_channels[0] * 4
            down_block_p = get_down_block(
                "DownBlock2D",
                # num_layers=layers_per_block,
                num_layers=3,
                in_channels=input_channel,#,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                cross_attention_dim=1280,
                attn_num_head_channels=8,
                downsample_padding=1,
            )
            self.down_blocks_patch.append(down_block_p)
        self.down_blocks_patch.stop_gradient = False
        #self.ROIAlign = ROIAlign((32, 128), 1, -1)
        #self.crnn = CRNN_2(32, 768, 37, 256)
        '''self.attnpool_1 = AttentionPool2d(28, 768)
        std = self.attnpool_1.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool_1.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.c_proj.weight, std=std)
        self.linear_1 = nn.Linear(768, 1280)
        self.linear_2 = nn.Linear(768, 640)
        self.linear_3 = nn.Linear(768, 320)'''




    def pad(self, patches):
        N, C, H, W = patches.shape
        min_h = int(math.ceil(H / 4) * 4)
        min_w = int(math.ceil(W / 4) * 4)
        padded = torch.zeros((N, C, min_h, min_w))
        padded[:, :, :H, :W] = patches
        return padded

    def forward(self, patches, pos=None, boxes=None):
        N, C, H, W = patches.shape
        #padded = self.pad(patches)

        with_pos = pos is not None#False
       
        if with_pos:
            #print(self.conv_in_patch(patches).shape, pos.shape)
            patches = torch.cat([self.conv_in_patch(patches), pos.to(patches.dtype)], 1)
            patches = self.conv_in_patch_2(patches)
        else:
            patches = self.conv_in_patch(patches)

        for block in self.down_blocks_patch:
            patches,_ = block(patches)
        if boxes is not None:
            rois = self.ROIAlign(patches, boxes)
            preds = self.crnn(rois)
            patches = patches.flatten(2).permute(0,2,1)
            return patches, preds

        #spatial_patch_embeddings_3 = self.attnpool_1(patches)
        #f1 = self.linear_1(spatial_patch_embeddings_3)
        #f2 = self.linear_2(spatial_patch_embeddings_3)
        #f3 = self.linear_3(spatial_patch_embeddings_3)
    


        patches = patches.flatten(2).permute(0,2,1)
        return patches#, [f1, f2, f3]#spatial_patch_embeddings_3


class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        self.backbone = ResNet(BasicBlock, [3, 4, 6, 3])#eval("resnet34")(pretrained=True, return_list=True)
      
        self.conv_in = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_out_2 = nn.Conv2d(64, 768, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_out_3 = nn.Conv2d(128, 768, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_out_4 = nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_out_5 = nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.attnpool_1 = AttentionPool2d(7, 128)
        #self.attnpool_2 = AttentionPool2d(2, 512)
        std = self.attnpool_1.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool_1.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool_1.c_proj.weight, std=std)

        '''std = self.attnpool_2.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool_2.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool_2.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool_2.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool_2.c_proj.weight, std=std)'''

    def forward(self,imgs):
        imgs = self.conv_in(imgs)
        feat_2, feat_3, feat_4, feat_5 = self.backbone(imgs)
        spatial_patch_embeddings_3 = self.attnpool_1(feat_3)
        feat_2 = self.conv_out_2(feat_2).flatten(2).permute(0,2,1)
        feat_3 = self.conv_out_3(feat_3).flatten(2).permute(0,2,1)
        feat_4 = self.conv_out_4(feat_4).flatten(2).permute(0,2,1)
        feat_5 = self.conv_out_5(feat_5).flatten(2).permute(0,2,1)
        fg_feature = torch.cat([feat_2, feat_3, feat_4, feat_5], 1)
        
        return spatial_patch_embeddings_3, fg_feature, feat_5

class PositionalEncoding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=160, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def forward(self, tensors, mask=None):
        x = tensors
        if mask is None:
            n, c, h, w = tensors.shape
            mask = torch.zeros((n,h,w)).bool().cuda()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
    
       
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)


        self.activation = _get_activation_fn(activation)
        
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        q = k = self.with_pos_embed(src, pos)
        src_attn = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src_attn)
        src = self.norm1(src)

        src1 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src1 = src + self.dropout2(src1)
        src1 = self.norm2(src1)

        return src1

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)




class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim=7, embed_dim=768, num_heads=8, output_dim=768):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x_ = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x_[:1], key=x_, value=x_,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )  
        return x.squeeze(0)

class UNet2DConditionModel_with_patch_local_inpaint(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 9+768,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D_local",
            "CrossAttnDownBlock2D_local",
            "CrossAttnDownBlock2D_local",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn_local",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D_local", "CrossAttnUpBlock2D_local", "CrossAttnUpBlock2D_local"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        time_embedding_type: str = "positional",
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2

        # 
        '''self.conv_in_1 = nn.Conv2d(
            7, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )'''
        self.conv_in = nn.Conv2d(
            10, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        
        # time
        if time_embedding_type == "fourier":
            time_embed_dim = block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

       

        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn_local" or mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn_local(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )

        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        out_channels = 3
        #self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)
        self.conv_out_1 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)
        #self.referenceEncoder = PatchEncoder()
        #self.mlp_1 = MLP(768, 768, 768, 5)



        #self.pos = PositionalEncoding2D(768//2)
        #self.pos_2 = PositionalEncoding2D(320//2)

        self.referenceEncoder = mae_vit_base_patch16_dec512d8b() #PatchEncoder_v2()
        #self.linear_style = MLP(4, 768, 768, 3)
        dlatent_size = 768
        channels = 768
        use_wscale = True
        #self.style_mod = nn.ModuleList([])
        #self.style_mod.append(ApplyStyle(1280, 1280, use_wscale=use_wscale)),
        #self.style_mod.append(ApplyStyle(640, 640, use_wscale=use_wscale)),
        #self.style_mod.append(ApplyStyle(320, 320, use_wscale=use_wscale))

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:

        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):

        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):

        self.set_attn_processor(AttnProcessor())

    def set_attention_slice(self, slice_size):
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D_local, DownBlock2D, CrossAttnUpBlock2D_local, UpBlock2D)):
            module.gradient_checkpointing = value
    def get_input_v2(self, latent_bg_for_layout, latent_bg_for_inpaint, mask, mask_2, stroke_seg_1, stroke_seg_1_no):
        latent_bg_for_layout = torch.zeros_like(latent_bg_for_layout)
        pos_embed = self.pos(latent_bg_for_layout)
        pos_embed = pos_embed.to(latent_bg_for_layout.dtype)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        layoutnet_input = mask #torch.cat([latent_bg_for_layout, mask], dim=1)
        
        layoutnet_input = self.conv_in_layout(layoutnet_input)
        bs, c, h, w = layoutnet_input.shape
        src = layoutnet_input.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed) #[hw, b, c]
        memory = memory.permute(1, 2, 0).view(bs, c, h, w)
       
        #reference_1, fg_reference, feat_5 = self.referenceEncoder(latent_stroke_seg_1)
        #_, fg_reference_no_style, _ = self.referenceEncoder(latent_stroke_seg_1_no_style)
        # patches: [b, c, HW]
        fg_reference_v2 = self.referenceEncoder_v2_(stroke_seg_1)
        reference_1 = torch.mean(fg_reference_v2, 1)
        fg_reference_v2_no = self.referenceEncoder_v2_(stroke_seg_1_no)
        #print(reference_1.shape)
        #print(fg_reference_v2.shape, fg_reference.shape)
        

        layout_reference = self.mlp_1(reference_1)
        #fg_reference = self.mlp_2(reference_2)
        
        layout_reference = layout_reference.unsqueeze(2).unsqueeze(3).expand_as(memory) 
        memory_1 =  self.fusion_layout(torch.cat([memory, layout_reference], 1))
    
        # 4+1+768
        memory_2 = torch.cat([latent_bg_for_inpaint, mask_2], 1)
        return memory_1, pos_embed, memory_2, fg_reference_v2, fg_reference_v2_no#, fg_reference_no_style, feat_5

    def get_input_v3(self, latent_bg_for_inpaint, mask, stroke_seg_1, stroke_seg_1_no, con_flag=True, ratio=0):

        fg_reference_v2 = None
        fg_reference_v2_no = None

        
        if stroke_seg_1 is not None:
            fg_reference_v2, mask_loss, _ = self.referenceEncoder.forward_encoder(stroke_seg_1, ratio)
        if con_flag == False:
            fg_reference_v2_no, mask_loss, _ = self.referenceEncoder.forward_encoder(stroke_seg_1_no, ratio)

        #memory_2 = mask
        memory_2 = torch.cat([latent_bg_for_inpaint, mask], 1)

        return memory_2, fg_reference_v2, fg_reference_v2_no, mask_loss

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        memory: torch.Tensor,
        layout_embeddings: torch.Tensor,
        char_glyph: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        random_char_index: Optional[int] = None,
        char_masks: Optional[str] = None,
        pos: Optional[torch.Tensor] = None,
        style_adain: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
        """
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        # 2. pre-process
        # 4 + 5 + 768
        #print(sample.shape, memory.shape, layout_embeddings.shape, char_glyph.shape, "!!!!")

        #sample = torch.cat([sample, memory, layout_embeddings, char_glyph], 1)
        sample = torch.cat([sample, memory, char_glyph], 1)
        # 3 + 1 + 3 + 3
        # sample = torch.cat([sample, memory], 1)
        #print(sample.shape)
        
        sample = self.conv_in(sample)#.detach()
        
        
        
        #2.5 img_condition generation
        #sample_text, sample_patch = encoder_hidden_states[0], encoder_hidden_states[1]
      

        #sample_condition = self.conv_in_c(sample_condition)
        #sample_patch = self.conv_in_patch(sample_patch)

        #for downsample_block in self.down_blocks_patch:

        #    sample_patch,_ = downsample_block(sample_patch)
      
        #for downsample_block in self.down_blocks_condition:
        #    sample_condition, _ = downsample_block(hidden_states=sample_condition)
      
        # 3. down
        #print(sample_text.shape, sample_patch.shape)
        #print(encoder_hidden_states[0].shape, "!!!")
        b, _len, dim = encoder_hidden_states[0].shape

        encoder_hidden_states_padded = [torch.zeros((b, 1, dim)).to(encoder_hidden_states[0].device).to(encoder_hidden_states[0].dtype)]

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_padded, #encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
        
        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states_padded, #encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        #####
        #sample = self.style_mod(sample, style_adain)
        #####
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            
            #print(i, sample.shape, is_final_block, hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention)
     
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,##=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
                #print(i, sample.shape)
                #sample = self.style_mod[i-1](sample, style_adain[i-1])

            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            #print(i, sample.shape)#, res_samples.shape)
        
        #####
        #sample = self.style_mod(sample, style_adain)
        #####
        
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out_1(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
        
    def sum_attn_maps(self, down_attn_maps, mid_attn_maps, up_attn_maps, timesteps, random_char_index=0):
        #if random_char_index is None:
        #    return None

        import math
        
        res_attn_maps = []
        for ms in up_attn_maps: # un_sampling_blocks有三个
            temp = []
            for i in range(len(ms)): #每个up_sampling block有三个layers
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b, 8, h*w, 25] 推理时，每个bs的第一个是uncon，第二个才是con
                temp.append(torch.mean(ms_,1).unsqueeze(1)) #torch.Size([b,1, h*w, 25]) 对每个head做一个平均
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1) # [b, layers, h*w, 25] ==> [b, 1,h*w, 25] ==> [b, 25, h*w]
            
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw))) #[bs, h, w]
            
            
            temp = torch.nn.functional.interpolate(
                    temp, size=(56, 56)
                ).to(memory_format=torch.contiguous_format) #[bs, 1, h, w]
            
            res_attn_maps.append(temp)
        
        for ms in mid_attn_maps: # mid_sampling_blocks有一个
            temp = []
            for i in range(len(ms)): #每个up_sampling block有三个layers
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 因为推理时，每个bs的第一个是uncon，第二个才是con
                temp.append(torch.mean(ms_, 1).unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, 25] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw)))
            
            temp = torch.nn.functional.interpolate(
                    temp, size=(56, 56)
                ).to(memory_format=torch.contiguous_format)
            res_attn_maps.append(temp)
        
        for ms in down_attn_maps: # un_sampling_blocks有三个
            temp = []
            for i in range(len(ms)): #每个up_sampling block有三个layers
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 
                temp.append(torch.mean(ms_, 1).unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, max_len] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw)))
            
            temp = torch.nn.functional.interpolate(
                    temp, size=(56, 56)
                ).to(memory_format=torch.contiguous_format)
            res_attn_maps.append(temp)        
        # torch.stack(res_attn_maps, 1) : [bs, 3+2+3, 25, h, w]
        res_attn_maps_mean = torch.mean(torch.stack(res_attn_maps, 1), 1)[:, :100] #[bs, 25, h, w]
        #return res_attn_maps
        #print(torch.max(res_attn_maps), torch.min(res_attn_maps))
        #print(res_attn_maps.shape) #[b, 25, 56, 56]
        vis = False
        if vis and timesteps[-1]==21:
            for i in range(40):
                attn_weight_map = res_attn_maps_mean[1][i].cpu().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0719/vis/vis_attn/{}_{}.png".format(timesteps[-1], i))


        
        return res_attn_maps_mean, torch.stack(res_attn_maps, 1)

    def sum_attn_maps_v2(self, down_attn_maps, mid_attn_maps, up_attn_maps, timesteps, random_char_index=0):

        import math
        res_attn_maps = []
        up_4_map = None
        up_8_map = None
        count = 0 
        for ms in up_attn_maps: # un_sampling_blocks有三个，每个block内的三个layers的尺寸相同
            temp = []
            #print(len(ms), "up")
            for i in range(len(ms)): #每个up_sampling block有三个layers，这三个layers的map.shape相同
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 
                ms_mean = torch.mean(ms_, 1) #[b, h*w, 25]
           
                res_attn_maps.append(ms_mean.permute(0,2,1).contiguous().view(int(b_8//8), _c, int(math.sqrt(hw)), int(math.sqrt(hw))))
                
                temp.append(ms_mean.unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
                
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, max_len] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw))) #[2, 100, 28, 28]]
            
            if temp.shape[-1] == 56:
                up_4_map = temp
            if temp.shape[-1] == 28:
                up_8_map = temp
        
        for ms in mid_attn_maps: 
            temp = []
            #print(len(ms),"mid")
            for i in range(len(ms)): 
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 
                ms_mean = torch.mean(ms_, 1)
                res_attn_maps.append(ms_mean.permute(0,2,1).contiguous().view(int(b_8//8) ,_c, int(math.sqrt(hw)), int(math.sqrt(hw))))
                temp.append(ms_mean.unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
                
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, max_len] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw))) #[2, 100, 28, 28]]
            #res_attn_maps.append(temp)
        
        for ms in down_attn_maps: 
            temp = []
            #print(len(ms), "down")
            for i in range(len(ms)): 
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 
                ms_mean = torch.mean(ms_, 1)
                res_attn_maps.append(ms_mean.permute(0,2,1).contiguous().view(int(b_8//8),_c, int(math.sqrt(hw)), int(math.sqrt(hw))))
                
                temp.append(ms_mean.unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
                
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, max_len] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw))) #[2, 100, 28, 28]]
            #res_attn_maps.append(temp)
        #res_attn_maps_mean_down = torch.mean(torch.stack(res_attn_maps, 1), 1)[:, :100] #[bs, 25, h, w]
        #res_attn_maps_mean = torch.mean(torch.stack(res_attn_maps, 1), 1) #[bs, 25, h, w]
        
        pred_char_mask_4 = [] 
        pred_char_mask_8 = []
        pred_char_mask_16 = []
        pred_char_mask_32 = []
        for attn in res_attn_maps:
            #print(attn.shape)
            # attn.shape [bs, 100, h, w]
            if attn.shape[-1] == 56:
                pred_char_mask_4.append(attn)
            if attn.shape[-1] == 28:
                pred_char_mask_8.append(attn)
            if attn.shape[-1] == 14:
                pred_char_mask_16.append(attn)
            if attn.shape[-1] == 7:
                pred_char_mask_32.append(attn)
        
        pred_char_mask_4 = torch.mean(torch.stack(pred_char_mask_4),dim=0) #[bs, 100, h, w]
        pred_char_mask_8 = torch.mean(torch.stack(pred_char_mask_8),dim=0) #[bs, 100, h, w]
        pred_char_mask_16 = torch.mean(torch.stack(pred_char_mask_16),dim=0) #[bs, 100, h, w]
        pred_char_mask_32 = torch.mean(torch.stack(pred_char_mask_32),dim=0) #[bs, 100, h, w]
        
        vis = False
        #if vis and timesteps[-1]==21:
        if vis and timesteps[-1] == 21:
            for i in range(20):
                attn_weight_map = up_4_map[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0808/vis/vis_attn/{}_{}.png".format("up", i))


                attn_weight_map = pred_char_mask_4[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0808/vis/vis_attn/{}_{}.png".format(4, i))
                attn_weight_map = pred_char_mask_8[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0808/vis/vis_attn/{}_{}.png".format(8, i))
                attn_weight_map = pred_char_mask_16[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0808/vis/vis_attn/{}_{}.png".format(16, i))
                attn_weight_map = pred_char_mask_32[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0808/vis/vis_attn/{}_{}.png".format(32, i))


                
            '''for i in range(12):
                for j in range(len(res_attn_maps)):
                    attn_weight_map = res_attn_maps[j][1][i].cpu().numpy()
                    plt.imshow(attn_weight_map)
                    plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0719/vis/vis_up_attn/{}_{}_{}.png".format(timesteps[-1], j, i))'''

        
        return up_8_map, res_attn_maps #torch.stack(res_attn_maps, 1)

    def sum_attn_maps_v3(self, down_attn_maps, mid_attn_maps, up_attn_maps, timesteps, random_char_index=0):
        #if random_char_index is None:
        #    return None

        import math
        res_attn_maps = []
        for ms in up_attn_maps: # un_sampling_blocks有三个
            temp = []
            for i in range(len(ms)): #每个up_sampling block有三个layers
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(int(b_8//8),8,hw,_c) #[b,8, h*w, 25] 
                temp.append(torch.mean(ms_, 1).unsqueeze(1)) #torch.Size([3136, 25]) 对每个head做一个平均
            temp = torch.mean(torch.stack(temp, 1), 1).squeeze(1).permute(0,2,1)  # [layers, h*w, max_len] ==> [h*w, 25]
            temp = temp.view(int(b_8//8), -1, int(math.sqrt(hw)), int(math.sqrt(hw))) #[2, 100, 28, 28]]
            
            if temp.shape[-1] == 56:
                res_attn_maps.append(temp)
        
        #res_attn_maps_mean_down = torch.mean(torch.stack(res_attn_maps, 1), 1)[:, :100] #[bs, 25, h, w]
        #res_attn_maps_mean = torch.mean(torch.stack(res_attn_maps, 1), 1) #[bs, 25, h, w]
        res_attn_maps_mean = res_attn_maps[0]

        vis = False
        #if vis and timesteps[-1]==21:
        if vis and timesteps[-1] == 21:
            for i in range(30):
                attn_weight_map = res_attn_maps_mean[1][i].cpu().detach().numpy()
                plt.imshow(attn_weight_map)
                plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0728/vis/vis_attn/{}_{}.png".format(timesteps[-1], i))
            '''for i in range(12):
                for j in range(len(res_attn_maps)):
                    attn_weight_map = res_attn_maps[j][1][i].cpu().numpy()
                    plt.imshow(attn_weight_map)
                    plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0719/vis/vis_up_attn/{}_{}_{}.png".format(timesteps[-1], j, i))'''
                #attn_weight_map = res_attn_maps_mean_down[1][i].cpu().numpy()
                #plt.imshow(attn_weight_map)
                #plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/0719/vis/vis_down_attn/{}_{}.png".format(timesteps[-1], i))
                
        #res = []
        #for c in range(len(random_char_index)):
        #    res.append(res_attn_maps[c][random_char_index[c]]) #[h,w]'''
        #return torch.stack(res,0)
        
        return res_attn_maps_mean, torch.stack(res_attn_maps, 1)

        
        
        
    def vis_attn_maps(self, down_attn_maps, mid_attn_maps, up_attn_maps, timesteps):
        # print(len(down_attn_maps), len(mid_attn_maps), len(up_attn_maps)) 4 1 3
        attn_maps_per = []
        import math
        count = 0
        #print(len(up_attn_maps))
        for ms in mid_attn_maps: # un_sampling_blocks有三个
            temp = []
            for i in range(len(ms)): #每个up_sampling block有三个layers
                b_8, hw, _c = ms[i].shape
                ms_ = ms[i].view(2,8,hw,_c)[1] #[b,8, h*w, 25] 因为推理时，每个bs的第一个是uncon，第二个才是con
                temp.append(torch.mean(ms_,0)) #torch.Size([3136, 25]) 对每个head做一个平均
            
            temp = torch.mean(torch.stack(temp, 0),0) # [layers, h*w, 25] ==> [h*w, 25]
            hw = temp.shape[0]
            vis_temp = temp[:,1].view(int(math.sqrt(hw)), int(math.sqrt(hw))) + temp[:,0].view(int(math.sqrt(hw)), int(math.sqrt(hw)))
            #print(temp.shape, len(timesteps), torch.max(temp), torch.min(temp))
            #print(temp.shape)
            attn_weight_map = vis_temp.cpu().numpy()
            #plt.imshow(attn_weight_map)
            #plt.savefig("/root/paddlejob/workspace/env_run/wujingjing/diffusers_pytorch/vis/vis_attn/{}_{}.png".format(timesteps[-1], count))
            vis_temp = Image.fromarray((temp * 255).cpu().detach().numpy().astype("uint8"))
            
            #vis_temp.save("/root/paddlejob/workspace/env_run/wujingjing/diffusers_pytorch/vis/vis_attn/{}_{}.png".format(timesteps[-1], count))
            count += 1
            attn_maps_per.append(temp)
        
        
            
            


