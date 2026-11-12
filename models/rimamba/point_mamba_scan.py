from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ulip.pointbert import misc
from utils.logger import *
from utils.pose_utils import compute_pca, content_orientation_disentanglement, transform_to_pca_frame
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba
from .block_scan import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .serialization import Point


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, in_dim=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups, color_groups=None):
        '''
            point_groups : B G N 3 (xyz)
            color_groups: B G N 3 (rgb if provided)
            -----------------
            feature_global : B G C
        '''
        if color_groups is not None:
            input_groups = torch.cat([point_groups, color_groups], dim=-1)
        else:
            input_groups = point_groups

        bs, g, n, _ = input_groups.shape
        input_groups = input_groups.reshape(bs * g, n, -1)
        # encoder
        feature = self.first_conv(input_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, rgb=None):
        '''
            input: B N 3
            rgb: B N 3 (if provided)
            ---------------------------
            neighborhood: B G K 3
            center : B G 3
            color_groups: B G K 3 (if rgb is not None)
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if rgb is not None:
            color_groups = rgb.view(batch_size * num_points, -1)[idx, :]
            color_groups = color_groups.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            return neighborhood, center, color_groups
        
        return neighborhood, center


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def serialization(pos, feat=None, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))

        return hidden_states

class FeatureModulation(nn.Module):
    def __init__(self, embed_dim):
        super(FeatureModulation, self).__init__()
        
        # Single MLP for both γ and β
        self.modulation_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Project concatenated (pos, ori) to embed_dim
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2)  # Output both γ and β
        )

    def forward(self, pos_embedding, ori_embedding):
        """
        pos_embedding: (B, N, embed_dim)
        ori_embedding: (B, N, embed_dim)
        """
        modulation_input = torch.cat([pos_embedding, ori_embedding], dim=-1)  # (B, N, 2 * embed_dim)
        modulation_output = self.modulation_mlp(modulation_input)  # (B, N, 2 * embed_dim)
        
        gamma, beta = modulation_output.chunk(2, dim=-1)  # Split into two (B, N, embed_dim)

        gamma = 1 + 0.5 * torch.sigmoid(gamma)  # Scale in [1, 1.5]
        
        return gamma, beta

class FeatureModulationWithBottleneck(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim=128):
        super(FeatureModulationWithBottleneck, self).__init__()

        self.bottleneck = nn.Linear(embed_dim, bottleneck_dim)

        # Single MLP for both γ and β
        self.modulation_mlp = nn.Sequential(
            nn.Linear(bottleneck_dim * 3, bottleneck_dim),  # Project concatenated (pos, ori) to embed_dim
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim * 2)  # Output both γ and β
        )

    def forward(self, pos_embedding, ori_embedding):
        """
        pos_embedding: (B, N, embed_dim)
        ori_embedding: (B, N, embed_dim)
        """

        modulation_input = torch.cat([
            self.bottleneck(pos_embedding),
            self.bottleneck(ori_embedding),
            self.bottleneck(pos_embedding * ori_embedding),
        ], dim=-1)  # (B, N, 3 * bottleneck_dim)

        modulation_output = self.modulation_mlp(modulation_input)  # (B, N, 2 * embed_dim)
        
        gamma, beta = modulation_output.chunk(2, dim=-1)  # Split into two (B, N, embed_dim)

        gamma = 0.5 + torch.sigmoid(gamma)  # Scale in [1, 1.5]
        
        return gamma, beta
    
class RIMixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
            bottleneck_film=False,
            reverse=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Feature Modulation (gamma and beta)
        # self.feature_modulations = nn.ModuleList([
        #     FeatureModulation(d_model) for _ in range(n_layer)
        # ]) if not bottleneck_film else nn.ModuleList([
        #     FeatureModulationWithBottleneck(d_model) for _ in range(n_layer)
        # ])
        self.feature_modulations = nn.ModuleList([
            FeatureModulationWithBottleneck(d_model) for _ in range(n_layer)
        ]) if bottleneck_film else None 
        
        self.reverse=reverse
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, ori, inference_params=None):
        hidden_states = input_ids # + pos + ori # use all three 
        if pos is not None: hidden_states = hidden_states + pos 
        if ori is not None: hidden_states = hidden_states + ori 
        
        for i, layer in enumerate(self.layers):
            if self.feature_modulations is not None and pos is not None and ori is not None:
                # whether to use FiLM or not 
                # Compute gamma and beta for feature modulation
                gamma, beta = self.feature_modulations[i](pos, ori)
                
                # Apply modulation: FiLM-style scaling and shifting
                hidden_states = gamma * hidden_states + beta

            # Reverse hidden states on even layers if self.reverse is True
            if self.reverse and i % 2 == 0:
                hidden_states = hidden_states.flip(dims=[1])

            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            # Reverse back after processing
            if self.reverse and i % 2 == 0:
                hidden_states = hidden_states.flip(dims=[1])

            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))

        return hidden_states

class PointMambaScan(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMambaScan, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.use_color = False if not hasattr(self.config, "use_color") else self.config.use_color
        
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims, in_dim=(6 if self.use_color else 3))

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out = 0. if not hasattr(self.config, "drop_out") else self.config.drop_out
        self.max_head = False if not hasattr(self.config, "max_head") else self.config.max_head
        self.avg_head = False if not hasattr(self.config, "avg_head") else self.config.avg_head
        self.pca_norm = False if not hasattr(self.config, "pca_norm") else self.config.pca_norm

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out=self.drop_out,
                                 drop_path=dpr)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        if self.max_head and self.avg_head:
            self.HEAD_CHANEL += 1

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        self.build_loss_func()

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, features=None):
        if self.pca_norm:
            mean, grf = compute_pca(pts, disambiguate=True)
            pts, _ = transform_to_pca_frame(pts, None, mean, grf)
            pts = pts.contiguous()
        
        if self.use_color:
            rgb = features[:, :, 3:] # input features is concat of xyz and rgb 
            neighborhood, center, color = self.group_divider(pts, rgb)
        else:
            neighborhood, center = self.group_divider(pts)
            color = None 
            
        group_input_tokens = self.encoder(neighborhood, color_groups=color)  # B G N
        pos = self.pos_embed(center)  # B G C

        # # reordering strategy
        _, _, _, group_input_tokens_forward, pos_forward = serialization_func(center, group_input_tokens, pos,
                                                                              'hilbert')
        _, _, _, group_input_tokens_backward, pos_backward = serialization_func(center, group_input_tokens, pos,
                                                                                'hilbert-trans')
        group_input_tokens_forward = apply_OrderScale(group_input_tokens_forward,
                                                      self.OrderScale_gamma_1, self.OrderScale_beta_1)
        group_input_tokens_backward = apply_OrderScale(group_input_tokens_backward,
                                                       self.OrderScale_gamma_2, self.OrderScale_beta_2)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(group_input_tokens_forward.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens_forward.size(0), -1, -1)
            pos = torch.cat([pos_forward, pos_backward, cls_token], dim=1)
            group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward, cls_pos], dim=1)
        else:
            pos = torch.cat([pos_forward, pos_backward], dim=1)
            group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward], dim=1)


        x = group_input_tokens
        x = self.blocks(x, pos)
        # x = self.norm(x)
        if self.use_cls_token:
            cls_token = x[:, -1, :]
            concat_f = torch.cat([cls_token, x.mean(1)], dim=1)
        else:
            concat_f = None
            if self.avg_head:
                if concat_f is None:
                    concat_f = x[:, :].mean(1)
                else:
                    concat_f = torch.cat([concat_f, x[:, :].mean(1)], dim=1)
            if self.max_head:
                if concat_f is None:
                    concat_f = x[:, :].max(1)[0]
                    # concat_f = x[:, -1]
                else:
                    concat_f = torch.cat([concat_f, x[:, :].max(1)[0]], dim=1)
                    # concat_f = torch.cat([concat_f, x[:, -1]], dim=1)
            if concat_f is None:
                assert False
        
        return concat_f

class RIMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(RIMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.use_color = False if not hasattr(self.config, "use_color") else self.config.use_color

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims, in_dim=(6 if self.use_color else 3))


        drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out = 0. if not hasattr(self.config, "drop_out") else self.config.drop_out
        self.relative = True if not hasattr(self.config, "relative") else self.config.relative # use relative pos and ori 
        self.bottleneck_film = False if not hasattr(self.config, "bottleneck_film") else self.config.bottleneck_film 
        self.reverse = False if not hasattr(self.config, "reverse") else self.config.reverse 
        self.use_pos_emb = True if not hasattr(self.config, "use_pos_emb") else self.config.use_pos_emb 
        self.use_ori_emb = True if not hasattr(self.config, "use_ori_emb") else self.config.use_ori_emb 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim)) if self.use_pos_emb else None 
        self.cls_ori = nn.Parameter(torch.randn(1, 1, self.trans_dim)) if self.use_ori_emb else None 
        
        trunc_normal_(self.cls_token, std=.02)
        if self.use_pos_emb: trunc_normal_(self.cls_pos, std=.02)
        if self.use_ori_emb: trunc_normal_(self.cls_ori, std=.02)
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        ) if self.use_pos_emb else None 
        
        self.ori_embed = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        ) if self.use_ori_emb else None 
        
        self.blocks = RIMixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out=self.drop_out,
                                 drop_path=dpr,
                                 bottleneck_film=self.bottleneck_film,
                                 reverse=self.reverse)

        # self.HEAD_CHANEL = 2

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def orig_forward(self, pts, features=None):
        # GLOBAL PCA on all pts AND center the pts
        mean, grf = compute_pca(pts, disambiguate=True)
        # pts = pts - mean 
        
        if self.use_color:
            rgb = features[:, :, 3:] # input features is concat of xyz and rgb 
            neighborhood, center, color = self.group_divider(pts, rgb)
        else:
            neighborhood, center = self.group_divider(pts)
            color = None 
        
        r_center = torch.einsum('bij,bnj->bni', grf.transpose(1, 2), center)
        # TODO r_center = torch.matmul(center, grf) - should be the same as above 
        _, order, _, _, _ = serialization(r_center, order='hilbert')
                        
        # disambiguate the lrf 
        neighborhood, ori_mat = content_orientation_disentanglement(neighborhood)  # (B, G, S, 3), (B, G, 3, 3)
        # for a given matrix ori_mat[b, g]: eigen are COL vectors
        
        # miniPointNet to compute patch feat from local patches
        group_input_tokens = self.encoder(neighborhood, color_groups=color)  # B G N
                
        B, G, _ = center.shape 
        center = center.flatten(0, 1)[order].reshape(B, G, -1).contiguous() 
        ori_mat = ori_mat.flatten(0, 1)[order].reshape(B, G, 3, 3).contiguous() 
        group_input_tokens = group_input_tokens.flatten(0, 1)[order].reshape(B, G, -1).contiguous() 
        
        
        if self.relative:
            rel_pos = torch.zeros_like(center)  # Shape: (b, g, 3)
            rel_pos[:, 0] = center[:, 0]
            rel_pos[:, 1:] = center[:, 1:] - center[:, :-1]
        else:
            rel_pos = center 

        rel_pos = torch.matmul(rel_pos.reshape(B*G, 1, 3), ori_mat.reshape(B*G, 3, 3))  
        rel_pos = rel_pos.reshape(B, G, 3)
        rel_ori = get_rel_ori(ori_mat, grf, relative=self.relative) # (B, G, 9)
        
        # compute rel pos and or EMBEDDINGS
        if self.pos_embed is not None:
            pos_emb = self.pos_embed(rel_pos)
            cls_pos = self.cls_pos.expand(pos_emb.size(0), -1, -1) 
            pos_emb = torch.cat([pos_emb, cls_pos], dim=1) 
        else: 
            pos_emb = None 
            
        if self.ori_embed is not None:
            ori_emb = self.ori_embed(rel_ori) if self.ori_embed else None # B, G, C 
            cls_ori = self.cls_ori.expand(ori_emb.size(0), -1, -1)
            ori_emb = torch.cat([ori_emb, cls_ori], dim=1) 
        else:
            ori_emb = None 
            
        # expand cls token
        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat([group_input_tokens, cls_token], dim=1) 

        x = self.blocks(x, pos_emb, ori_emb) # B, G+1, C
        
        features = torch.mean(x, dim=1)
        
        return features

    def forward_with_grf(self, pts, grf, features=None):
        if self.use_color:
            rgb = features[:, :, 3:]
            neighborhood, center, color = self.group_divider(pts, rgb)
        else:
            neighborhood, center = self.group_divider(pts)
            color = None

        r_center = torch.einsum('bij,bnj->bni', grf.transpose(1, 2), center)
        _, order, _, _, _ = serialization(r_center, order='hilbert')

        neighborhood, ori_mat = content_orientation_disentanglement(neighborhood)
        group_input_tokens = self.encoder(neighborhood, color_groups=color)

        B, G, _ = center.shape
        center = center.flatten(0, 1)[order].reshape(B, G, -1).contiguous()
        ori_mat = ori_mat.flatten(0, 1)[order].reshape(B, G, 3, 3).contiguous()
        group_input_tokens = group_input_tokens.flatten(0, 1)[order].reshape(B, G, -1).contiguous()

        if self.relative:
            rel_pos = torch.zeros_like(center)
            rel_pos[:, 0] = center[:, 0]
            rel_pos[:, 1:] = center[:, 1:] - center[:, :-1]
        else:
            rel_pos = center

        rel_pos = torch.matmul(
            rel_pos.reshape(B * G, 1, 3),
            ori_mat.reshape(B * G, 3, 3)
        ).reshape(B, G, 3)

        rel_ori = get_rel_ori(ori_mat, grf, relative=self.relative)

        pos_emb = None
        if self.pos_embed is not None:
            pos_emb = self.pos_embed(rel_pos)
            pos_emb = torch.cat(
                [pos_emb, self.cls_pos.expand(B, -1, -1)], dim=1
            )

        ori_emb = None
        if self.ori_embed is not None:
            ori_emb = self.ori_embed(rel_ori)
            ori_emb = torch.cat(
                [ori_emb, self.cls_ori.expand(B, -1, -1)], dim=1
            )

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([group_input_tokens, cls_token], dim=1)

        x = self.blocks(x, pos_emb, ori_emb)
        return torch.mean(x, dim=1)

    def forward(self, pts, features=None):
        swap = True 
        mean, grf, eigenvals = compute_pca(
            pts, disambiguate=True, return_eigenvals=True
        )
        # --- normal pass ---
        feat_normal = self.forward_with_grf(pts, grf, features)

        if swap:
            # Identify symmetric objects
            eig_ratio = eigenvals[:, 0] / (eigenvals[:, 2] + 1e-8)
            sym_mask = (eig_ratio > 0.95) & (eig_ratio < 1.05)
            
            if sym_mask.any():
                # --- swapped GRF ---
                grf_swapped = grf.clone()
                grf_swapped[sym_mask] = grf_swapped[sym_mask][:, :, [2, 1, 0]]

                feat_swapped = self.forward_with_grf(pts, grf_swapped, features)

                # average only symmetric objects
                feat_normal[sym_mask] = 0.5 * (
                    feat_normal[sym_mask] + feat_swapped[sym_mask]
                )

        return feat_normal        

def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta

def get_rel_ori(ori, grf, relative=True):
    B, G, _, _ = ori.shape
    # Initialize the relative orientations tensor
    rel_ori = torch.zeros(B, G, 9, device=ori.device)  # (B, G, 9)
    
    if relative:
        # Compute the relative orientation for the first patch with respect to grf
        rel_ori[:, 0] = (grf.transpose(-2, -1) @ ori[:, 0]).reshape(B, 9)  # Flatten 3x3 matrix to 9
        
        # For the rest of the patches, compute the relative orientation with respect to the previous patch
        ori_mat_1 = ori[:, :-1, :, :] # Shape (B, G-1, 3, 3)
        ori_mat_2 = ori[:, 1:, :, :]  # Shape (B, G-1, 3, 3)

        # Compute relative orientations for all patches except the first
        rel_ori[:, 1:] = (ori_mat_1.transpose(-2, -1) @ ori_mat_2).reshape(B, G-1, 9)
    else:
        # Expand grf to (B, 1, 3, 3) to match ori's batch size and perform batch-wise multiplication
        rel_ori = (grf.transpose(-2, -1).unsqueeze(1) @ ori).reshape(B, G, 9)
            
    return rel_ori

def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

def serialization_func(p, x, x_res, order, layers_outputs=[]):
    p, order, inverse_order, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                                      layers_outputs=layers_outputs,
                                                      grid_size=0.02)
    return p, order, inverse_order, x, x_res
