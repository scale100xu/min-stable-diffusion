# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
import gzip
import argparse
import math
import os
import re
import torch
from functools import lru_cache
from collections import namedtuple

import numpy as np
from tqdm import tqdm

# ,
from torch.nn import Conv2d, Linear,  Module, SiLU, UpsamplingNearest2d,ModuleList,ZeroPad2d
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

device = "mps"

def apply_seq(seqs, x):
    for seq in seqs:
        x = seq(x)
    return x

def gelu(self):
    return 0.5 * self * (1 + torch.tanh(self * 0.7978845608 * (1 + 0.044715 * self * self)))

def quick_gelu(x):
    return x * torch.sigmoid(x * 1.702)

# class SiLU(Module):
#     def __init__(self):
#         super(SiLU, self).__init__()
#         self.gelu = quick_gelu
#     def forward(self,x ):
#         return self.gelu(x)
class Normalize(Module):
    def __init__(self, in_channels, num_groups=32, name="normalize"):
        super(Normalize, self).__init__()
        self.weight = Parameter(torch.ones(in_channels))
        self.bias = Parameter(torch.zeros(in_channels))
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.normSelf = None
        self.name = name


    def forward(self, x):

        # reshape for layernorm to work as group norm
        # subtract mean and divide stddev
        if self.num_groups == None:  # just layernorm
            return  F.layer_norm(x, self.weight.shape, self.weight, self.bias)
        else:
            x_shape = x.shape
            return F.group_norm(x, self.num_groups, self.weight, self.bias).reshape(*x_shape)

class AttnBlock(Module):
    def __init__(self, in_channels, name="AttnBlock"):
        super(AttnBlock, self).__init__()
        self.norm = Normalize(in_channels, name=name+"_norm_Normalize")
        self.q = Conv2d(in_channels, in_channels, 1)
        self.k = Conv2d(in_channels, in_channels, 1)
        self.v = Conv2d(in_channels, in_channels, 1)
        self.proj_out = Conv2d(in_channels, in_channels, 1)
        self.name = name

    # copied from AttnBlock in ldm repo
    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = q @ k
        w_ = w_ * (c ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = v @ w_
        h_ = h_.reshape(b, c, h, w)

        del q,k,v, w_
        return x + self.proj_out(h_)

class ResnetBlock(Module):
    def __init__(self, in_channels, out_channels=None, name="ResnetBlock"):
        super(ResnetBlock, self).__init__()
        self.norm1 = Normalize(in_channels, name=name+"_norm1_Normalize")
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, name=name+"_norm2_Normalize")
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x
        self.name = name

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h

class Mid(Module):
    def __init__(self, block_in, name="Mid"):
        super(Mid, self).__init__()
        self.block_1 = ResnetBlock(block_in, block_in, name=name+"_block_1_ResnetBlock")
        self.attn_1 = AttnBlock(block_in, name=name+"_attn_1_AttnBlock")
        self.block_2 = ResnetBlock(block_in, block_in, name=name+"_block_2_ResnetBlock")
        self.name = name

    def forward(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))

class Decoder(Module):
    def __init__(self, name="Decoder"):
        super(Decoder, self).__init__()

        self.conv_in = Conv2d(4, 512, 3, padding=1)
        self.mid = Mid(512, name=name+"_mid_Mid")

        # invert forward
        self.up = ModuleList([

            ResnetBlock(128, 128, name=name + "_up_0_block_2_ResnetBlock"),
            ResnetBlock(128, 128, name=name + "_up_0_block_1_ResnetBlock"),
            ResnetBlock(256, 128, name=name + "_up_0_block_0_ResnetBlock"),

            Conv2d(256, 256, 3, padding=1),
            UpsamplingNearest2d(scale_factor=2.0),
            ResnetBlock(256, 256, name=name + "_up_1_block_2_ResnetBlock"),
            ResnetBlock(256, 256, name=name + "_up_1_block_1_ResnetBlock"),
            ResnetBlock(512, 256, name=name + "_up_1_block_0_ResnetBlock"),

            Conv2d(512, 512, 3, padding=1),
            UpsamplingNearest2d(scale_factor=2.0),
            ResnetBlock(512, 512, name=name + "_up_2_block_2_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_up_2_block_1_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_up_2_block_0_ResnetBlock"),


            Conv2d(512, 512, 3, padding=1),
            UpsamplingNearest2d(scale_factor=2.0),
            ResnetBlock(512, 512, name=name + "_up_3_block_2_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_up_3_block_1_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_up_3_block_0_ResnetBlock"),]
        )

        self.norm_out = Normalize(128, name=name+"_norm_out_Normalize")
        self.conv_out = Conv2d(128, 3, 3, padding=1)
        self.name = name

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)

        for l in self.up[::-1]:
            x = l(x)

        return self.conv_out(F.silu(self.norm_out(x)))

class Encoder(Module):
    def __init__(self, name="Encoder"):
        super(Encoder, self).__init__()
        self.conv_in = Conv2d(3, 128, 3, padding=1)

        self.down = ModuleList([
            ResnetBlock(128, 128, name=name + "_down_block_0_0_ResnetBlock"),
            ResnetBlock(128, 128, name=name + "_down_block_0_1_ResnetBlock"),
            Conv2d(128, 128, 3, stride=2, padding=(0, 0)),
            ResnetBlock(128, 256, name=name + "_down_block_1_0_ResnetBlock"),
            ResnetBlock(256, 256, name=name + "_down_block_1_1_ResnetBlock"),
            Conv2d(256, 256, 3, stride=2, padding=(0, 0)),
            ResnetBlock(256, 512, name=name + "_down_block_2_0_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_down_block_2_1_ResnetBlock"),
            Conv2d(512, 512, 3, stride=2, padding=(0, 0)),
            ResnetBlock(512, 512, name=name + "_down_block_3_0_ResnetBlock"),
            ResnetBlock(512, 512, name=name + "_down_block_3_1_ResnetBlock"),
        ])

        self.mid = Mid(512, name=name+"_mid_Mid")
        self.norm_out = Normalize(512, name=name+"_norm_out_Normalize")
        self.conv_out = Conv2d(512, 8, 3, padding=1)
        self.name = name
        self.zero_pad2d_0_1 = ZeroPad2d((0,1,0,1))

    def forward(self, x):
        x = self.conv_in(x)

        for l in self.down:
            # x = l(x)
            if isinstance(l, Conv2d):
                x = l(self.zero_pad2d_0_1(x))
            else:
                x = l(x)
        x = self.mid(x)
        return self.conv_out(F.silu(self.norm_out(x)))

class AutoencoderKL(Module):
    def __init__(self, name="AutoencoderKL"):
        super(AutoencoderKL, self).__init__()
        self.encoder = Encoder(name=name+"_encoder_Encoder")
        self.decoder = Decoder(name=name+"_decoder_Decoder")
        self.quant_conv = Conv2d(8, 8, 1)
        self.post_quant_conv = Conv2d(4, 4, 1)
        self.name = name

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        print("latent", latent.shape)
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)

# not to be confused with ResnetBlock
class ResBlock(Module):
    def __init__(self, channels, emb_channels, out_channels, name="ResBlock"):
        super(ResBlock, self).__init__()
        self.in_layers = ModuleList([
            Normalize(channels, name=name +"_in_layers_Normalize"),
            SiLU(),
            Conv2d(channels, out_channels, 3, padding=1)
        ])
        self.emb_layers = ModuleList([
            SiLU(),
            Linear(emb_channels, out_channels)
        ])
        self.out_layers = ModuleList([
            Normalize(out_channels, name=name +"_out_layers_Normalize"),
            SiLU(),
            Conv2d(out_channels, out_channels, 3, padding=1)
        ])
        self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x
        self.name = name

    def forward(self, x, emb):
        h = apply_seq(self.in_layers, x)
        emb_out = apply_seq(self.emb_layers, emb)
        h = h + emb_out.reshape(*emb_out.shape, 1, 1)
        h = apply_seq(self.out_layers, h)
        ret = self.skip_connection(x) + h
        del emb_out, h

        return ret

class CrossAttention(Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head, name="CrossAttention"):
        super(CrossAttention, self).__init__()
        self.to_q = Linear(query_dim, n_heads * d_head, bias=False)
        self.to_k = Linear(context_dim, n_heads * d_head, bias=False)
        self.to_v = Linear(context_dim, n_heads * d_head, bias=False)
        self.scale = d_head ** -0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = ModuleList([Linear(n_heads * d_head, query_dim)])
        self.name = name

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q = q.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0, 2, 1,
                                                                              3)  # (bs, num_heads, time, head_size)
        k = k.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0, 2, 3,
                                                                              1)  # (bs, num_heads, head_size, time)
        v = v.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0, 2, 1,
                                                                              3)  # (bs, num_heads, time, head_size)

        score = q@k * self.scale
        score = F.softmax(score, dim=-1)  # (bs, num_heads, time, time)
        attention = (score@v).permute(0, 2, 1, 3)  # (bs, time, num_heads, head_size)

        h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
        del q,k,v,score

        return apply_seq(self.to_out, h_)

class GEGLU(Module):
    def __init__(self, dim_in, dim_out, name ="GEGLU"):
        super(GEGLU, self).__init__()
        self.proj = Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out
        self.name = name

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * quick_gelu(gate)

class FeedForward(Module):
    def __init__(self, dim, mult=4, name="FeedForward"):
        super(FeedForward, self).__init__()
        self.net = ModuleList([
            GEGLU(dim, dim * mult, name=name+"_net_0_GEGLU"),
            Linear(dim * mult, dim)
        ])

        self.name = name

    def forward(self, x):
        return apply_seq(self.net, x)

class BasicTransformerBlock(Module):
    def __init__(self, dim, context_dim, n_heads, d_head, name="BasicTransformerBlock"):
        super(BasicTransformerBlock, self).__init__()
        self.attn1 = CrossAttention(dim, dim, n_heads, d_head, name=name+"_attn1_CrossAttention")
        self.ff = FeedForward(dim, name=name+"_ff_FeedForward")
        self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head, name=name+"_attn2_CrossAttention")
        self.norm1 = Normalize(dim, num_groups=None, name=name+"_norm1_Normalize")
        self.norm2 = Normalize(dim, num_groups=None, name=name+"_norm2_Normalize")
        self.norm3 = Normalize(dim, num_groups=None, name=name+"_norm3_Normalize")
        self.name = name


    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(Module):
    def __init__(self, channels, context_dim, n_heads, d_head, name="SpatialTransformer"):
        super(SpatialTransformer, self).__init__()
        self.norm = Normalize(channels, name=name+"_norm_Normalize")
        assert channels == n_heads * d_head
        self.proj_in = Conv2d(channels, n_heads * d_head, 1)
        self.transformer_blocks = ModuleList([BasicTransformerBlock(channels, context_dim, n_heads, d_head, name=name+"_transformer_blocks_0_BasicTransformerBlock")])
        self.proj_out = Conv2d(n_heads * d_head, channels, 1)
        self.name = name


    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        ret = self.proj_out(x) + x_in
        del x_in, x
        return ret

class Downsample(Module):
    def __init__(self, channels, name = "Downsample"):
        super(Downsample, self).__init__()
        self.op = Conv2d(channels, channels, 3, stride=2, padding=1)
        self.name = name

    def forward(self, x):
        return self.op(x)

class Upsample(Module):
    def __init__(self, channels, name ="Upsample"):
        super(Upsample, self).__init__()
        self.conv = Conv2d(channels, channels, 3, padding=1)
        self.name = name

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps.cpu().numpy() * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return Tensor(embedding).to(device).reshape(1, -1)

class GroupGap(Module):
    def __init__(self):
        super(GroupGap, self).__init__()

class UNetModel(Module):
    def __init__(self,name = "UNetModel"):
        super(UNetModel, self).__init__()
        self.time_embed = ModuleList([
            Linear(320, 1280),
            SiLU(),
            Linear(1280, 1280),
        ])
        self.input_blocks = ModuleList([
            Conv2d(4, 320, kernel_size=3, padding=1),
            GroupGap(),

            # TODO: my head sizes and counts are a guess
            ResBlock(320, 1280, 320, name=name+"_input_blocks_1_ResBlock"),
            SpatialTransformer(320, 768, 8, 40,name=name+"_input_blocks_1_SpatialTransformer"),
            GroupGap(),

            ResBlock(320, 1280, 320, name=name+"_input_blocks_2_ResBlock"),
            SpatialTransformer(320, 768, 8, 40,name=name+"_input_blocks_2_SpatialTransformer"),
            GroupGap(),

            Downsample(320, name=name+"_input_blocks_3_Downsample"),
            GroupGap(),

            ResBlock(320, 1280, 640, name=name+"_input_blocks_4_ResBlock"),
            SpatialTransformer(640, 768, 8, 80, name=name+"_input_blocks_4_SpatialTransformer"),
            GroupGap(),

            ResBlock(640, 1280, 640, name=name+"_input_blocks_5_ResBlock"),
            SpatialTransformer(640, 768, 8, 80, name=name+"_input_blocks_5_SpatialTransformer"),
            GroupGap(),

            Downsample(640, name=name+"_input_blocks_6_Downsample"),
            GroupGap(),

            ResBlock(640, 1280, 1280, name=name+"_input_blocks_7_ResBlock"),
            SpatialTransformer(1280, 768, 8, 160,  name=name+"_input_blocks_7_SpatialTransformer"),
            GroupGap(),

            ResBlock(1280, 1280, 1280, name=name+"_input_blocks_8_ResBlock"),
            SpatialTransformer(1280, 768, 8, 160,  name=name+"_input_blocks_8_SpatialTransformer"),
            GroupGap(),

            Downsample(1280,name=name+"_input_blocks_9_Downsample"),
            GroupGap(),

            ResBlock(1280, 1280, 1280, name=name+"_input_blocks_10_ResBlock"),
            GroupGap(),

            ResBlock(1280, 1280, 1280, name=name+"_input_blocks_11_ResBlock"),
            GroupGap(),
        ])
        self.middle_block = ModuleList([
            ResBlock(1280, 1280, 1280, name=name+"_middle_block_1_ResBlock"),
            SpatialTransformer(1280, 768, 8, 160, name=name+"_middle_block_2_SpatialTransformer"),
            ResBlock(1280, 1280, 1280, name=name+"_middle_block_3_ResBlock")
        ])
        self.output_blocks = ModuleList([
            GroupGap(),
            ResBlock(2560, 1280, 1280,  name=name+"_output_blocks_1_ResBlock"),

            GroupGap(),
            ResBlock(2560, 1280, 1280,  name=name+"_output_blocks_2_ResBlock"),

            GroupGap(),
            ResBlock(2560, 1280, 1280,  name=name+"_output_blocks_3_ResBlock"),
             Upsample(1280,  name=name+"_output_blocks_3_Upsample"),

            GroupGap(),
            ResBlock(2560, 1280, 1280,  name=name+"_output_blocks_4_ResBlock"),
             SpatialTransformer(1280, 768, 8, 160,  name=name+"_output_blocks_4_SpatialTransformer"),

            GroupGap(),
            ResBlock(2560, 1280, 1280,  name=name+"_output_blocks_5_ResBlock"),
             SpatialTransformer(1280, 768, 8, 160,  name=name+"_output_blocks_5_SpatialTransformer"),

            GroupGap(),
            ResBlock(1920, 1280, 1280,  name=name+"_output_blocks_6_ResBlock"),
             SpatialTransformer(1280, 768, 8, 160,  name=name+"_output_blocks_6_SpatialTransformer"),
             Upsample(1280,  name=name+"_output_blocks_6_Upsample"),

            GroupGap(),
            ResBlock(1920, 1280, 640,  name=name+"_output_blocks_7_ResBlock"),
             SpatialTransformer(640, 768, 8, 80,  name=name+"_output_blocks_7_SpatialTransformer"),  # 6

            GroupGap(),
            ResBlock(1280, 1280, 640,  name=name+"_output_blocks_8_ResBlock"),
             SpatialTransformer(640, 768, 8, 80,  name=name+"_output_blocks_8_SpatialTransformer"),

            GroupGap(),
            ResBlock(960, 1280, 640,  name=name+"_output_blocks_9_ResBlock"),
            SpatialTransformer(640, 768, 8, 80,  name=name+"_output_blocks_9_SpatialTransformer"),
            Upsample(640,  name=name+"_output_blocks_9_Upsample"),

            GroupGap(),
            ResBlock(960, 1280, 320,  name=name+"_output_blocks_10_ResBlock"),
            SpatialTransformer(320, 768, 8, 40,  name=name+"_output_blocks_10_SpatialTransformer"),

            GroupGap(),
            ResBlock(640, 1280, 320,  name=name+"_output_blocks_11_ResBlock"),
             SpatialTransformer(320, 768, 8, 40,  name=name+"_output_blocks_11_SpatialTransformer"),

            GroupGap(),
            ResBlock(640, 1280, 320,  name=name+"_output_blocks_12_ResBlock"),
            SpatialTransformer(320, 768, 8, 40,  name=name+"_output_blocks_12_SpatialTransformer"),]

        )
        self.out = ModuleList([
            Normalize(320, name=name+"_out_1_Normalize"),
            SiLU(),
            Conv2d(320, 4, kernel_size=3, padding=1)
        ])

        self.name = name


    def forward(self, x, timesteps=None, context=None):
        # TODO: real time embedding
        t_emb = timestep_embedding(timesteps, 320)
        emb = apply_seq(self.time_embed, t_emb)

        def run(x, bb):
            if isinstance(bb, ResBlock):
                x = bb(x, emb)
            elif isinstance(bb, SpatialTransformer):
                x = bb(x, context)
            else:
                x = bb(x)
            return x

        saved_inputs = []
        for i, b in enumerate(self.input_blocks):
            # print("input block", i)
            if isinstance(b, GroupGap):
                saved_inputs.append(x)
                continue
            x = run(x, b)


        for bb in self.middle_block:
            x = run(x, bb)


        for i, b in enumerate(self.output_blocks):
            # print("output block", i)
            if isinstance(b, GroupGap):
               x = torch.cat([x,saved_inputs.pop()], dim=1)
               continue
            x = run(x, b)

        return apply_seq(self.out, x)

class CLIPMLP(Module):
    def __init__(self, name ="CLIPMLP"):
        super(CLIPMLP, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.name = name

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = quick_gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class CLIPAttention(Module):
    def __init__(self, name="CLIPAttention"):
        super(CLIPAttention, self).__init__()
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = Linear(self.embed_dim, self.embed_dim)
        self.v_proj = Linear(self.embed_dim, self.embed_dim)
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
        self.name = name

    def _shape(self, tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, causal_attention_mask):
        bsz, tgt_len, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        src_len = key_states.shape[1]
        value_states = value_states.reshape(*proj_shape)

        attn_weights = query_states @ key_states.permute(0, 2, 1)

        attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ value_states

        attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        del query_states, key_states, value_states, attn_weights
        return attn_output

class CLIPEncoderLayer(Module):
    def __init__(self, name="CLIPEncoderLayer"):
        super(CLIPEncoderLayer, self).__init__()
        self.layer_norm1 = Normalize(768, num_groups=None, name=name+"_Normalize_0")
        self.self_attn = CLIPAttention(name=name+"_CLIPAttention_0")
        self.layer_norm2 = Normalize(768, num_groups=None,name=name+"_Normalize_1")
        self.mlp = CLIPMLP(name=name+"_CLIPMLP_0")
        self.name = name

    def forward(self, hidden_states, causal_attention_mask):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        del residual

        return hidden_states

class CLIPEncoder(Module):
    def __init__(self, name="CLIPEncoder"):
        super(CLIPEncoder, self).__init__()
        self.layers = ModuleList([CLIPEncoderLayer(name=name+"_"+str(i)) for i in range(12)])
        self.name = name

    def forward(self, hidden_states, causal_attention_mask):
        for i, l in enumerate(self.layers):
            hidden_states = l(hidden_states, causal_attention_mask)
        return hidden_states

class CLIPTextEmbeddings(Module):
    def __init__(self, name="CLIPTextEmbeddings"):
        super(CLIPTextEmbeddings, self ).__init__()
        self.token_embedding_weight = Parameter(torch.zeros(49408, 768))
        self.position_embedding_weight = Parameter(torch.zeros(77, 768))
        self.name = name

    def forward(self, input_ids, position_ids):
        # TODO: actually support batches
        inputs = torch.zeros((1, len(input_ids), 49408))
        inputs = inputs.to(device)
        positions = torch.zeros((1, len(position_ids), 77))
        positions = positions.to(device)
        for i, x in enumerate(input_ids): inputs[0][i][x] = 1
        for i, x in enumerate(position_ids): positions[0][i][x] = 1
        inputs_embeds = inputs @ self.token_embedding_weight
        position_embeddings = positions @ \
                              self.position_embedding_weight
        return inputs_embeds + position_embeddings

class CLIPTextTransformer(Module):
    def __init__(self, name="CLIPTextTransformer"):
        super(CLIPTextTransformer, self).__init__()
        self.embeddings = CLIPTextEmbeddings(name=name+"_CLIPTextEmbeddings_0")
        self.encoder = CLIPEncoder(name=name+"_CLIPEncoder_0")
        self.final_layer_norm = Normalize(768, num_groups=None, name=name+"_CLIPTextTransformer_normalizer_0")
        # 上三角都是 -inf 值
        triu = np.triu(np.ones((1, 1, 77, 77), dtype=np.float32) * -np.inf, k=1)
        self.causal_attention_mask = Tensor(triu).to(device)
        self.name = name

    def forward(self, input_ids):
        x = self.embeddings(input_ids, list(range(len(input_ids))))
        x = self.encoder(x, self.causal_attention_mask)
        return self.final_layer_norm(x)

# Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "./clip_tokenizer/bpe_simple_vocab_16e6.txt.gz")

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

import threading

class ClipTokenizer:
    _instance_lock = threading.Lock()
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""",
                                         re.IGNORECASE)

    @classmethod
    def instance(cls, *args, **kwargs):
        with ClipTokenizer._instance_lock:
            if not hasattr(ClipTokenizer, "_instance"):
                ClipTokenizer._instance = ClipTokenizer(*args, **kwargs)
        return ClipTokenizer._instance

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(text.strip()).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # Truncation, keeping two slots for start and end tokens.
        if len(bpe_tokens) > 75:
            bpe_tokens = bpe_tokens[:75]
        return [49406] + bpe_tokens + [49407] * (77 - len(bpe_tokens) - 1)
class StableDiffusion(Module):
    _instance_lock = threading.Lock()
    def __init__(self, name="StableDiffusion"):
        super(StableDiffusion, self).__init__()
        self.betas = Parameter(torch.zeros(1000))
        self.alphas_cumprod = Parameter(torch.zeros(1000))
        self.alphas_cumprod_prev = Parameter(torch.zeros(1000))
        self.sqrt_alphas_cumprod = Parameter(torch.zeros(1000))
        self.sqrt_one_minus_alphas_cumprod = Parameter(torch.zeros(1000))
        self.log_one_minus_alphas_cumprod = Parameter(torch.zeros(1000))
        self.sqrt_recip_alphas_cumprod = Parameter(torch.zeros(1000))
        self.sqrt_recipm1_alphas_cumprod = Parameter(torch.zeros(1000))
        self.posterior_variance = Parameter(torch.zeros(1000))
        self.posterior_log_variance_clipped = Parameter(torch.zeros(1000))
        self.posterior_mean_coef1 = Parameter(torch.zeros(1000))
        self.posterior_mean_coef2 = Parameter(torch.zeros(1000))
        self.unet = UNetModel(name=name+"_unet")
        self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model=self.unet)
        self.first_stage_model = AutoencoderKL(name=name+"_AutoencoderKL")
        self.text_decoder = CLIPTextTransformer(name=name+"_CLIPTextTransformer")
        self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(
            transformer=namedtuple("Transformer", ["text_model"])(text_model=self.text_decoder))
        self.name = name


    @classmethod
    def instance(cls, *args, **kwargs):
        with StableDiffusion._instance_lock:
            if not hasattr(StableDiffusion, "_instance"):
                StableDiffusion._instance = StableDiffusion(*args, **kwargs)
        return StableDiffusion._instance
    # TODO: make forward run the model

# Set Numpy and PyTorch seeds



class Args(object):
    def __init__(self, phrase, steps, model_type, guidance_scale, img_width, img_height, seed, device, model_file, input_image:str="", input_mask:str="", input_image_strength=0.5, unphrase=""):
        self.phrase = phrase
        self.steps = steps
        self.model_type = model_type
        self.scale = guidance_scale
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.seed = seed
        self.device = device
        self.model_file = model_file
        self.input_image = input_image
        self.input_mask = input_mask
        self.input_image_strength = input_image_strength
        self.unphrase = unphrase

from PIL import Image

class Generate2img(Module):
    _instance_lock = threading.Lock()
    def __init__(self, args: Args):
        super(Generate2img, self).__init__()
        self.is_load_model=False
        self.args = args
        self.model = StableDiffusion().instance()
        self.get_input_image_tensor()
        # self.get_input_mask_tensor()


    def get_input_image_tensor(self):
        if self.args.input_image!="":
            input_image = Image.open(args.input_image).convert("RGB").resize((self.args.img_width, self.args.img_height), resample=Image.Resampling.LANCZOS)
            self.input_image_array = torch.from_numpy(np.array(input_image)).to(device)
            self.input_image_tensor = torch.from_numpy((np.array(input_image, dtype=np.float32)[None, ..., :3]/ 255.0*2.0-1))
            self.input_image_tensor = self.input_image_tensor.permute(0, 3, 1, 2) # bs, channel, height, width
        else:
            self.input_image_tensor = None
        return self.input_image_tensor


    @classmethod
    def instance(cls, *args, **kwargs):
        with Generate2img._instance_lock:
            if not hasattr(Generate2img, "_instance"):
                Generate2img._instance = Generate2img(*args, **kwargs)
        return Generate2img._instance

    def load_model(self):
        if self.args.model_file != "" and self.is_load_model==False:
            net = torch.load(self.args.model_file )
            self.model.load_state_dict(net)
            self.model = self.model.to(device)
            self.is_load_model=True

    def get_token_encode(self, phrase):
        tokenizer = ClipTokenizer().instance()
        phrase = tokenizer.encode(phrase)
        # phrase = phrase + [49407] * (77 - len(phrase))
        with torch.no_grad():
            context = self.model.text_decoder(phrase)
            return context.to(self.args.device)
    def forward(self, phrase:str):
        self.set_seeds(True)
        self.load_model()
        context = self.get_token_encode(phrase)
        unconditional_context = self.get_token_encode(self.args.unphrase)

        timesteps = list(np.arange(1, 1000, 1000 // self.args.steps))
        print(f"running for {timesteps} timesteps")
        alphas = [self.model.alphas_cumprod[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        latent_width = int(self.args.img_width) // 8
        latent_height = int(self.args.img_height) // 8


        input_image_latent = None
        input_img_noise_t = None
        if self.input_image_tensor!=None:
            noise_index = int(len(timesteps) * self.args.input_image_strength)
            if noise_index >= len(timesteps):
                noise_index = noise_index - 1
            input_img_noise_t = timesteps[noise_index]
            with torch.no_grad():
                filter = lambda x:x[:,:4,:,:] * 0.18215
                input_image_latent = self.model.first_stage_model.encoder(self.input_image_tensor.to(device))
                input_image_latent = self.model.first_stage_model.quant_conv(input_image_latent)
                input_image_latent = filter(input_image_latent) # only the means

        # start with random noise
        latent = self.get_noise_latent( 1,  latent_height, latent_width, input_image_latent, input_img_noise_t, None)

        latent = latent.to(self.args.device)

        with torch.no_grad():
            # this is diffusion
            for index, timestep in (t := tqdm(list(enumerate(timesteps))[::-1])):
                t.set_description("%3d %3d" % (index, timestep))
                e_t = self.get_model_latent_output(latent.clone(), timestep, self.model.unet, context.clone(),
                                       unconditional_context.clone())
                x_prev, pred_x0 = self.get_x_prev_and_pred_x0(latent, e_t, index, alphas, alphas_prev)
                latent = x_prev
                # e_t_next = get_model_output(x_prev)
                # e_t_prime = (e_t + e_t_next) / 2
                # x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
        decode = self.latent_decode(latent, latent_height, latent_width)

        return decode


    def get_x_prev_and_pred_x0(self, x, e_t, index, alphas, alphas_prev):
                temperature = 1
                a_t, a_prev = alphas[index], alphas_prev[index]
                sigma_t = 0
                sqrt_one_minus_at = math.sqrt(1 - a_t)
                # print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

                pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

                # direction pointing to x_t
                dir_xt = math.sqrt(1. - a_prev - sigma_t ** 2) * e_t
                noise = sigma_t * torch.randn(*x.shape) * temperature

                x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt  # + noise
                return x_prev, pred_x0

    def get_model_latent_output(self, latent, t, unet, context, unconditional_context):
            timesteps = torch.Tensor([t])
            timesteps = timesteps.to(self.args.device)
            unconditional_latent = unet(latent, timesteps, unconditional_context)
            latent = unet(latent, timesteps, context)

            unconditional_guidance_scale = self.args.scale
            e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
            del unconditional_latent, latent, timesteps, context
            return e_t


    def add_noise(self, x , t , noise=None ):
        # batch_size, channel, h, w = x.shape
        if noise is None:
            noise = torch.normal(0,1, size=(x.shape))
        # sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
        sqrt_alpha_prod = self.model.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.model.sqrt_one_minus_alphas_cumprod[t]
        # sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return  sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise.to(device)

    def get_noise_latent(self, batch_size, latent_height, latent_width, input_image_latent=None, input_img_noise_t=None, noise=None):

        if input_image_latent is None:
            latent = torch.normal(0,1, size=(batch_size, 4, latent_height, latent_width))
            # latent = torch.randn((batch_size, 4, latent_height, latent_width))
        else:
            latent = self.add_noise(input_image_latent, input_img_noise_t, noise)
        return latent.to(device)

    def latent_decode(self, latent, latent_height, latent_width):
        # upsample latent space to image with autoencoder
        # x = model.first_stage_model.post_quant_conv( 8* latent)
        x = self.model.first_stage_model.post_quant_conv(1 / 0.18215 * latent)
        x = x.to(self.args.device)
        x = self.model.first_stage_model.decoder(x)
        x = x.to(self.args.device)

        # make image correct size and scale
        x = (x + 1.0) / 2.0
        x = x.reshape(3, latent_height * 8, latent_width * 8).permute(1, 2, 0)
        decode_latent = (x.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        return decode_latent
    def decode_latent2img(self, decode_latent):
        # save image
        img = Image.fromarray(decode_latent, mode="RGB")
        return img

    def set_seeds(self, cuda):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if cuda:
            torch.cuda.manual_seed_all(self.args.seed)
@lru_cache()
def generate2img(phrase, steps, model_file, guidance_scale, img_width, img_height, seed, device, input_image, input_mask, input_image_strength=0.5, unphrase=""):
    try:
        args = Args(phrase, steps, None, guidance_scale, img_width, img_height, seed, device, model_file, input_image, input_mask, input_image_strength, unphrase)
        im = Generate2img.instance(args).forward(args.phrase)
        im = Generate2img.instance(args).decode_latent2img(im)
    finally:
        pass
    return im

# this is sd-v1-4.ckpt
FILENAME = "/tmp/stable_diffusion_v1_4.pt"
# this is sd-v1-5.ckpt
# FILENAME = "/tmp/stable_diffusion_v1_5.pt"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Stable Diffusion',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--steps', type=int, default=50, help="Number of steps in diffusion")
    parser.add_argument('--phrase', type=str, default="anthropomorphic cat portrait art ", help="Phrase to render")
    parser.add_argument('--unphrase', type=str, default="", help="unconditional Phrase to render")
    parser.add_argument('--out', type=str, default="/tmp/rendered.png", help="Output filename")
    parser.add_argument('--scale', type=float, default=7.5,  help="unconditional guidance scale")
    parser.add_argument('--model_file', type=str, default="../min-stable-diffusion-pt/mdjrny-v4.pt",  help="model weight file")
    parser.add_argument('--img_width', type=int, default=512,  help="output image width")
    parser.add_argument('--img_height', type=int, default=512,  help="output image height")
    parser.add_argument('--seed', type=int, default=443,  help="random seed")
    parser.add_argument('--device_type', type=str, default="cpu",  help="device type, support: cpu;cuda;mps")
    parser.add_argument('--input_image', type=str, default="",  help="input image file")
    args = parser.parse_args()
    device = args.device_type

    im = generate2img(args.phrase, args.steps, args.model_file, args.scale, args.img_width, args.img_height, args.seed, args.device_type, args.input_image, "", 1, args.unphrase)
    print(f"saving {args.out}")
    im.save(args.out)
