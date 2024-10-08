"""
Copyright 2021 S-Lab
"""

import matplotlib.pylab as plt
import random
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
from torch.nn import functional

import math
body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
hand = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
hand_full = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='bwr')
    plt.clim(-10, 10)
    plt.colorbar()
    plt.show()
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

         
        for pos in range(max_len):
            for i in range(0, d_model-1, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))


        pe = pe.unsqueeze(0)        # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def timestep_embedding(timesteps, dim, freqs):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.latent_dim//2, dtype=torch.float32) / (self.latent_dim//2)).to(device)
        
    # timesteps= timesteps.to('cpu')
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h



class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
    


class TemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask, eps=1e-8):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / max(math.sqrt(D // H), eps)
        attention = attention * src_mask.unsqueeze(-1)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y, attention

class TemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, mot1_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.mot1_norm = nn.LayerNorm(mot1_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(mot1_latent_dim, latent_dim)
        self.value = nn.Linear(mot1_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, src_mask, emb, eps=1e-8):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.mot1_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / max(math.sqrt(D // H), eps)
        attention = attention * src_mask.unsqueeze(-1)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.mot1_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y, attention

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 mot1_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = TemporalCrossAttention(
            seq_len, latent_dim, mot1_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x, s_attn = self.sa_block(x, emb, src_mask)
        x, c_attn = self.ca_block(x, xf, src_mask, emb)
        x = self.ffn(x, emb)
        return x, s_attn, c_attn


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 device= 'cuda',
                 num_jts=27,
                 num_frames=100,
                 input_feats=3,
                 latent_dim=32,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.05,
                 activations="gelu", 
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames 
        self.num_jts = num_jts
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.latent_dim//2, dtype=torch.float32) / (self.latent_dim//2)).to(device)
        self.dropout = dropout
        self.activation = activations
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim 
        self.spatio_temp = self.num_frames * self.num_jts
        
        # encode motion 1
        self.motion1_pre_proj = nn.Linear(self.input_feats, self.latent_dim)
        self.m1_temporal_pos_encoder = PositionalEncoding(d_model=self.latent_dim, dropout=self.dropout, max_len=self.spatio_temp)
        mot1TransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
            activation='gelu')
        self.mot1TransEncoder = nn.TransformerEncoder(
            mot1TransEncoderLayer,
            num_layers=2)
        self.mot1_ln = nn.LayerNorm(latent_dim)
        #Classifier-free guidance
        # self.null_cond = nn.Parameter(torch.randn(self.num_frames * self.num_jts, latent_dim))
 
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # motion2 decoding
        self.motion2_pre_proj = nn.Linear(self.input_feats, self.latent_dim)
        self.m2_temporal_pos_encoder = PositionalEncoding(d_model=self.latent_dim, dropout=self.dropout, max_len=self.spatio_temp)
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
            TemporalDiffusionTransformerDecoderLayer(
                seq_len=self.spatio_temp,
                latent_dim=latent_dim,
                mot1_latent_dim=latent_dim,
                time_embed_dim=self.time_embed_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout
            )
        )
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
    
        
    def generate_src_mask(self, tgt):
        length = tgt.size(1)
        src_mask = (1 - torch.triu(torch.ones(1, length, length), diagonal=1))
        return src_mask

    # def forward(self, motion2, timesteps, length=None, motion1=None, xf_out=None, contact_map=None):
    def forward(self, motion2, timesteps, motion1=None, contact_maps=None, spatial_guidance=None, guidance_scale=0):
        """
        x: B, T, D
        """
        B, T, J, _ = motion1.shape
        m1 = self.motion1_pre_proj(motion1)    # GCN
        m2 = self.motion2_pre_proj(motion2)    # GCN
        m1 = m1.reshape(B, T*J, -1)
        m2 = m2.reshape(B, T*J, -1)
        src_mask = self.generate_src_mask(m2).to(m2.device)
        
        m1_pe = self.m1_temporal_pos_encoder(m1)
        m1_cond = self.mot1_ln(self.mot1TransEncoder(m1_pe))
        # null_cond = torch.repeat_interleave(
        #     self.null_cond.to(m2.device).unsqueeze(0), B, dim=0)
        # m1_enc = m1_cond if random.random() > 0.25 else null_cond
        m1_enc = m1_cond
        m2_pe = self.m2_temporal_pos_encoder(m2)
        emb = self.time_embed(timestep_embedding(
            timesteps, self.latent_dim, self.freqs) ) 
        h_pe = m2_pe
        for module in self.temporal_decoder_blocks:
            h_pe, s_attn, c_attn = module(h_pe, m1_enc, emb, src_mask)
        output = self.out(h_pe).view(B, T, J, -1).contiguous()
        # if timesteps == 1:
        #     c_attn_map = c_attn[0,:,:,0].cpu().detach().numpy()
        #     # ax = sns.heatmap(c_attn_map, vmin=-15, vmax=15, linewidth=2)
        #     # plt.show()
        #     heatmap2d(c_attn_map)
        #     tmp=1
        return output, s_attn, c_attn
    
    def get_motion_embedding(self, motion1):
        B, T, J, _ = motion1.shape
        m1 = self.motion1_pre_proj(motion1)    # GCN
        m1 = m1.reshape(B, T*J, -1)
        m1_pe = self.m1_temporal_pos_encoder(m1)
        m1_cond = self.mot1_ln(self.mot1TransEncoder(m1_pe))
        return m1_cond