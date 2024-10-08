import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


from torch.distributions.distribution import Distribution
from torch import Tensor
from typing import List, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
right_hand_finger_joints = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
left_hand_finger_joints = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
used_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43, 44, 65, 66, 67, 68]

def initialize_weights(m):
    std_dev = 0.02
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std_dev)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=std_dev)
        # nn.init.constant_(m.bias.data, 1e-5)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, std=std_dev)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=std_dev)
        # nn.init.constant_(m.bias.data, 1e-5)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=std_dev)  
        if m.bias is not None:
            nn.init.normal_(m.bias, std=std_dev)  

class motion_transformer(nn.Module): 
    def __init__(self, args, njoints=69, nfeats=3, num_frames=50, latent_dim=20,
                 hidden_dim=300, ff_size=1024, num_layers=6, num_heads=10, dropout=0.05,
                 ablation=None, activation="gelu"):
        super(motion_transformer, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.nframes = args.frames
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.latent_dim_root = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints*self.nfeats
        self.skelEmbedding = nn.Linear(self.input_feats, self.hidden_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.hidden_dim, self.input_feats)

        self.rootEmbedding = nn.Linear(3, self.latent_dim_root)
        
        self.sequence_pos_encoder_root = PositionalEncoding(self.latent_dim_root, self.dropout)

        seqTransEncoderLayer_root = nn.TransformerEncoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder_Root = nn.TransformerEncoder(seqTransEncoderLayer_root,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer_root = nn.TransformerDecoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder_Root = nn.TransformerDecoder(seqTransDecoderLayer_root,
                                                     num_layers=self.num_layers)
        
        self.finallayer_root = nn.Linear(self.latent_dim_root, 3)


    def forward(self, pose1, num_epochs=0):
        bs = pose1.shape[0]
        T = pose1.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = pose1.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries_)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)

        return output

    def sample(self, pose1):
        bs = pose1.shape[0]
        T = pose1.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        features_action = pose1.reshape(bs, T, -1)
        action_z = self.actionencoder(features_action, mask)
        reaction_z = torch.randn(bs, self.latent_dim_root).to(device)
        out_reaction = self.reactiondecoder(reaction_z, action_z, mask)
        return out_reaction


class transVAE_rot6d(nn.Module): 
    def __init__(self, args, njoints=69, nfeats=6, num_frames=50, latent_dim=20,
                 hidden_dim=300, ff_size=1024, num_layers=6, num_heads=10, dropout=0.05,
                 activation="gelu"):
        super(transVAE_rot6d, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.nframes = args.frames
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.latent_dim_root = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints*self.nfeats
        self.skelEmbedding = nn.Linear(self.input_feats, self.hidden_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.hidden_dim, self.input_feats)

        self.rootEmbedding = nn.Linear(3, self.latent_dim_root)
        
        self.sequence_pos_encoder_root = PositionalEncoding(self.latent_dim_root, self.dropout)

        seqTransEncoderLayer_root = nn.TransformerEncoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder_Root = nn.TransformerEncoder(seqTransEncoderLayer_root,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer_root = nn.TransformerDecoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder_Root = nn.TransformerDecoder(seqTransDecoderLayer_root,
                                                     num_layers=self.num_layers)
        
        self.finallayer_root = nn.Linear(self.latent_dim_root, 3)


    def forward(self, rot1, rot2, pose1_root, pose2_root, num_epochs=0):
        bs = rot1.shape[0]
        T = rot1.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = rot1.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries_)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)
        rec_loss_a2r =  F.l1_loss(rot2, output)
        
        # root movements
        x_root = pose1_root.reshape(bs, T, -1).permute((1, 0, 2))
        x_root_ = self.rootEmbedding(x_root)
       
        x1_root = self.sequence_pos_encoder_root(x_root_)
        z_root = self.seqTransEncoder_Root(x1_root, src_key_padding_mask=~mask)
        timequeries_root_ = torch.zeros(T, bs, self.latent_dim_root, device=z_root.device)
        timequeries_root = self.sequence_pos_encoder_root(timequeries_root_)
        output_root = self.seqTransDecoder_Root(tgt=timequeries_root, memory=z_root,
                                      tgt_key_padding_mask=~mask)
        output_root = self.finallayer_root(output_root).reshape(T, bs, -1)
        # zero for padded area
        output_root[~mask.T] = 0
        output_root = output_root
        rec_loss_root =  F.l1_loss(pose2_root.permute(1, 0, 2), output_root)
        return output, output_root, [rec_loss_a2r, rec_loss_root]

    def forward_diff(self, pose, timesteps, num_epochs=0):
        pose_root = pose[:,:, :3]
        rot = pose[:,:, 3:]
        bs = rot.shape[0]
        T = rot.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = rot.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        # x1 = self.sequence_pos_encoder(x_)
        x1 = positional_timestep_encoding(x_, timesteps, self.hidden_dim, device)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        # timequeries = self.sequence_pos_encoder(timequeries_)
        timequeries = positional_timestep_encoding(timequeries_, timesteps, self.hidden_dim, device)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)

        # root movements
        x_root = pose_root.reshape(bs, T, -1).permute((1, 0, 2))
        x_root_ = self.rootEmbedding(x_root)
       
        # x1_root = self.sequence_pos_encoder_root(x_root_)
        x1_root = positional_timestep_encoding(x_root_, timesteps, self.latent_dim_root, device)
        z_root = self.seqTransEncoder_Root(x1_root, src_key_padding_mask=~mask)
        timequeries_root_ = torch.zeros(T, bs, self.latent_dim_root, device=z_root.device)
        # timequeries_root = self.sequence_pos_encoder_root(timequeries_root_)
        timequeries_root = positional_timestep_encoding(timequeries_root_, timesteps, self.latent_dim_root, device)
        output_root = self.seqTransDecoder_Root(tgt=timequeries_root, memory=z_root,
                                      tgt_key_padding_mask=~mask)
        output_root = self.finallayer_root(output_root).reshape(T, bs, -1)
        # zero for padded area
        output_root[~mask.T] = 0
        output_root = output_root.permute(1, 0, 2)
        
        return torch.cat((output_root, output), -1)

class transVAE_rotmat(nn.Module): 
    def __init__(self, args, njoints=69, nfeats=9, num_frames=50, latent_dim=20,
                 hidden_dim=300, ff_size=1024, num_layers=6, num_heads=10, dropout=0.05,
                 activation="gelu"):
        super(transVAE_rotmat, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.nframes = args.frames
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.latent_dim_root = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints*self.nfeats
        self.skelEmbedding = nn.Linear(self.input_feats, self.hidden_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.hidden_dim, self.input_feats)

        self.rootEmbedding = nn.Linear(3, self.latent_dim_root)
        
        self.sequence_pos_encoder_root = PositionalEncoding(self.latent_dim_root, self.dropout)

        seqTransEncoderLayer_root = nn.TransformerEncoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder_Root = nn.TransformerEncoder(seqTransEncoderLayer_root,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer_root = nn.TransformerDecoderLayer(d_model=self.latent_dim_root,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder_Root = nn.TransformerDecoder(seqTransDecoderLayer_root,
                                                     num_layers=self.num_layers)
        
        self.finallayer_root = nn.Linear(self.latent_dim_root, 3)


    def forward(self, rot1, rot2, pose1_root, pose2_root, num_epochs=0):
        bs = rot1.shape[0]
        T = rot1.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = rot1.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries_)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)
        rec_loss_a2r =  F.l1_loss(rot2, output)
        
        # root movements
        x_root = pose1_root.reshape(bs, T, -1).permute((1, 0, 2))
        x_root_ = self.rootEmbedding(x_root)
       
        x1_root = self.sequence_pos_encoder_root(x_root_)
        z_root = self.seqTransEncoder_Root(x1_root, src_key_padding_mask=~mask)
        timequeries_root_ = torch.zeros(T, bs, self.latent_dim_root, device=z_root.device)
        timequeries_root = self.sequence_pos_encoder_root(timequeries_root_)
        output_root = self.seqTransDecoder_Root(tgt=timequeries_root, memory=z_root,
                                      tgt_key_padding_mask=~mask)
        output_root = self.finallayer_root(output_root).reshape(T, bs, -1)
        # zero for padded area
        output_root[~mask.T] = 0
        output_root = output_root
        rec_loss_root =  F.l1_loss(pose2_root.permute(1, 0, 2), output_root)
        return output, output_root, [rec_loss_a2r, rec_loss_root]



def loss_kldiv(mu, sig, wkl):
        L_KL = -0.5 * torch.mean(1 + torch.log(sig*sig) - mu.pow(2) - sig.pow(2))
        return wkl * L_KL


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

def positional_timestep_encoding(x_t, position, d_model, device='cuda'):
    pe = torch.zeros(len(position), d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).to(device)
    pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
    pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
    pe = pe.unsqueeze(0).to(device)
    x = x_t + pe
    return x

class VanillaTransformer(nn.Module): 
    def __init__(self, args, njoints=27, nfeats=3, num_frames=20, latent_dim=150,
                 hidden_dim=128, ff_size=512, num_layers=2, num_heads=2, dropout=0.05,
                 ablation=None, activation="gelu"):
        super(VanillaTransformer, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.nframes = args.frames
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints*self.nfeats
        self.skelEmbedding = nn.Linear(self.input_feats, self.hidden_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)      
        self.finallayer = nn.Linear(self.hidden_dim, self.input_feats)



    def forward(self, pose1, pose2):
        bs = pose1.shape[0]
        T = pose1.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = pose1.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries_)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)
        loss = self.mse_loss(output, pose2.reshape(bs, T, -1))
        return output

    def encode(self, pose):
        bs = pose.shape[0]
        T = pose.shape[1]
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(device)
        x = pose.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        return z.permute(1, 0, 2)
