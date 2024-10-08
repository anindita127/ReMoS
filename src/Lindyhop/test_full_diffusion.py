import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import time
import torch
torch.cuda.empty_cache()
import torch.nn as nn

from cmath import nan
from collections import OrderedDict
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Lindyhop.argUtils import argparseNloop
from src.Lindyhop.LindyHop_dataloader import LindyHopDataset
from src.Lindyhop.models.MotionDiffuse_body import DiffusionTransformer as BodyDiffusionTransformer
from src.Lindyhop.models.MotionDiffusion_hand import DiffusionTransformer as HandDiffusionTransformer
from src.Lindyhop.models.transAE import VanillaTransformer
from src.Lindyhop.models.Gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from src.Lindyhop.skeleton import *
from src.Lindyhop.visualizer import plot_contacts3D
from src.tools.bookkeeper import *
from src.tools.calculate_ev_metrics import *
from src.tools.transformations import *
from src.tools.utils import makepath


# stat_metrics = CalculateMetricsDanceData()
def dist(x, y):
    # return torch.mean(x - y)
    return torch.mean(torch.cdist(x, y, p=2))

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

class Trainer:
    def __init__(self, args,split='test', num_body_jts = 27, num_hand_jts=42):
        torch.manual_seed(args.seed)
        self.model_path = args.model_path
        makepath(args.work_dir, isfile=False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
            self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
            gpu_brand = torch.cuda.get_device_name(args.cuda) if use_cuda else None
            gpu_count = torch.cuda.device_count() if args.use_multigpu else 1
            print('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        args_subset = ['exp', 'model', 'batch_size', 'frames']
        
        # load the body model
        args.load = os.path.join('save', 'Lindyhop', 'diffusion', 'exp_4_model_DiffusionTransformer_batchsize_32_frames_20_',
                            'exp_4_model_DiffusionTransformer_batchsize_32_frames_20_000600.p' )
        assert os.path.exists(args.load)
        self.book1 = BookKeeper(args, args_subset)
        self.batch_size = args.batch_size
        self.curriculum = args.curriculum
        self.scale = args.scale
        self.dtype = torch.float32
        self.body_model_epochs_completed = self.book1.last_epoch
        self.frames = args.frames 
        self.model = args.model

        self.testtime_split = split
        self.num_body_jts = num_body_jts
        self.num_hand_jts = num_hand_jts
        self.model_pose = BodyDiffusionTransformer(device=self.device,
                                           num_jts=self.num_body_jts,
                                           num_frames=self.frames,
                                           input_feats=args.input_feats,
                                        #    jt_latent_dim=args.jt_latent,
                                           latent_dim=args.d_model,
                                           num_heads=args.num_head,
                                           num_layers=args.num_layer,
                                           ff_size=args.d_ff,
                                           activations=args.activations
                                           ).to(self.device).float() 
        self.book1._load_model(self.model_pose, 'model_pose')    
        # load the fingers model
        args.load = os.path.join('save', 'Lindyhop', 'diffusion_hand', 'exp_35_model_DiffusionTransformer_batchsize_64_frames_20_',
                            'exp_35_model_DiffusionTransformer_batchsize_64_frames_20_000700.p' )
        assert os.path.exists(args.load)
        self.book2 = BookKeeper(args, args_subset)
        self.hand_model_pose = HandDiffusionTransformer(device=self.device,
                                           num_frames=self.frames,
                                           input_condn_feats=args.hand_input_condn_feats,
                                           input_feats=args.hand_out_feats,
                                           latent_dim=args.d_modelhand,
                                           num_heads=args.num_head_hands,
                                           num_layers=args.num_layer_hands,
                                           ff_size=args.d_ffhand,
                                           activations=args.activations
                                           ).to(self.device).float()     
        self.book2._load_model(self.hand_model_pose, 'model_pose')    
        self.hand_model_epochs_completed = self.book2.last_epoch
        trainable_count_body = sum(p.numel() for p in self.model_pose.parameters() if p.requires_grad)
        trainable_count_hand = sum(p.numel() for p in self.hand_model_pose.parameters() if p.requires_grad)
        
        self.diffusion_steps = args.diffusion_steps
        self.beta_scheduler = args.noise_schedule
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler_name = args.sampler
        self.sampler = create_named_schedule_sampler(self.sampler_name, self.diffusion)
        self.skel = InhouseStudioSkeleton()
        self.load_data_testtime(args)
        
        if is_eval:
            self.mot_enc = VanillaTransformer(args, njoints=27).to(self.device).float()    
            mot_enc_weights_path = os.path.join('save', 'Lindyhop', 'mot_enc', 'exp_82_model_transAE_batchsize_32_frames_20_',
                                'exp_82_model_transAE_batchsize_32_frames_20_000300.p' )
            m = torch.load(open(mot_enc_weights_path, 'rb'))
            self.mot_enc.load_state_dict(m['model_pose'])
            self.hand_mot_enc = VanillaTransformer(args, njoints=22).to(self.device).float()    
            hand_mot_enc_weights_path = os.path.join('save', 'Lindyhop', 'mot_enc', 'exp_94_model_transAE_batchsize_32_frames_20_',
                                'exp_94_model_transAE_batchsize_32_frames_20_000033.p' )
            m = torch.load(open(hand_mot_enc_weights_path, 'rb'))
            self.hand_mot_enc.load_state_dict(m['model_pose'])
        
    def load_data_testtime(self, args):
        self.ds_data = LindyHopDataset(args, window_size=self.frames, split=self.testtime_split)
        self.load_ds_data = DataLoader(self.ds_data, batch_size=100, shuffle=False, num_workers=0, drop_last=True)
               
    def generate_body(self, motion1, motion2=None, contact_maps=None):
        B, T, J, dim_pose = motion1.shape
        output = self.diffusion.p_sample_loop(
            self.model_pose,
            (B, T, J, dim_pose),
            clip_denoised=False,
            progress=True,
            pre_seq= motion2,
            model_kwargs={
                'motion1': motion1,
                'contact_maps': contact_maps,
                'spatial_guidance': True,
                'guidance_scale': 0.001
            })
        return output

    def generate_hand(self, motion1, motion2=None):
        B, T, _ = motion1.shape
        output = self.diffusion.p_sample_loop(
            self.hand_model_pose,
            (B, T, 66),
            clip_denoised=False,
            progress=True,
            pre_seq= motion2,
            model_kwargs={
                'motion1': motion1,
                'spatial_guidance': False
            })
       
        return output
    
    def root_relative_normalization(self, global_pose1, global_pose2=None):
        
        global_pose1 = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
        pose1_root_rel = global_pose1 - torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_body_jts, axis=-2)
        self.pose1_root_rel = pose1_root_rel / self.scale
        if global_pose2 is not None:
            global_pose2 = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
            pose2_root_rel = global_pose2 - torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_body_jts, axis=-2)
        self.pose2_root_rel = pose2_root_rel / self.scale
        
    
    def root_relative_unnormalization(self, pose1_normalized, pose2_normalized, gt_pose2_normalized=None):
        pose1_unnormalized = pose1_normalized * self.scale
        pose2_unnormalized = pose2_normalized * self.scale
        global_pose1 = pose1_unnormalized + torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_body_jts, axis=-2)
        global_pose2 = pose2_unnormalized + torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_body_jts, axis=-2)
        if gt_pose2_normalized is not None:
            gt_pose2_unnormalized = gt_pose2_normalized * self.scale
            gt_global_pose2 = gt_pose2_unnormalized + torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_body_jts, axis=-2)
            return global_pose1, global_pose2, gt_global_pose2
        return global_pose1, global_pose2

             
    def hand_relative_normalization(self, global_pose1, global_pose2, global_rot1):
        self.p1_rhand_wrist_pos = global_pose1[:, :, 18]
        self.p1_lhand_wrist_pos = global_pose1[:, :, 22]
        p1_rhand_wrist_pos = (global_pose1[:, :, 18] - self.p1_rhand_wrist_pos) / self.scale
        p1_lhand_wrist_pos = (global_pose1[:, :, 22] - self.p1_lhand_wrist_pos) / self.scale
        p2_rhand_wrist_pos = (global_pose2[:, :, 18] - self.p1_rhand_wrist_pos) / self.scale
        p2_lhand_wrist_pos = (global_pose2[:, :, 22] - self.p1_lhand_wrist_pos) / self.scale
        B = p1_rhand_wrist_pos.shape[0]
        T = p1_rhand_wrist_pos.shape[1]
        self.p1_rhand_rot = self.skel.select_bvh_joints(global_rot1.unsqueeze(0), original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only).reshape(B, T, -1)
        self.p1_lhand_rot = self.skel.select_bvh_joints(global_rot1.unsqueeze(0), original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only).reshape(B, T, -1)
        
        # create a contact map based on threshold of wrists
        self.contact_dist = torch.zeros(B, T, 4).to(self.p1_lhand_rot.device).float()
        self.contact_dist[:,:, 0] = ((p1_rhand_wrist_pos - p2_rhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 1] = ((p1_rhand_wrist_pos - p2_lhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 2] = ((p1_lhand_wrist_pos - p2_rhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 3] = ((p1_lhand_wrist_pos - p2_lhand_wrist_pos)**2).norm(dim=-1)
        
        self.input_condn = torch.cat((p1_rhand_wrist_pos, p1_lhand_wrist_pos,
                                      p2_rhand_wrist_pos, p2_lhand_wrist_pos,
                                      self.p1_rhand_rot, self.p1_lhand_rot, self.contact_dist), dim=-1)
        
    def hand_pose_relative_normalization(self, global_pose1, contact_maps, global_pose2=None):
        self.glob_p1_rhand_pos = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only)
        self.glob_p1_lhand_pos = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only)
        if global_pose2 is not None:
            self.glob_p2_rhand_pos = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only)
            self.glob_p2_lhand_pos = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only)
        self.p1_rhand_wrist_pos = self.glob_p1_rhand_pos[:, :, 0]
        self.p1_lhand_wrist_pos = self.glob_p1_lhand_pos[:, :, 0]
        self.p1_rhand_pos = (self.glob_p1_rhand_pos - torch.repeat_interleave(self.p1_rhand_wrist_pos.unsqueeze(-2), self.num_hand_jts, axis=-2))/self.scale
        self.p1_lhand_pos = (self.glob_p1_lhand_pos - torch.repeat_interleave(self.p1_lhand_wrist_pos.unsqueeze(-2), self.num_hand_jts, axis=-2))/self.scale
        B = self.p1_rhand_wrist_pos.shape[0]
        T = self.p1_rhand_wrist_pos.shape[1]
            
        self.contact_map = contact_maps.to(self.device).float()
        self.input_condn = torch.cat((self.p1_rhand_pos.reshape(B, T, -1), self.p1_lhand_pos.reshape(B, T, -1),
                                      self.contact_map), dim=-1)
    
    def hand_pose_relative_unnormalization(self, reaction_body_pose, normalized_reaction_hand_pose):
        B = reaction_body_pose.shape[0]
        global_rhand_wrist = reaction_body_pose[:, :, 18]
        global_lhand_wrist = reaction_body_pose[:, :, 22]
        normalized_rhand_reaction_pose = normalized_reaction_hand_pose[:, :, :33].reshape(B, self.frames, self.num_hand_jts, 3) * self.scale
        normalized_lhand_reaction_pose = normalized_reaction_hand_pose[:, :, 33:].reshape(B, self.frames, self.num_hand_jts, 3) * self.scale

        self.p2_rhand_pos = (normalized_rhand_reaction_pose + torch.repeat_interleave(global_rhand_wrist.unsqueeze(-2), self.num_hand_jts, axis=-2))
        self.p2_lhand_pos = (normalized_lhand_reaction_pose + torch.repeat_interleave(global_lhand_wrist.unsqueeze(-2), self.num_hand_jts, axis=-2))
 
         
    def test(self, plotfiles=True):
        self.model_pose.eval()
        self.hand_model_pose.eval()
        T = self.frames
        annot_dict = self.ds_data.annot_dict
        full_global_pose1 = torch.tensor(annot_dict['pose_canon_1'][0]).to(self.device).float()
        total_frames = 300 #len(full_global_pose1)

        full_contact_maps = torch.tensor(annot_dict['contacts'][0]).to(self.device).float()
        _, J, dim = full_global_pose1.shape

        
        global_action_pose = []
        global_reaction_pose = []
        global_gt_reaction_pose = []
        reaction_pose_out = None
        hand_reaction_pose_out = None
        for count in range(0, total_frames, T):
            if count+T > total_frames:
                break 
            
            start_time = time.time()
            global_pose1 = full_global_pose1[count:count+T].reshape(1, T, J, dim)

            contact_maps = full_contact_maps[count:count+T].reshape(1, T, 4)
               
            self.global_root_origin = global_pose1[:, :, 0].to(self.device).float() 
            self.root_relative_normalization(global_pose1)
            reaction_pose_out = self.generate_body(self.pose1_root_rel, motion2=reaction_pose_out, contact_maps= contact_maps)
            
            action_pose, reaction_pose = self.root_relative_unnormalization(self.pose1_root_rel, reaction_pose_out)
            
            self.hand_pose_relative_normalization(global_pose1=global_pose1, contact_maps=contact_maps)
            hand_reaction_pose_out = self.generate_hand(self.input_condn, motion2=hand_reaction_pose_out)
            self.hand_pose_relative_unnormalization(reaction_body_pose= reaction_pose, 
                                                    normalized_reaction_hand_pose=hand_reaction_pose_out)

            _global_action_pose = self.skel.revert_original_bvh_joints_poses(self.glob_p1_lhand_pos, self.skel.bvh_joint_reduced,
                                                                    self.skel.lh_fingers_only)
            _global_action_pose = self.skel.revert_original_bvh_joints_poses(self.glob_p1_rhand_pos, self.skel.bvh_joint_reduced,
                                                                  self.skel.rh_fingers_only, _global_action_pose)
            _global_action_pose = self.skel.revert_original_bvh_joints_poses(action_pose, self.skel.bvh_joint_reduced,
                                                                                  self.skel.body_only, _global_action_pose)
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(self.p2_lhand_pos, self.skel.bvh_joint_reduced,
                                                                    self.skel.lh_fingers_only)
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(self.p2_rhand_pos, self.skel.bvh_joint_reduced,
                                                                  self.skel.rh_fingers_only, rec_global_reaction_pose)
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(reaction_pose, self.skel.bvh_joint_reduced,
                                                                                  self.skel.body_only, rec_global_reaction_pose)

            global_reaction_pose.append(rec_global_reaction_pose)
            global_action_pose.append(_global_action_pose)
            print("--- %s seconds ---" % (time.time() - start_time))
       
        global_action_pose = torch.cat(global_action_pose, dim=1)
        global_reaction_pose = torch.cat(global_reaction_pose, dim=1)
       
        if plotfiles:
            plot_contacts3D(pose1=(global_action_pose[0].detach().cpu().numpy()),
                            pose2=(global_reaction_pose[0].detach().cpu().numpy()),
                            gt_pose2=None,
                            savepath='./save/LindyHop/render_result', kinematic_chain = 'reduced', onlyone=False)
    
    def evaluate_stat_metric(self):
        reaction_pose_out = None
        hand_reaction_pose_out = None
        gt_motion_embedding = []
        gt_hand_motion_embedding = []
        syn_motion_embedding = []
        syn_hand_motion_embedding = []
        global_action_pose = []
        global_action_pose = []
        global_reaction_pose = []
        global_gt_reaction_pose = [] 
        self.model_pose.eval()
        
        eval_tqdm = tqdm(self.load_ds_data, desc='eval' + ' {:.10f}'.format(0), leave=False, ncols=120)

        for count, batch in enumerate(eval_tqdm):
            global_pose1 = batch['pose_canon_1'].to(self.device).float()
           
            global_pose2 = batch['pose_canon_2'].to(self.device).float()
            
            self.contact_map = batch['contacts'].to(self.device).float()
            self.global_root_origin = batch['global_root_origin'].to(device).float()
            if global_pose1.shape[1] == 0:
                continue
            B = global_pose1.shape[0]
            T = global_pose1.shape[1]
            
            self.root_relative_normalization(global_pose1, global_pose2)
            reaction_pose_out = self.generate_body(self.pose1_root_rel, motion2=reaction_pose_out, contact_maps= self.contact_map)
            
            
            action_pose, reaction_pose, gt_reaction_pose = self.root_relative_unnormalization(self.pose1_root_rel, reaction_pose_out, self.pose2_root_rel)
            gt_motion_embedding_ = self.mot_enc.encode(gt_reaction_pose)
            syn_motion_embedding_ = self.mot_enc.encode(reaction_pose)
            
            self.hand_pose_relative_normalization(global_pose1=global_pose1, contact_maps=self.contact_map, global_pose2=global_pose2)
            hand_reaction_pose_out = self.generate_hand(self.input_condn, motion2=hand_reaction_pose_out)
            self.hand_pose_relative_unnormalization(reaction_body_pose= reaction_pose, 
                                                    normalized_reaction_hand_pose=hand_reaction_pose_out)
            gt_hand_motion_embedding_ = self.hand_mot_enc.encode(torch.cat((self.skel.select_bvh_joints(
                global_pose2, original_joint_order=self.skel.bvh_joint_order, new_joint_order=self.skel.rh_fingers_only),
                                                                            self.skel.select_bvh_joints(
                global_pose2, original_joint_order=self.skel.bvh_joint_order, new_joint_order=self.skel.lh_fingers_only)),dim = -2))
            syn_hand_motion_embedding_ = self.hand_mot_enc.encode(torch.cat((self.p2_rhand_pos, self.p1_lhand_pos), dim=-2))
            
            _global_action_pose = self.skel.revert_original_bvh_joints_poses(self.glob_p1_lhand_pos, self.skel.bvh_joint_reduced,
                                                                    self.skel.lh_fingers_only)
            _global_action_pose = self.skel.revert_original_bvh_joints_poses(self.glob_p1_rhand_pos, self.skel.bvh_joint_reduced,
                                                                  self.skel.rh_fingers_only, _global_action_pose)
            _global_action_pose = self.skel.revert_original_bvh_joints_poses(action_pose, self.skel.bvh_joint_reduced,
                                                                                  self.skel.body_only, _global_action_pose)
            gt_global_reaction_pose2 = self.skel.revert_original_bvh_joints_poses(self.glob_p1_lhand_pos, self.skel.bvh_joint_reduced,
                                                                    self.skel.lh_fingers_only)
            gt_global_reaction_pose2 = self.skel.revert_original_bvh_joints_poses(self.glob_p1_rhand_pos, self.skel.bvh_joint_reduced,
                                                                  self.skel.rh_fingers_only, gt_global_reaction_pose2)
            gt_global_reaction_pose2 = self.skel.revert_original_bvh_joints_poses(gt_reaction_pose, self.skel.bvh_joint_reduced,
                                                                                  self.skel.body_only, gt_global_reaction_pose2)
            
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(self.p2_lhand_pos, self.skel.bvh_joint_reduced,
                                                                    self.skel.lh_fingers_only)
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(self.p2_rhand_pos, self.skel.bvh_joint_reduced,
                                                                  self.skel.rh_fingers_only, rec_global_reaction_pose)
            rec_global_reaction_pose = self.skel.revert_original_bvh_joints_poses(reaction_pose, self.skel.bvh_joint_reduced,
                                                                                  self.skel.body_only, rec_global_reaction_pose)
            
            global_reaction_pose.append(rec_global_reaction_pose)
            global_action_pose.append(_global_action_pose)
            global_gt_reaction_pose.append(gt_global_reaction_pose2)
            gt_motion_embedding.append(gt_motion_embedding_.reshape(B*T, 128))
            syn_motion_embedding.append(syn_motion_embedding_.reshape(B*T, 128))
            gt_hand_motion_embedding.append(gt_hand_motion_embedding_.reshape(B*T, 128))
            syn_hand_motion_embedding.append(syn_hand_motion_embedding_.reshape(B*T, 128))
            
        global_action_pose = torch.cat(global_action_pose, dim=0).detach().cpu().numpy()
        global_reaction_pose = torch.cat(global_reaction_pose, dim=0).detach().cpu().numpy()
        global_gt_reaction_pose = torch.cat(global_gt_reaction_pose, dim=0).detach().cpu().numpy()
        gt_motion_embedding = torch.cat(gt_motion_embedding, dim=0).detach().cpu().numpy()
        syn_motion_embedding = torch.cat(syn_motion_embedding, dim=0).detach().cpu().numpy()
        gt_hand_motion_embedding = torch.cat(gt_hand_motion_embedding, dim=0).detach().cpu().numpy()
        syn_hand_motion_embedding = torch.cat(syn_hand_motion_embedding, dim=0).detach().cpu().numpy()
        mpjpe =  mean_l2di_(global_reaction_pose, global_gt_reaction_pose).item()
        jitter =  mean_l2di_(global_reaction_pose[:, 1:] - global_reaction_pose[:, :-1],
                             global_gt_reaction_pose[:, 1:] - global_gt_reaction_pose[:, :-1]).item()
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embedding)
        syn_mu, syn_cov = calculate_activation_statistics(syn_motion_embedding)
        gt_hand_mu, gt_hand_cov = calculate_activation_statistics(gt_hand_motion_embedding)
        syn_hand_mu, syn_hand_cov = calculate_activation_statistics(syn_hand_motion_embedding)
        fid = calculate_frechet_distance(gt_mu, gt_cov, syn_mu, syn_cov)
        hand_fid = calculate_frechet_distance(gt_hand_mu, gt_hand_cov, gt_hand_mu, gt_hand_cov)
        diversity = calculate_diversity(syn_motion_embedding, diversity_times=300)
        GT_diversity = calculate_diversity(gt_motion_embedding, diversity_times=300)
        print('mpjpe', mpjpe)
        print('jitter', jitter)
        print('FID', fid)
        print('Diversity', diversity)
        output_metrics_dict = {
            'mpjpe': float(mpjpe),
            'jitter': float(jitter),
            'diversity': float(diversity),
            'gt_diversity': float(GT_diversity),
            'fid': float(fid),
        }

        # Save JSON string to a text file
        savefile = makepath(os.path.join(args.load[:-2], self.testtime_split, 'metrics.txt'), isfile=True)
        
        with open(savefile, "w") as filep:
            json.dump(output_metrics_dict, filep)        
            
if __name__ == '__main__':
    args = argparseNloop()
    args.lambda_loss = {
        'fk': 1.0,
        'fk_vel': 1.0,
        'rot': 1.0,
        'rot_vel': 1.0,
        'kldiv': 1.0,
        'pos': 1e+3,
        'vel': 1e+1,
        'bone': 1.0,
        'foot': 0.0
    }
    is_eval = True     # 'True' to get the evaluation metrics, 'false' to test the motions visually.
    model_trainer = Trainer(args=args, split='test', num_body_jts=27, num_hand_jts=11)
    print("** Method Inititalization Complete **")
    if is_eval:
        model_trainer.evaluate_stat_metric()   
    else:
        model_trainer.test(plotfiles=True) 