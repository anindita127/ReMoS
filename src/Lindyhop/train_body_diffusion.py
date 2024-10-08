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
from src.Lindyhop.models.MotionDiffuse_body import *
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
from src.tools.transformations import *
from src.tools.utils import makepath

right_side = [15, 16, 17, 18]
left_side = [19, 20, 21, 22]
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
    def __init__(self, args, is_train=True, split='test', JT_POSITION=False, num_jts = 69):
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
        self.book = BookKeeper(args, args_subset)
        self.args = self.book.args
        self.batch_size = args.batch_size
        self.curriculum = args.curriculum
        self.scale = args.scale
        self.dtype = torch.float32
        self.epochs_completed = self.book.last_epoch
        self.frames = args.frames 
        self.model = args.model
        self.lambda_loss = args.lambda_loss
        self.testtime_split = split
        self.num_jts = num_jts
        self.model_pose = eval(args.model)(device=self.device,
                                           num_jts=self.num_jts,
                                           num_frames=self.frames,
                                           input_feats=args.input_feats,
                                        #    jt_latent_dim=args.jt_latent,
                                           latent_dim=args.d_model,
                                           num_heads=args.num_head,
                                           num_layers=args.num_layer,
                                           ff_size=args.d_ff,
                                           activations=args.activations
                                           ).to(self.device).float()     
        trainable_count_body = sum(p.numel() for p in self.model_pose.parameters() if p.requires_grad)
        
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
        self.model_pose.apply(initialize_weights)
        self.optimizer_model_pose = eval(args.optimizer)(self.model_pose.parameters(), lr = args.lr)
        self.scheduler_pose = eval(args.scheduler)(self.optimizer_model_pose, step_size=args.stepsize, gamma=args.gamma)
        self.skel = InhouseStudioSkeleton()
        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        
        print(args.model, 'Model Created')
        if args.load:
            print('Loading Model', args.model)
            self.book._load_model(self.model_pose, 'model_pose')
        print('Loading the data')
        if is_train:
            self.load_data(args)
        else:
            self.load_data_testtime(args)
        
        
    def load_data_testtime(self, args):
        self.ds_data = LindyHopDataset(args, window_size=self.frames, split=self.testtime_split)
        self.load_ds_data = DataLoader(self.ds_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        
        
    def load_data(self, args):
        
        ds_train = LindyHopDataset(args, window_size=self.frames, split='train')
        self.ds_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        print('Train set loaded. Size=', len(self.ds_train.dataset))
        ds_val = LindyHopDataset(args, window_size=self.frames, split='test')
        self.ds_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        print('Validation set loaded. Size=', len(self.ds_val.dataset))
            
    def calc_kldiv(self, dist_m):
        mu_ref = torch.zeros_like(dist_m.loc)
        scale_ref = torch.ones_like(dist_m.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        return torch.distributions.kl_divergence(dist_m, dist_ref)
    
    def calc_loss(self, num_epoch):
        bs, seq, J, dim = self.generated.shape
        pos_loss = self.lambda_loss['pos'] * self.mse_criterion(self.gt_pose2, self.generated)
        vel_gt = self.gt_pose2[:, 1:] - self.gt_pose2[:, :-1]
        vel_gen = self.generated[:, 1:] - self.generated[:, :-1]
        velocity_loss = self.lambda_loss['vel'] * self.mse_criterion(vel_gt, vel_gen)
        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
        acc_gen = vel_gen[:, 1:] - vel_gen[:, :-1]
        acc_loss = self.lambda_loss['vel'] * self.mse_criterion(acc_gt, acc_gen)
        bone_len_gt = (self.gt_pose2[:, :, 1:] - self.gt_pose2[:, :, [self.skel.parents_body_only[x] for x in range(1, J)]]).norm(dim=-1)
        bone_len_gen = (self.generated[:, :, 1:] - self.generated[:, :, [self.skel.parents_body_only[x] for x in range(1, J)]]).norm(dim=-1)
        bone_len_consistency_loss = self.lambda_loss['bone'] * self.mse_criterion(bone_len_gt, bone_len_gen)
        if num_epoch > 100: 
            self.lambda_loss['foot'] = 20.0
        else:
            self.lambda_loss['foot'] = 0.0
        rightfoot_idx = [4, 5]
        leftfoot_idx = [9, 10]
        gen_leftfoot_joint = self.generated[:, :, leftfoot_idx] 
        static_left_foot_index = gen_leftfoot_joint[..., 1] <= 0.02
        gen_rightfoot_joint = self.generated[:, :, rightfoot_idx] 
        static_right_foot_index = gen_rightfoot_joint[..., 1] <= 0.02
        gen_leftfoot_vel = torch.zeros_like(gen_leftfoot_joint)
        gen_leftfoot_vel[:, :-1] = gen_leftfoot_joint[:,  1:] - gen_leftfoot_joint[:, :-1]
        gen_leftfoot_vel[~static_left_foot_index] = 0
        gen_rightfoot_vel = torch.zeros_like(gen_rightfoot_joint)
        gen_rightfoot_vel[:, :-1] = gen_rightfoot_joint[:,  1:] - gen_rightfoot_joint[:, :-1]
        gen_rightfoot_vel[~static_right_foot_index] = 0
        footskate_loss =  self.lambda_loss['foot'] * (self.mse_criterion(gen_leftfoot_vel, torch.zeros_like(gen_leftfoot_vel)) +  
                                                self.mse_criterion(gen_rightfoot_vel, torch.zeros_like(gen_rightfoot_vel)) )  
        
        loss_logs = [pos_loss, velocity_loss, bone_len_consistency_loss,
                     footskate_loss, acc_loss]
        
        #include the interaction loss
        self.lambda_loss['in'] = 50.0
        rh_rh = self.contact_map[:,:, 0] == 1
        rh_lh = self.contact_map[:,:, 1] == 1
        lh_rh = self.contact_map[:,:, 2] == 1
        lh_lh = self.contact_map[:,:, 3] == 1
       
        arm_interact_loss = self.lambda_loss['in'] * torch.mean(
            rh_lh * ((self.pose1[:, :, right_side] - self.gt_pose2[:, :, left_side]).norm(dim=-1) - (
                self.pose1[:, :, right_side] - self.generated[:, :, left_side]).norm(dim=-1)).norm(dim=-1) + rh_rh * (
                    (self.pose1[:, :, right_side] - self.gt_pose2[:, :, right_side]).norm(dim=-1) - (
                        self.pose1[:, :, right_side] - self.generated[:, :, right_side]).norm(dim=-1)).norm(dim=-1) + lh_rh * ((
                            self.pose1[:, :, left_side] - self.gt_pose2[:, :, right_side]).norm(dim=-1) - (
                                self.pose1[:, :, left_side] - self.generated[:, :, right_side]).norm(dim=-1)).norm(dim=-1) + lh_lh * ((
                                    self.pose1[:, :, left_side] - self.gt_pose2[:, :, left_side]).norm(dim=-1) - (
                                        self.pose1[:, :, left_side] - self.generated[:, :, left_side]).norm(dim=-1)).norm(dim=-1) )
        
        loss_logs.append(arm_interact_loss)
        interact_loss = self.mse_criterion((self.pose1 - self.gt_pose2), (self.pose1 - self.generated))
        loss_logs.append(interact_loss)
        return loss_logs
    
    def forward(self, motions1, motions2, t=None):
        B, T = motions2.shape[:2]
        if t == None:
            t, _ = self.sampler.sample(B, motions1.device)
        self.diffusion_timestep = t
        output = self.diffusion.training_losses(
            model=self.model_pose,
            x_start=motions2,
            t=t,
            model_kwargs={"motion1": motions1}
        )
        
        self.pose1 = motions1
        self.gt_pose2 = motions2      #gt pose 2
        self.generated = output['pred'] # synthesized pose 2
        return t, output['x_noisy']
    

    
    def root_relative_normalization(self, global_pose1, global_pose2):
        
        global_pose1 = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
        pose1_root_rel = global_pose1 - torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_jts, axis=-2)
        self.pose1_root_rel = pose1_root_rel / self.scale
        global_pose2 = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
        
        pose2_root_rel = global_pose2 - torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_jts, axis=-2)
        self.pose2_root_rel = pose2_root_rel / self.scale
        tmp=1
    
    def root_relative_unnormalization(self, pose1_normalized, pose2_normalized):
        pose1_unnormalized = pose1_normalized * self.scale
        pose2_unnormalized = pose2_normalized * self.scale
        global_pose1 = pose1_unnormalized + torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_jts, axis=-2)
        global_pose2 = pose2_unnormalized + torch.repeat_interleave(self.global_root_origin.unsqueeze(-2), self.num_jts, axis=-2)
        return global_pose1, global_pose2
        
    def train(self, num_epoch, ablation=None):
        total_train_loss = 0.0
        total_pos_loss = 0.0
        total_vel_loss = 0.0
        total_bone_loss = 0.0
        total_footskate_loss = 0.0
        self.model_pose.train()
        training_tqdm = tqdm(self.ds_train, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        diff_count = [0, 5, 10, 50, 100, 200, 300, 400, 499]
        for count, batch in enumerate(training_tqdm):
            self.optimizer_model_pose.zero_grad()
            
            with torch.autograd.detect_anomaly():
                global_pose1 = batch['pose_canon_1'].to(self.device).float()
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                self.contact_map = batch['contacts'].to(self.device).float()
                self.global_root_origin = batch['global_root_origin'].to(device).float()
                if global_pose1.shape[1] == 0:
                    continue
                self.root_relative_normalization(global_pose1, global_pose2)
                t, noisy = self.forward(self.pose1_root_rel, self.pose2_root_rel)
                
                loss_logs = self.calc_loss(num_epoch)
                loss_model = sum(loss_logs)
                total_train_loss += loss_model.item()
                total_pos_loss += loss_logs[0].item()
                total_vel_loss += loss_logs[1].item()
                total_bone_loss += loss_logs[2].item()
                total_footskate_loss += loss_logs[3].item()
                             
                if loss_model == float('inf') or torch.isnan(loss_model):
                    print('Train loss is nan')
                    exit()
                loss_model.backward()
                torch.nn.utils.clip_grad_value_(self.model_pose.parameters(), 0.01)
                self.optimizer_model_pose.step()
                
            
        avg_train_loss = total_train_loss/(count + 1)
        avg_pos_loss = total_pos_loss/(count + 1)
        avg_vel_loss = total_vel_loss/(count + 1)
        avg_bone_loss = total_bone_loss/(count + 1)
        avg_footskate_loss = total_footskate_loss/(count + 1)
        
        return avg_train_loss, (avg_pos_loss, avg_vel_loss, avg_bone_loss, avg_footskate_loss)

    def evaluate(self, num_epoch, ablation=None):
        total_eval_loss = 0.0
        total_pos_loss = 0.0
        total_vel_loss = 0.0
        total_bone_loss = 0.0
        total_footskate_loss = 0.0
        self.model_pose.eval()
        T = self.frames
        eval_tqdm = tqdm(self.ds_val, desc='eval' + ' {:.10f}'.format(0), leave=False, ncols=120)

        for count, batch in enumerate(eval_tqdm):
            if True:
                global_pose1 = batch['pose_canon_1'].to(self.device).float()
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                self.contact_map = batch['contacts'].to(self.device).float()
                
                self.global_root_origin = batch['global_root_origin'].to(device).float()
                if global_pose1.shape[1] == 0:
                    continue
                self.root_relative_normalization(global_pose1, global_pose2)
                t, noisy = self.forward(self.pose1_root_rel, self.pose2_root_rel)
                loss_logs = self.calc_loss(num_epoch)
                loss_model = sum(loss_logs)
                total_eval_loss += loss_model.item()
                total_pos_loss += loss_logs[0].item()
                total_vel_loss += loss_logs[1].item()
                total_bone_loss += loss_logs[2].item()
                total_footskate_loss += loss_logs[3].item()
                                  
        avg_eval_loss = total_eval_loss/(count + 1)
        avg_pos_loss = total_pos_loss/(count + 1)
        avg_vel_loss = total_vel_loss/(count + 1)
        avg_bone_loss = total_bone_loss/(count + 1)
        avg_footskate_loss = total_footskate_loss/(count + 1)
        
        return avg_eval_loss, (avg_pos_loss, avg_vel_loss, avg_bone_loss, avg_footskate_loss)
    
    def fit(self, n_epochs=None, ablation=False):
        print('*****Inside Trainer.fit *****')
        if n_epochs is None:
            n_epochs = self.args.num_epochs
        starttime = datetime.now().replace(microsecond=0)
        print('Started Training at', datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), 'Total epochs: ', n_epochs)
        save_model_dict = {}
        best_eval = 1000

        train_pos_loss = []
        train_vel_loss = []
        train_bone_loss = []
        train_footskate_loss = []
        eval_pos_loss = []
        eval_vel_loss = []
        eval_bone_loss = []
        eval_footskate_loss = []
        for epoch_num in range(self.epochs_completed, n_epochs + 1):
            tqdm.write('--- starting Epoch # %03d' % epoch_num)
            train_loss, (train_pos_loss_, train_vel_loss_, train_bone_loss_,
                         train_footskate_loss_) = self.train(epoch_num, ablation)
            train_pos_loss.append(train_pos_loss_)
            train_vel_loss.append(train_vel_loss_)
            train_bone_loss.append(train_bone_loss_)
            train_footskate_loss.append(train_footskate_loss_)
            if epoch_num % 5 == 0:
                eval_loss,  (eval_pos_loss_, eval_vel_loss_, eval_bone_loss_,
                            eval_footskate_loss_) = self.evaluate(epoch_num, ablation)
                eval_pos_loss.append(eval_pos_loss_)
                eval_vel_loss.append(eval_vel_loss_)
                eval_bone_loss.append(eval_bone_loss_)
                eval_footskate_loss.append(eval_footskate_loss_)
            else:
                eval_loss = 0.0
            self.scheduler_pose.step()
            self.book.update_res({'epoch': epoch_num, 'train': train_loss, 'val': eval_loss, 'test': 0.0})
            self.book._save_res()
            self.book.print_res(epoch_num, key_order=['train', 'val', 'test'], lr=self.optimizer_model_pose.param_groups[0]['lr'])
            
            if epoch_num > 100 and eval_loss < best_eval:
                print('Best eval at epoch {}'.format(epoch_num))
                f = open(os.path.join(self.args.save_dir, self.book.name.name, self.book.name.name + 'best.p'), 'wb') 
                save_model_dict.update({'model_pose': self.model_pose.state_dict()})
                torch.save(save_model_dict, f)
                f.close()   
                best_eval = eval_loss
            if epoch_num > 20 and epoch_num % 20 == 0 :
                f = open(os.path.join(self.args.save_dir, self.book.name.name, self.book.name.name + '{:06d}'.format(epoch_num) + '.p'), 'wb') 
                save_model_dict.update({'model_pose': self.model_pose.state_dict()})
                torch.save(save_model_dict, f)
                f.close()   
        endtime = datetime.now().replace(microsecond=0)
        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training complete in %s!\n' % (endtime - starttime))

        
                       
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
    is_train = True
    ablation = None       # if True then ablation: no_IAC_loss
    model_trainer = Trainer(args=args, is_train=is_train, split='test', JT_POSITION=True, num_jts=27)
    print("** Method Initialization Complete **")
    model_trainer.fit(ablation=ablation)
     
