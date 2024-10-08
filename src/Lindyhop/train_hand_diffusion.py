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
from src.Lindyhop.models.MotionDiffusion_hand import *
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
                                           num_frames=self.frames,
                                           num_jts=self.num_jts,
                                           input_feats=args.hand_out_feats,
                                           latent_dim=args.d_modelhand,
                                           num_heads=args.num_head_hands,
                                           num_layers=args.num_layer_hands,
                                           ff_size=args.d_ffhand,
                                           activations=args.activations
                                           ).to(self.device).float()     
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
        # self.scheduler_pose = eval(args.scheduler)(self.optimizer_model_pose, factor=args.factor, patience=args.patience, threshold= args.threshold,  min_lr = 2e-7)
        self.skel = InhouseStudioSkeleton()
        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.BCE_criterion = torch.nn.BCELoss()
        print(args.model, 'Model Created')
        if args.load:
            print('Loading Model', args.model)
            self.book._load_model(self.model_pose, 'model_pose')
        print('Loading the data')
        if is_train:
            self.load_data(args)
        else:
            self.load_data_testtime(args)
        self.mean_var_norm = torch.load(args.mean_var_norm)
        
    def load_data_testtime(self, args):
        self.ds_data = LindyHopDataset(args, window_size=self.frames, split=self.testtime_split)
        # self.ds_data = DataLoader(ds_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        # print('Data split loaded. Size=', len(self.ds_data.dataset))
        
    def load_data(self, args):
        # ds_train = LindyHandsDataset(args, window_size=self.frames, split='train')
        ds_train = LindyHopDataset(args, window_size=self.frames, split='train_full')
        self.ds_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        print('Train set loaded. Size=', len(self.ds_train.dataset))
        ds_val = LindyHopDataset(args, window_size=self.frames, split='test')
        self.ds_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        print('Validation set loaded. Size=', len(self.ds_val.dataset))
            
 
    def calc_loss(self, num_epoch):
        bs, seq, dim = self.generated.shape
        pos_loss = self.lambda_loss['pos'] * self.mse_criterion(self.input, self.generated)
        vel_gt = self.input[:, 1:] - self.input[:, :-1]
        vel_gen = self.generated[:, 1:] - self.generated[:, :-1]
        velocity_loss = self.lambda_loss['vel'] * self.mse_criterion(vel_gt, vel_gen)
        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
        acc_gen = vel_gen[:, 1:] - vel_gen[:, :-1]
        acc_loss = self.lambda_loss['vel'] * self.mse_criterion(acc_gt, acc_gen)
        gt_pose = self.input.reshape(bs, seq, -1, 3)
        gen_pose = self.generated.reshape(bs, seq, -1, 3)
        p1_rhand = self.p1_rhand_pos
        p2gt_rhand = self.p2_rhand_pos
        p2gen_rhand = gen_pose[:, :, :11]
        p1_lhand = self.p1_lhand_pos
        p2gt_lhand = self.p2_lhand_pos
        p2gen_lhand = gen_pose[:, :, 11:]
        
        bone_len_gt = (gt_pose[:, :, 1:] - gt_pose[:, :, [self.skel.parent_fingers[x] for x in range(1, 2*self.num_jts)]]).norm(dim=-1)
        bone_len_gen = (gen_pose[:, :, 1:] - gen_pose[:, :, [self.skel.parent_fingers[x] for x in range(1, 2*self.num_jts)]]).norm(dim=-1)
        bone_len_consistency_loss = self.lambda_loss['bone'] * self.mse_criterion(bone_len_gt, bone_len_gen)
        
        loss_logs = [pos_loss, velocity_loss, acc_loss, bone_len_consistency_loss]
        
        # #include the interaction loss
        
        self.lambda_loss['in'] = 10.0
        rh_rh = self.contact_map[:,:, 0] == 1
        rh_lh = self.contact_map[:,:, 1] == 1
        lh_rh = self.contact_map[:,:, 2] == 1
        lh_lh = self.contact_map[:,:, 3] == 1
       
        interact_loss = self.lambda_loss['in'] * torch.mean( rh_lh * ((p1_rhand - p2gt_lhand).norm(dim=-1) - (p1_rhand - p2gen_lhand).norm(dim=-1)).norm(dim=-1) + 
                                                rh_rh * ((p1_rhand - p2gt_rhand).norm(dim=-1) - (p1_rhand - p2gen_rhand).norm(dim=-1)).norm(dim=-1) + 
                                                lh_rh * ((p1_lhand - p2gt_rhand).norm(dim=-1) - (p1_lhand - p2gen_rhand).norm(dim=-1)).norm(dim=-1) + 
                                                lh_lh * ((p1_lhand - p2gt_lhand).norm(dim=-1) - (p1_lhand - p2gen_lhand).norm(dim=-1)).norm(dim=-1) )
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
        
        self.generated = output['pred'] # synthesized pose 2
        return t, output['x_noisy']
    
    def generate(self, motion1, motion2=None):
        B, T, J, dim_pose = motion1.shape
        output = self.diffusion.p_sample_loop(
            self.model_pose,
            (B, T, J, dim_pose),
            clip_denoised=False,
            progress=True,
            pre_seq= motion2,
            model_kwargs={
                'motion1': motion1,
            })
        return output

         
    def relative_normalization(self, global_pose1, global_pose2, global_rot1, global_rot2):
        self.p1_rhand_wrist_pos = global_pose1[:, :, 18]
        self.p1_lhand_wrist_pos = global_pose1[:, :, 43]
        p1_rhand_wrist_pos = (global_pose1[:, :, 18] - self.p1_rhand_wrist_pos) / self.scale
        p1_lhand_wrist_pos = (global_pose1[:, :, 43] - self.p1_lhand_wrist_pos) / self.scale
        p2_rhand_wrist_pos = (global_pose2[:, :, 18] - self.p1_rhand_wrist_pos) / self.scale
        p2_lhand_wrist_pos = (global_pose2[:, :, 43] - self.p1_lhand_wrist_pos) / self.scale
        B = p1_rhand_wrist_pos.shape[0]
        T = p1_rhand_wrist_pos.shape[1]
        p1_rhand_rot = self.skel.select_bvh_joints(global_rot1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only).reshape(B, T, -1)
        p1_lhand_rot = self.skel.select_bvh_joints(global_rot1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only).reshape(B, T, -1)
        
        p2_rhand_rot = self.skel.select_bvh_joints(global_rot2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only).reshape(B, T, -1)
        p2_lhand_rot = self.skel.select_bvh_joints(global_rot2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only).reshape(B, T, -1)
        
        # create a contact map based on threshold of wrists
        self.contact_dist = torch.zeros(B, T, 4).to(p1_lhand_rot.device).float()
        self.contact_dist[:,:, 0] = ((p1_rhand_wrist_pos - p2_rhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 1] = ((p1_rhand_wrist_pos - p2_lhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 2] = ((p1_lhand_wrist_pos - p2_rhand_wrist_pos)**2).norm(dim=-1)
        self.contact_dist[:,:, 3] = ((p1_lhand_wrist_pos - p2_lhand_wrist_pos)**2).norm(dim=-1)
        
        self.input_condn = torch.cat((p1_rhand_wrist_pos, p1_lhand_wrist_pos,
                                      p2_rhand_wrist_pos, p2_lhand_wrist_pos,
                                      p1_rhand_rot, p1_lhand_rot, self.contact_dist), dim=-1)
        self.input = torch.cat((p2_rhand_rot, p2_lhand_rot), dim=-1)
        
    def pose_relative_normalization(self, global_pose1, global_pose2, contact_maps):
        p1_rhand_pos = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only)
        p1_lhand_pos = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only)
        
        p2_rhand_pos = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.rh_fingers_only)
        p2_lhand_pos = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.lh_fingers_only)
        self.p1_rhand_wrist_pos = p1_rhand_pos[:, :, 0]
        self.p2_rhand_wrist_pos = p2_rhand_pos[:, :, 0]
        self.p1_lhand_wrist_pos = p1_lhand_pos[:, :, 0]
        self.p2_lhand_wrist_pos = p2_lhand_pos[:, :, 0]
        self.p1_rhand_pos = (p1_rhand_pos - torch.repeat_interleave(self.p1_rhand_wrist_pos.unsqueeze(-2), self.num_jts, axis=-2))/self.scale
        self.p1_lhand_pos = (p1_lhand_pos - torch.repeat_interleave(self.p1_lhand_wrist_pos.unsqueeze(-2), self.num_jts, axis=-2))/self.scale
        self.p2_rhand_pos = (p2_rhand_pos - torch.repeat_interleave(self.p2_rhand_wrist_pos.unsqueeze(-2), self.num_jts, axis=-2))/self.scale
        self.p2_lhand_pos = (p2_lhand_pos - torch.repeat_interleave(self.p2_lhand_wrist_pos.unsqueeze(-2), self.num_jts, axis=-2))/self.scale
        B = self.p1_rhand_wrist_pos.shape[0]
        T = self.p1_rhand_wrist_pos.shape[1]
        
        # # create a contact map based on threshold of wrists  
        # # contact maps 0: rh-rh, 1: lh-lh, 2: lh-rh , 3: rh-lh
        # self.contact_dist = torch.zeros(B, T, 4).to(p1_lhand_pos.device).float()
        # self.contact_dist[:, :, 0] = ((self.p1_rhand_wrist_pos - self.p2_rhand_wrist_pos)).norm(dim=-1)/self.scale
        # self.contact_dist[:, :, 1] = ((self.p1_lhand_wrist_pos - self.p2_lhand_wrist_pos)).norm(dim=-1)/self.scale
        # self.contact_dist[:, :, 2] = ((self.p1_lhand_wrist_pos - self.p2_rhand_wrist_pos)).norm(dim=-1)/self.scale
        # self.contact_dist[:, :, 3] = ((self.p1_rhand_wrist_pos - self.p2_lhand_wrist_pos)).norm(dim=-1)/self.scale
        
        self.contact_map = contact_maps.to(self.device).float()
        self.input_condn = torch.cat((self.p1_rhand_pos.reshape(B, T, -1), self.p1_lhand_pos.reshape(B, T, -1),
                                      self.contact_map), dim=-1)
        self.input = torch.cat((self.p2_rhand_pos.reshape(B, T, -1), self.p2_lhand_pos.reshape(B, T, -1)), dim=-1)
    
        
    def train(self, num_epoch, ablation=None):
        total_train_loss = 0.0
        self.model_pose.train()
        training_tqdm = tqdm(self.ds_train, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        # self.joint_parent = self.ds_train.dataset.bvh_joint_parents_list
        diff_count = [0, 5, 10, 50, 100, 200, 300, 400, 499]
        for count, batch in enumerate(training_tqdm):
            self.optimizer_model_pose.zero_grad()
            
            # with torch.autograd.detect_anomaly():
            if True:
                global_pose1 = batch['pose_canon_1'].to(self.device).float()
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                # global_rot1 = rotmat2d6_tensor(batch['rotmat_1']).to(self.device).float()
                # global_rot2 = rotmat2d6_tensor(batch['rotmat_2']).to(self.device).float()
                if global_pose1.shape[1] == 0:
                    continue
                # self.relative_normalization(global_pose1, global_pose2, global_rot1, global_rot2)
                self.pose_relative_normalization(global_pose1, global_pose2, batch['contacts'])
                t, noisy = self.forward(self.input_condn, self.input)
                # #control timesteps
                # timesteps = torch.LongTensor([diff_count[count]]).to(self.device)
                # timesteps, x_start_noisy = self.forward(self.pose1_root_rel, self.pose2_root_rel, timesteps)
                # save_folder = 'diffusion_visuals/mean_std_' + self.beta_scheduler + '_' + str(self.diffusion_steps)+'_/'+ str(timesteps.item())
                # unnormalized_xstart_noisy = x_start_noisy[0] * self.mean_var_norm['p1_body_std'] + self.mean_var_norm['p1_body_mean'] 
                # unnormalized_p1 = self.pose1_root_rel[0] * self.mean_var_norm['p1_body_std'] + self.mean_var_norm['p1_body_mean'] 
                # unnormalized_p2 = self.pose2_root_rel[0] * self.mean_var_norm['p2_body_std'] + self.mean_var_norm['p2_body_mean'] 
                
                # plot_contacts3D(unnormalized_p1.detach().cpu(), unnormalized_xstart_noisy.detach().cpu(), 
                #                     savepath=save_folder, kinematic_chain='no_fingers', onlyone=False, gif=True)
                # torch.cuda.empty_cache()
                # if count >= 8:
                #     break
                # else:
                #     continue
                loss_logs = self.calc_loss(num_epoch)
                loss_model = sum(loss_logs)
                total_train_loss += loss_model.item()
                             
                if loss_model == float('inf') or torch.isnan(loss_model):
                    print('Train loss is nan')
                    exit()
                loss_model.backward()
                # torch.nn.utils.clip_grad_norm_(self.model_pose.parameters(), 1)
                torch.nn.utils.clip_grad_value_(self.model_pose.parameters(), 0.01)
                self.optimizer_model_pose.step()
                # if torch.sum(torch.isnan(self.model_pose.motion1_pre_joint_proj.weight)):
                #     exit()
            
        avg_train_loss = total_train_loss/(count + 1)
        return avg_train_loss

    def evaluate(self, num_epoch, ablation=None):
        total_eval_loss = 0.0
        self.model_pose.eval()
        T = self.frames
        eval_tqdm = tqdm(self.ds_val, desc='eval' + ' {:.10f}'.format(0), leave=False, ncols=120)

        for count, batch in enumerate(eval_tqdm):
            if True:
                global_pose1 = batch['pose_canon_1'].to(self.device).float()
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                # global_rot1 = rotmat2d6_tensor(batch['rotmat_1']).to(self.device).float()
                # global_rot2 = rotmat2d6_tensor(batch['rotmat_2']).to(self.device).float()
                if global_pose1.shape[1] == 0:
                    continue
                # self.relative_normalization(global_pose1, global_pose2, global_rot1, global_rot2)
                self.pose_relative_normalization(global_pose1, global_pose2, batch['contacts'])
                t, noisy = self.forward(self.input_condn, self.input)
                loss_logs = self.calc_loss(num_epoch)
                loss_model = sum(loss_logs)
                total_eval_loss += loss_model.item()
               
                                  
        avg_eval_loss = total_eval_loss/(count + 1)
       
        return avg_eval_loss
    
    def fit(self, n_epochs=None, ablation=False):
        print('*****Inside Trainer.fit *****')
        if n_epochs is None:
            n_epochs = self.args.num_epochs
        starttime = datetime.now().replace(microsecond=0)
        print('Started Training at', datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), 'Total epochs: ', n_epochs)
        save_model_dict = {}
        best_eval = 1000

        for epoch_num in range(self.epochs_completed, n_epochs + 1):
            tqdm.write('--- starting Epoch # %03d' % epoch_num)
            train_loss = self.train(epoch_num, ablation)
            if epoch_num % 5 == 0:
                eval_loss = self.evaluate(epoch_num, ablation)
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
                # pos_loss_plot = makepath(os.path.join(self.args.save_dir, self.book.name.name, 'loss_plot', self.book.name.name + '{:06d}'.format(epoch_num), 'pos_loss.png'), isfile=True)
                # vel_loss_plot = makepath(os.path.join(self.args.save_dir, self.book.name.name, 'loss_plot', self.book.name.name + '{:06d}'.format(epoch_num), 'vel_loss.png'), isfile=True)
                # bone_loss_plot = makepath(os.path.join(self.args.save_dir, self.book.name.name, 'loss_plot', self.book.name.name + '{:06d}'.format(epoch_num), 'bone_loss.png'), isfile=True)
                # foot_loss_plot = makepath(os.path.join(self.args.save_dir, self.book.name.name, 'loss_plot', self.book.name.name + '{:06d}'.format(epoch_num), 'foot_loss.png'), isfile=True)
                
                # fig = plt.figure()
                # # plt.plot(train_pos_loss, marker='o', c='b')
                # plt.plot(eval_pos_loss, marker='o', c='r')
                # fig.savefig(pos_loss_plot)
                # fig = plt.figure()
                # # plt.plot(train_vel_loss, marker='o', c='b')
                # plt.plot(eval_vel_loss, marker='o', c='r')
                # fig.savefig(vel_loss_plot)
                # fig = plt.figure()
                # # plt.plot(train_bone_loss, marker='o', c='b')
                # plt.plot(eval_bone_loss, marker='o', c='r')
                # fig.savefig(bone_loss_plot)
                # fig = plt.figure()
                # # plt.plot(train_footskate_loss, marker='o', c='b')
                # plt.plot(eval_footskate_loss, marker='o', c='r')
                # fig.savefig(foot_loss_plot)
                # fig = plt.figure()
                # plt.close()
                f = open(os.path.join(self.args.save_dir, self.book.name.name, self.book.name.name + '{:06d}'.format(epoch_num) + '.p'), 'wb') 
                save_model_dict.update({'model_pose': self.model_pose.state_dict()})
                torch.save(save_model_dict, f)
                f.close()   
        endtime = datetime.now().replace(microsecond=0)
        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training complete in %s!\n' % (endtime - starttime))

    def test(self, plotfiles=True):
        self.model_pose.eval()
        T = self.frames
        total_frames = 40
        annot_dict = self.ds_data.annot_dict
        pose_canon_1 = torch.tensor(annot_dict['pose_canon_1'][0]).to(self.device).float()
        rotmat_1 = torch.tensor(annot_dict['rotmat_1'][0]).to(self.device).float()
        rotmat_2 = torch.tensor(annot_dict['rotmat_2'][0]).to(self.device).float()
        pose_canon_2 = torch.tensor(annot_dict['pose_canon_2'][0]).to(self.device).float()
        _, J, dim = pose_canon_1.shape
        global_action_pose = []
        global_reaction_pose = []
        global_gt_reaction_pose = []
        reaction_handrot_out = None
        for count in range(0, total_frames, T):
            global_rot1 = rotmat2d6_tensor(rotmat_1[count:count+T])
            global_rot2 = rotmat2d6_tensor(rotmat_2[count:count+T])
            global_pose1 = pose_canon_1[count:count+T].reshape(1, T, J, dim)
            global_pose2 = pose_canon_2[count:count+T].reshape(1, T, J, dim)
               
            self.relative_normalization(global_pose1, global_pose2, global_rot1, global_rot2)
            reaction_handrot_out = self.generate(self.input_condn, motion2=reaction_handrot_out)
            action_pose, reaction_hand_rot = self.root_relative_unnormalization(self.pose1_root_rel, reaction_handrot_out)
            global_reaction_pose.append(reaction_hand_rot)
            global_action_pose.append( action_pose)
            global_gt_reaction_pose.append( self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only))
            
            
        savefolder = makepath(os.path.join(args.load[:-2], self.testtime_split, str(annot_dict['seq'][0])+'_'+str(count)), isfile=True)
        global_action_pose = torch.cat(global_action_pose, dim=1)
        global_reaction_pose = torch.cat(global_reaction_pose, dim=1)
        global_gt_reaction_pose = torch.cat(global_gt_reaction_pose, dim=1)
        plot_contacts3D(pose1=(global_action_pose[0].detach().cpu().numpy()),
                            pose2=(global_reaction_pose[0].detach().cpu().numpy()),
                            gt_pose2=(global_gt_reaction_pose[0].detach().cpu().numpy()),
                            savepath=savefolder, kinematic_chain = 'no_fingers', onlyone=False)
            
            
if __name__ == '__main__':
    args = argparseNloop()
    args.lambda_loss = {
        'fk': 1.0,
        'fk_vel': 1.0,
        'rot': 1e+3,
        'rot_vel': 1e+1,
        'kldiv': 1.0,
        'pos': 1e+3,
        'vel': 1e+1,
        'bone': 1.0,
        'foot': 0.0,
        
    }
    # args.load = os.path.join('save', 'Lindyhop', 'diffusionhand', 'exp_12_model_DiffusionTransformer_batchsize_128_frames_25_',
    #                         'exp_12_model_DiffusionTransformer_batchsize_128_frames_25_001000.p' )
    is_train = True
    ablation = None       # if True then ablation: no_IAC_loss
    model_trainer = Trainer(args=args, is_train=is_train, split='train', JT_POSITION=True, num_jts=11)
    print("** Method Inititalization Complete **")
    if is_train:
        model_trainer.fit(ablation=ablation)
    else:
        assert os.path.exists(args.load)
        model_trainer.test()    