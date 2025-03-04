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
from src.Lindyhop.models.transAE import *

from src.Lindyhop.skeleton import *
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
        self.testtime_split = split
        self.num_jts = num_jts
        self.model_pose = VanillaTransformer(args).to(self.device).float()
        trainable_count_body = sum(p.numel() for p in self.model_pose.parameters() if p.requires_grad)
        self.model_pose.apply(initialize_weights)
        self.optimizer_model_pose = eval(args.optimizer)(self.model_pose.parameters(), lr = args.lr)
        self.scheduler_pose = eval(args.scheduler)(self.optimizer_model_pose, step_size=args.stepsize, gamma=args.gamma)
        self.skel = InhouseStudioSkeleton()
        
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
            
        
    def train(self, num_epoch, ablation=None):
        total_train_loss = 0.0
        self.model_pose.train()
        training_tqdm = tqdm(self.ds_train, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for count, batch in enumerate(training_tqdm):
            self.optimizer_model_pose.zero_grad()
            with torch.autograd.detect_anomaly():
                global_pose1 = batch['pose_canon_1'].to(self.device).float()
                global_pose1 = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                           new_joint_order=self.skel.body_only)
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                global_pose2 = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                           new_joint_order=self.skel.body_only)
                
                _, loss_model = self.model_pose(global_pose1, global_pose2)           
                total_train_loss += loss_model.item()
                
                if loss_model == float('inf') or torch.isnan(loss_model):
                    print('Train loss is nan')
                    exit()
                loss_model.backward()
                torch.nn.utils.clip_grad_value_(self.model_pose.parameters(), 0.01)
                self.optimizer_model_pose.step()
                       
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
                global_pose1 = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                           new_joint_order=self.skel.body_only)
                global_pose2 = batch['pose_canon_2'].to(self.device).float()
                global_pose2 = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                           new_joint_order=self.skel.body_only)
                
                _, loss_model = self.model_pose(global_pose1, global_pose2)   
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
                f = open(os.path.join(self.args.save_dir, self.book.name.name, self.book.name.name + '{:06d}'.format(epoch_num) + '.p'), 'wb') 
                save_model_dict.update({'model_pose': self.model_pose.state_dict()})
                torch.save(save_model_dict, f)
                f.close()   
        endtime = datetime.now().replace(microsecond=0)
        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training complete in %s!\n' % (endtime - starttime))

        
                       
if __name__ == '__main__':
    args = argparseNloop()
    
    is_train = True
    ablation = None       # if True then ablation: no_IAC_loss
    model_trainer = Trainer(args=args, is_train=is_train, split='test', JT_POSITION=True, num_jts=27)
    print("** Method Initialization Complete **")
    model_trainer.fit(ablation=ablation)
     
