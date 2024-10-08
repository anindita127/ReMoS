import numpy as np
import os 
import pickle
import pytorch3d.transforms as t3d
import random
import sys
sys.path.append('.')
sys.path.append('..')
import torch
from math import radians, cos, sin
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.Lindyhop.skeleton import InhouseStudioSkeleton
from src.Lindyhop.visualizer import plot_contacts3D
from src.tools.transformations import *
from src.tools.utils import makepath
from src.Lindyhop.argUtils import argparseNloop


class LindyHopDataset(torch.utils.data.Dataset):
    def __init__(self, args, window_size=10, split='val'):
        self.root = args.data_dir
        self.scale = args.scale
        self.split = split
        self.window_size = int(window_size)
        with open(os.path.join(self.root, self.split+'.pkl'), 'rb') as f:
            self.annot_dict = pickle.load(f)
        self.output_keys = ['seq', 'pose_canon_1', 'pose_canon_2', 
                             'contacts', 'dofs_1', 'dofs_2',
                             'rotmat_1', 'rotmat_2',
                             'offsets_1', 'offsets_2',
                        ]
        self.skel = InhouseStudioSkeleton()

            
    def __getitem__(self, ind):
        index = ind % len(self.annot_dict['pose_canon_1'])
        annot = {}
        for key in self.output_keys:
            annot[key] = self.annot_dict[key][index]
        skip = 1   
        start = np.random.randint(0, len(annot['pose_canon_1']) - self.window_size)
        end = start + self.window_size
        
        annot['contacts'] = annot['contacts'][start:end]           #  0.rh-rh, 1: lh-lh, 2: lh-rh , 3: rh-lh)
        annot['pose_canon_1'] = annot['pose_canon_1'][start: end: skip]
        annot['pose_canon_2'] = annot['pose_canon_2'][start: end: skip]
        annot['dofs_1'] = np.pi * (annot['dofs_1'][start:end: skip]) / 180.
        annot['dofs_2'] = np.pi * (annot['dofs_2'][start:end: skip]) / 180.
        annot['rotmat_1'] = annot['rotmat_1'][start:end: skip]
        annot['rotmat_2'] = annot['rotmat_2'][start:end: skip]
        annot['global_root_rotation'] = np.linalg.inv(annot['rotmat_1'][:, 0])
        annot['global_root_origin'] = annot['pose_canon_1'][:, 0]
        annot['p1_parent_rel'] = annot['pose_canon_1'][ :, 1:] - annot['pose_canon_1'][:, [self.skel.parents_full[x] for x in range(1, 69)]]
        annot['p2_parent_rel'] = annot['pose_canon_2'][:, 1:] - annot['pose_canon_2'][:, [self.skel.parents_full[x] for x in range(1, 69)]]
        return annot
    
    def __len__(self):
        return len(self.annot_dict['pose_canon_1'])
    
    def root_relative_normalization(self, global_pose1, global_pose2, global_root_rotation, global_root_origin):
        
        global_pose1 = self.skel.select_bvh_joints(global_pose1, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
        num_jts = global_pose1.shape[-2]
        pose1_root_rel = global_pose1 - torch.repeat_interleave(global_root_origin.unsqueeze(-2), num_jts, axis=-2)
        pose1_root_rel = torch.matmul(pose1_root_rel, global_root_rotation)
        pose1_root_rel = (pose1_root_rel - self.mean_var_norm['p1_body_mean']) / self.mean_var_norm['p1_body_std']
        # pose1_root_rel = (global_pose1 - self.mean_var_norm['p1_body_mean']) / self.mean_var_norm['p1_body_std']
        # self.pose1_root_rel = pose1_root_rel / self.scale
        global_pose2 = self.skel.select_bvh_joints(global_pose2, original_joint_order=self.skel.bvh_joint_order,
                                                             new_joint_order=self.skel.body_only)
        
        pose2_root_rel = global_pose2 - torch.repeat_interleave(global_root_origin.unsqueeze(-2), num_jts, axis=-2)
        pose2_root_rel = torch.matmul(pose2_root_rel, global_root_rotation)
        pose2_root_rel = (pose2_root_rel - self.mean_var_norm['p2_body_mean']) / self.mean_var_norm['p2_body_std']
        # pose2_root_rel = (global_pose2 - self.mean_var_norm['p2_body_mean']) / self.mean_var_norm['p2_body_std']
        # self.pose2_root_rel = pose2_root_rel / self.scale
        return pose1_root_rel, pose2_root_rel
    
    def root_relative_unnormalization(self, pose1_normalized, pose2_normalized, global_root_rotation, global_root_origin):
        num_jts = pose1_normalized.shape[-2]
        pose1_unnormalized = ((pose1_normalized * self.mean_var_norm['p1_body_std']) + self.mean_var_norm['p1_body_mean']).squeeze(0)
        pose2_unnormalized = ((pose2_normalized * self.mean_var_norm['p2_body_std']) + self.mean_var_norm['p2_body_mean']).squeeze(0) 
        # pose1_unnormalized = torch.matmul(pose1_unnormalized,  torch.linalg.inv(global_root_rotation))
        # pose2_unnormalized = torch.matmul(pose2_unnormalized,  torch.linalg.inv(global_root_rotation))
        global_pose1 = pose1_unnormalized + torch.repeat_interleave(global_root_origin.unsqueeze(-2), num_jts, axis=-2)
        global_pose2 = pose2_unnormalized + torch.repeat_interleave(global_root_origin.unsqueeze(-2), num_jts, axis=-2)
        return global_pose1, global_pose2
        # return pose1_unnormalized, pose2_unnormalized

if __name__ == "__main__":
    args = argparseNloop()
    use_pytorch = 1
    split = 'train_full'
    batch_size = 1
    window_size = 20

    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    dataset = LindyHopDataset(args, window_size = window_size, split=split)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True
                            )
    
    count = 0
    scale = args.scale
    
    bvh = 1
    
    # skel_ik = Studio_skeleton.StudioDataSkeleton()
    skel_fk = InhouseStudioSkeleton()

    for j, annot in enumerate(dataloader):
        global_pose1 = annot['pose_canon_1'].to(device).float()
        global_pose2 = annot['pose_canon_2'].to(device).float()
        # global_rot1 = rotmat2d6_tensor(annot['rotmat_1']).to(device).float()
        # global_rot2 = rotmat2d6_tensor(annot['rotmat_2']).to(device).float()
        # p1_rhand_rot = dataset.skel.select_bvh_joints(global_rot1, original_joint_order=dataset.skel.bvh_joint_order,
        #                                                      new_joint_order=dataset.skel.rh_fingers_only)
        # p1_lhand_rot = dataset.skel.select_bvh_joints(global_rot1, original_joint_order=dataset.skel.bvh_joint_order,
        #                                                      new_joint_order=dataset.skel.lh_fingers_only)
        
        # p2_rhand_rot = dataset.skel.select_bvh_joints(global_rot2, original_joint_order=dataset.skel.bvh_joint_order,
        #                                                      new_joint_order=dataset.skel.rh_fingers_only)
        # p2_lhand_rot = dataset.skel.select_bvh_joints(global_rot2, original_joint_order=dataset.skel.bvh_joint_order,
        #                                                      new_joint_order=dataset.skel.lh_fingers_only)
        # p1_rhand_wrist_pos = global_pose1[:, :, 18]
        # p2_rhand_wrist_pos = global_pose2[:, :, 18]
        # p1_lhand_wrist_pos = global_pose1[:, :, 43]
        # p2_lhand_wrist_pos = global_pose2[:, :, 43]
        # rhand_p1_offsets = dataset.skel.select_offset(offset=annot['offsets_1'], new_joint_order=dataset.skel.rh_fingers_only)
        # rhand_p2_offsets = dataset.skel.select_offset(offset=annot['offsets_2'], new_joint_order=dataset.skel.rh_fingers_only)
        # lhand_p1_offsets = dataset.skel.select_offset(offset=annot['offsets_1'], new_joint_order=dataset.skel.lh_fingers_only)
        # lhand_p2_offsets = dataset.skel.select_offset(offset=annot['offsets_2'], new_joint_order=dataset.skel.lh_fingers_only)
        # rhand_fk1 = dataset.skel.forward_kinematics_rhand(pose_root=p1_rhand_wrist_pos, full_rot=p1_rhand_rot,
        #                                       offsets=rhand_p1_offsets, scale=1, rotation_format='6d')
        # lhand_fk1 = dataset.skel.forward_kinematics_lhand(pose_root=p1_lhand_wrist_pos, full_rot=p1_lhand_rot,
        #                                       offsets=lhand_p1_offsets, scale=1, rotation_format='6d')

        # fk_fullp1 = dataset.skel.revert_original_bvh_joints_poses(lhand_fk1, dataset.skel.bvh_joint_order,
        #                                                           dataset.skel.lh_fingers_only, global_pose1)
        # fk_fullp1 = dataset.skel.revert_original_bvh_joints_poses(rhand_fk1, dataset.skel.bvh_joint_order,
                                                                #   dataset.skel.rh_fingers_only, fk_fullp1)
        plot_contacts3D(global_pose1[0].detach().cpu().numpy(), global_pose2[0].detach().cpu().numpy(),
                        # gt_pose2=None,
                        # gt_pose2=global_pose1[0].detach().cpu().numpy(),
                        savepath=makepath(os.path.join(args.render_path, 'Lindyhop', 'fk_hands'+ split, str(j)), 
                                 isfile=True), kinematic_chain = 'full', onlyone=False)
    # for j, annot in enumerate(dataloader):
    #     global_pose1 = annot['pose_canon_1'].to(device).float()
    #     global_pose2 = annot['pose_canon_2'].to(device).float()
        
    #     # # if you only take the inverse of first frame and keep first frame of root at origin
    #     # inverse_eul_rot = batch_rotmat_to_euler(annot['global_root_rotation'].unsqueeze(1))
    #     # global_root_rotation = torch.repeat_interleave(inverse_eul_rot.unsqueeze(1), window_size, axis=1).to(device).float()
    #     # global_root_origin = torch.repeat_interleave(annot['global_root_origin'].unsqueeze(1),
    #     #                                                             window_size, axis=1).to(device).float()
        
        
    #     # if you take the inverse of root at all frames and keep all frames root at origin
    #     global_root_rotation = annot['global_root_rotation'].to(device).float()
        
    #     global_root_origin = annot['global_root_origin'].to(device).float()
        
    #     pose1_root_relative, pose2_root_relative = dataset.root_relative_normalization(global_pose1, global_pose2,
    #                                                                                    global_root_rotation, global_root_origin)
    #     global_action, global_reaction = dataset.root_relative_unnormalization(pose1_root_relative, pose2_root_relative,
    #                                                                            global_root_rotation, global_root_origin)
    #     plot_contacts3D(global_action[0].detach().cpu().numpy(), global_reaction[0].detach().cpu().numpy(),
    #                     gt_pose2=None, savepath=makepath(os.path.join(args.render_path, 'Lindyhop', 'normalized'+ split, str(j)), 
    #                              isfile=True), kinematic_chain = 'no_fingers', onlyone=False)
    # for j, annot in enumerate(dataloader):
    #     global_root_origin = annot['pose_canon_1'][:, :, 0]
    #     pose1_root =  annot['pose_canon_1'][:, :, 0] -  global_root_origin
    #     pose2_root = annot['pose_canon_2'][:, :, 0] - global_root_origin
    #     global_root_rotation = np.linalg.inv(annot['rotmat_1'][:, :, 0])
    #     rotmat_1 = annot['rotmat_1']
    #     rotmat_2 = annot['rotmat_2']
        # rotmat_1 = np.matmul(rotmat_1, global_root_rotation)
        # rotmat_2 = np.matmul(rotmat_2, global_root_rotation)
        # body_fk_1 = skel_fk.forward_kinematics(pose_root=pose1_root, full_rot=rotmat_1, offsets=annot['offsets_1'], scale=scale, rotation_format='rotmat')
    #     body_fk_2 = skel_fk.forward_kinematics(pose_root=pose2_root, full_rot=rotmat_2, offsets=annot['offsets_2'], scale=scale, rotation_format='rotmat')

         
        
    #     plot_contacts3D(body_fk_1[0], body_fk_2[0], makepath(os.path.join(args.render_path, 'Lindyhop', 'fk_'+ split, str(j)), isfile=True), kinematic_chain='full')
    #     plot_contacts3D(annot['pose_canon_1'][0], annot['pose_canon_2'][0], makepath(os.path.join(args.render_path, 'Lindyhop', 'canon_'+ split, str(j)), isfile=True), kinematic_chain='full')
    #     #
    #     # if not bvh:
    #     #     plot_contacts3D(pose_canon_1 , pose_canon_2 ,savepath, kinematic_chain='full')
    #     #     # pose_canon_1[..., 0] = - pose_canon_1[..., 0]
    #     #     # pose_canon_2[..., 0] = - pose_canon_2[..., 0]
    #     #     # pose_canon_1[..., 2] = - pose_canon_1[..., 2]
    #     #     # pose_canon_2[..., 2] = - pose_canon_2[..., 2]
    #     #     # plot_contacts3D(pose_canon_1, pose_canon_2, savepath+'_flip')    


    #     # I_K = 1
    #     # if I_K:
    #     #     bvh1 = makepath(os.path.join(args.render_path, 'bvh', split, str(j), '1.bvh'), isfile=True)
    #     #     bvh2 = makepath(os.path.join(args.render_path, 'bvh', split, str(j), '2.bvh'), isfile=True)
    #     #     channel1, header1 = skel_ik.poses2bvh(batch['pose_canon_1'][0], output_file=bvh1, offset=batch['offsets_1'])
    #     #     channel2, header2 = skel_ik.poses2bvh(batch['pose_canon_2'][0], output_file=bvh2, offset=batch['offsets_2'])
    #     #     # plot_contacts3D(pose_canon_1 , pose_canon_2 , makepath(os.path.join(args.render_path, framerate_folder, 'canon_output_'+split, str(j)), isfile=True), kinematic_chain='full',  onlyone=False)
        count+=1
    print(count)
    
