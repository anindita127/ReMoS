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
    
