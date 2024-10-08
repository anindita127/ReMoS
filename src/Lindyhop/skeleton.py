import numpy as np
import os 
import pickle
import sys
import torch
import pytorch3d.transforms as t3dt
from src.tools.transformations import batch_6d_to_euler_tensor, batch_rotmat_to_euler
from math import cos, sin

def select_bvh_joints_rot6d(data, joint_order_full, joint_order):  
    output = torch.zeros((data.shape[0], data.shape[1], len(joint_order_full.keys()), data.shape[-1]))
    output[..., 0] = 1.
    output[..., 4] = 1.
    for j, joint in enumerate(joint_order_full.keys()):
        if joint in joint_order.keys():
            idx_in_bvh = joint_order[joint]
            output[:, :, j] = data[:, :, idx_in_bvh]
    return output

def select_bvh_joints_rotmat(data, joint_order_full, joint_order): 
    if torch.is_tensor(data):
        output = torch.zeros((data.shape[0], data.shape[1], len(joint_order_full.keys()), 3, 3))
    else:
        output = np.zeros((data.shape[0], data.shape[1], len(joint_order_full.keys()), 3, 3))
    output[..., 0, 0] = 1.
    output[..., 1, 1] = 1.
    output[..., 2, 2] = 1.
    for j, joint in enumerate(joint_order_full.keys()):
        if joint in joint_order.keys():
            idx_in_bvh = joint_order[joint]
            output[:, :, j] = data[:, :, idx_in_bvh]
    return output

class InhouseStudioSkeleton:
    def __init__(self):
        

        self.bvh_joint_order = {
                'Hips': 0,
                'RightUpLeg': 1,
                'RightLeg': 2,
                'RightFoot': 3,
                'RightToeBase': 4,
                'RightToeBaseEnd': 5,
                'LeftUpLeg': 6,
                'LeftLeg': 7,
                'LeftFoot': 8,
                'LeftToeBase': 9,
                'LeftToeBaseEnd': 10,
                'Spine': 11,
                'Spine1': 12,
                'Spine2': 13,
                'Spine3': 14,
                'RightShoulder': 15,
                'RightArm': 16,
                'RightForeArm': 17,
                'RightHand': 18,
                'RightHandEnd': 19,
                'RightHandPinky1': 20,
                'RightHandPinky2': 21,
                'RightHandPinky3': 22,
                'RightHandPinky3End': 23,
                'RightHandRing1': 24,
                'RightHandRing2': 25,
                'RightHandRing3': 26,
                'RightHandRing3End': 27,
                'RightHandMiddle1': 28,
                'RightHandMiddle2': 29,
                'RightHandMiddle3': 30,
                'RightHandMiddle3End': 31,
                'RightHandIndex1': 32,
                'RightHandIndex2': 33,
                'RightHandIndex3': 34,
                'RightHandIndex3End': 35,
                'RightHandThumb1': 36,
                'RightHandThumb2': 37,
                'RightHandThumb3': 38,
                'RightHandThumb3End': 39,
                'LeftShoulder': 40,
                'LeftArm': 41,
                'LeftForeArm': 42,
                'LeftHand': 43,
                'LeftHandEnd': 44,
                'LeftHandPinky1': 45,
                'LeftHandPinky2': 46,
                'LeftHandPinky3': 47,
                'LeftHandPinky3End': 48,
                'LeftHandRing1': 49,
                'LeftHandRing2': 50,
                'LeftHandRing3': 51,
                'LeftHandRing3End': 52,
                'LeftHandMiddle1': 53,
                'LeftHandMiddle2': 54,
                'LeftHandMiddle3': 55,
                'LeftHandMiddle3End': 56,
                'LeftHandIndex1': 57,
                'LeftHandIndex2': 58,
                'LeftHandIndex3': 59,
                'LeftHandIndex3End': 60,
                'LeftHandThumb1': 61,
                'LeftHandThumb2': 62,
                'LeftHandThumb3': 63,
                'LeftHandThumb3End': 64,
                'Spine4': 65,
                'Neck': 66,
                'Head': 67,
                'HeadEnd': 68
                }
        
        self.bvh_joint_reduced = {
                'Hips': 0,
                'RightUpLeg': 1,
                'RightLeg': 2,
                'RightFoot': 3,
                'RightToeBase': 4,
                'RightToeBaseEnd': 5,
                'LeftUpLeg': 6,
                'LeftLeg': 7,
                'LeftFoot': 8,
                'LeftToeBase': 9,
                'LeftToeBaseEnd': 10,
                'Spine': 11,
                'Spine1': 12,
                'Spine2': 13,
                'Spine3': 14,
                'RightShoulder': 15,
                'RightArm': 16,
                'RightForeArm': 17,
                'RightHand': 18,
                'RightHandPinky1': 19,
                'RightHandPinky3End': 20,
                'RightHandRing1': 21,
                'RightHandRing3End': 22,
                'RightHandMiddle1': 23,
                'RightHandMiddle3End': 24,
                'RightHandIndex1': 25,
                'RightHandIndex3End': 26,
                'RightHandThumb1': 27,
                'RightHandThumb3End': 28,
                'LeftShoulder': 29,
                'LeftArm': 30,
                'LeftForeArm': 31,
                'LeftHand': 32,
                'LeftHandPinky1': 33,
                'LeftHandPinky3End': 34,
                'LeftHandRing1': 35,
                'LeftHandRing3End': 36,
                'LeftHandMiddle1': 37,
                'LeftHandMiddle3End': 38,
                'LeftHandIndex1': 39,
                'LeftHandIndex3End': 40,
                'LeftHandThumb1': 41,
                'LeftHandThumb3End': 42,
                'Spine4': 43,
                'Neck': 44,
                'Head': 45,
                'HeadEnd': 46
                }
        
        
        
        self.body_only = {
                'Hips': 0,
                'RightUpLeg': 1,
                'RightLeg': 2,
                'RightFoot': 3,
                'RightToeBase': 4,
                'RightToeBaseEnd': 5,
                'LeftUpLeg': 6,
                'LeftLeg': 7,
                'LeftFoot': 8,
                'LeftToeBase': 9,
                'LeftToeBaseEnd': 10,
                'Spine': 11,
                'Spine1': 12,
                'Spine2': 13,
                'Spine3': 14,
                'RightShoulder': 15,
                'RightArm': 16,
                'RightForeArm': 17,
                'RightHand': 18,
                'LeftShoulder': 19,
                'LeftArm': 20,
                'LeftForeArm': 21,
                'LeftHand': 22,
                'Spine4': 23,
                'Neck': 24,
                'Head': 25,
                'HeadEnd': 26
        }
        self.old_joint_order = {
                'Hips': 0,
                'LeftUpLeg': 1,
                'LeftLeg': 2,
                'LeftFoot': 3,
                'LeftToeBase': 4,
                'LeftToeBaseEnd': 5,
                'RightUpLeg': 6,
                'RightLeg': 7,
                'RightFoot': 8,
                'RightToeBase': 9,
                'RightToeBaseEnd': 10,
                'Spine': 11,
                'Spine3': 12,
                'Neck': 13,
                'HeadEnd': 14,
                'LeftShoulder': 15,
                'LeftArm': 16,
                'LeftForeArm': 17,
                'RightShoulder': 18,
                'RightArm': 19,
                'RightForeArm': 20,
                'LeftHand': 21,
                'LeftHandEnd': 22,
                'LeftHandPinky2': 23,              
                'LeftHandPinky3End': 24,
                'LeftHandRing2': 25,
                'LeftHandRing3End': 26,
                'LeftHandMiddle2': 27,
                'LeftHandMiddle3End': 28,
                'LeftHandIndex2': 29,
                'LeftHandIndex3End': 30,
                'LeftHandThumb2': 31,
                'LeftHandThumb3End': 32,
                'RightHand': 33,
                'RightHandEnd': 34,
                'RightHandPinky2': 35,              
                'RightHandPinky3End': 36,
                'RightHandRing2': 37,
                'RightHandRing3End': 38,
                'RightHandMiddle2': 39,
                'RightHandMiddle3End': 40,
                'RightHandIndex2': 41,
                'RightHandIndex3End': 42,
                'RightHandThumb2': 43,
                'RightHandThumb3End': 44,
                }
        self.rh_fingers = {
            'RightHand': 0,
            'RightHandEnd': 1,
            'RightHandPinky1': 2,
            'RightHandPinky2': 3,
            'RightHandPinky3': 4,
            'RightHandPinky3End': 5,
            'RightHandRing1': 6,
            'RightHandRing2': 7,
            'RightHandRing3': 8,
            'RightHandRing3End': 9,
            'RightHandMiddle1': 10,
            'RightHandMiddle2': 11,
            'RightHandMiddle3': 12,
            'RightHandMiddle3End': 13,
            'RightHandIndex1': 14,
            'RightHandIndex2': 15,
            'RightHandIndex3': 16,
            'RightHandIndex3End': 17,
            'RightHandThumb1': 18,
            'RightHandThumb2': 19,
            'RightHandThumb3': 20,
            'RightHandThumb3End': 21,
        }
        self.lh_fingers = {
            'LeftHand': 0,
            'LeftHandEnd': 1,
            'LeftHandPinky1': 2,
            'LeftHandPinky2': 3,
            'LeftHandPinky3': 4,
            'LeftHandPinky3End': 5,
            'LeftHandRing1': 6,
            'LeftHandRing2': 7,
            'LeftHandRing3': 8,
            'LeftHandRing3End': 9,
            'LeftHandMiddle1': 10,
            'LeftHandMiddle2': 11,
            'LeftHandMiddle3': 12,
            'LeftHandMiddle3End': 13,
            'LeftHandIndex1': 14,
            'LeftHandIndex2': 15,
            'LeftHandIndex3': 16,
            'LeftHandIndex3End': 17,
            'LeftHandThumb1': 18,
            'LeftHandThumb2': 19,
            'LeftHandThumb3': 20,
            'LeftHandThumb3End': 21,
        }
        self.rh_fingers_only = {
            'RightHand': 0,
            # 'RightHandEnd': 1,
            'RightHandPinky1': 1,
            # 'RightHandPinky2': 2,
            # 'RightHandPinky3': 4,
            'RightHandPinky3End': 2,
            'RightHandRing1': 3,
            # 'RightHandRing2': 5,
            # 'RightHandRing3': 8,
            'RightHandRing3End': 4,
            'RightHandMiddle1': 5,
            # 'RightHandMiddle2': 8,
            # 'RightHandMiddle3': 12,
            'RightHandMiddle3End': 6,
            'RightHandIndex1': 7,
            # 'RightHandIndex2': 11,
            # 'RightHandIndex3': 16,
            'RightHandIndex3End': 8,
            'RightHandThumb1': 9,
            # 'RightHandThumb2': 14,
            # 'RightHandThumb3': 20,
            'RightHandThumb3End': 10,
        }
        self.lh_fingers_only = {
            'LeftHand': 0,
            # 'LeftHandEnd': 1,
            'LeftHandPinky1': 1,
            # 'LeftHandPinky2': 2,
            # 'LeftHandPinky3': 4,
            'LeftHandPinky3End': 2,
            'LeftHandRing1': 3,
            # 'LeftHandRing2': 4,
            # 'LeftHandRing3': 8,
            'LeftHandRing3End': 4,
            'LeftHandMiddle1': 5,
            # 'LeftHandMiddle2': 8,
            # 'LeftHandMiddle3': 12,
            'LeftHandMiddle3End': 6,
            'LeftHandIndex1': 7,
            # 'LeftHandIndex2': 11,
            # 'LeftHandIndex3': 16,
            'LeftHandIndex3End': 8,
            'LeftHandThumb1': 9,
            # 'LeftHandThumb2': 14,
            # 'LeftHandThumb3': 20,
            'LeftHandThumb3End': 10,
        }
        self.parent_fingers = [-1, 0, 1, 0, 3, 0, 5, 0, 7, 0,  9, -1, 11, 12, 11, 14, 11, 16, 11, 18, 11, 20]
                            #   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        self.parents_body_only = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13,
                                  14, 15, 16, 17, 14, 19, 20, 21, 14, 23, 24, 25]
        self.parents_full = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8,  9,  0, 11, 12, 13, 14, 15, 16, 17, 18, 18, 20,
                          #   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
                              21, 22, 18, 24, 25, 26, 18, 28, 29, 30, 18, 32, 33, 34, 18, 36, 37, 38, 14, 40, 41, 42,
                          #   22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
                              43, 43, 45, 46, 47, 43, 49, 50, 51, 43, 53, 54, 55, 43, 57, 58, 59, 43, 61, 62, 63, 14, 65, 66, 67]
                          #   44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68
        self.bvh_joint_parents = {
                'Hips': -1,
                'RightUpLeg': 'Hips',
                'RightLeg': 'RightUpLeg',
                'RightFoot': 'RightLeg',
                'RightToeBase': 'RightFoot',
                'RightToeBaseEnd': 'RightToeBase',
                'LeftUpLeg': 'Hips',
                'LeftLeg': 'LeftUpLeg',
                'LeftFoot': 'LeftLeg',
                'LeftToeBase': 'LeftFoot',
                'LeftToeBaseEnd': 'LeftToeBase',
                'Spine': 'Hips',
                'Spine1': 'Spine',
                'Spine2': 'Spine1',
                'Spine3': 'Spine2',
                'RightShoulder': 'Spine3',
                'RightArm': 'RightShoulder',
                'RightForeArm': 'RightArm',
                'RightHand': 'RightForeArm',
                'RightHandEnd': 'RightHand',
                'RightHandPinky1': 'RightHand',
                'RightHandPinky2': 'RightHandPinky1',
                'RightHandPinky3': 'RightHandPinky2',
                'RightHandPinky3End': 'RightHandPinky3',
                'RightHandRing1': 'RightHand',
                'RightHandRing2': 'RightHandRing1',
                'RightHandRing3': 'RightHandRing2',
                'RightHandRing3End': 'RightHandRing3',
                'RightHandMiddle1': 'RightHand',
                'RightHandMiddle2': 'RightHandMiddle1',
                'RightHandMiddle3': 'RightHandMiddle2',
                'RightHandMiddle3End': 'RightHandMiddle3',
                'RightHandIndex1': 'RightHand',
                'RightHandIndex2': 'RightHandIndex1',
                'RightHandIndex3': 'RightHandIndex2',
                'RightHandIndex3End': 'RightHandIndex3',
                'RightHandThumb1': 'RightHand',
                'RightHandThumb2': 'RightHandThumb1',
                'RightHandThumb3': 'RightHandThumb2',
                'RightHandThumb3End': 'RightHandThumb3',
                'LeftShoulder': 'Spine3',
                'LeftArm': 'LeftShoulder',
                'LeftForeArm': 'LeftArm',
                'LeftHand': 'LeftForeArm',
                'LeftHandEnd': 'LeftHand',
                'LeftHandPinky1': 'LeftHand',
                'LeftHandPinky2': 'LeftHandPinky1',
                'LeftHandPinky3': 'LeftHandPinky2',
                'LeftHandPinky3End': 'LeftHandPinky3',
                'LeftHandRing1': 'LeftHand',
                'LeftHandRing2': 'LeftHandRing1',
                'LeftHandRing3': 'LeftHandRing2',
                'LeftHandRing3End': 'LeftHandRing3',
                'LeftHandMiddle1': 'LeftHand',
                'LeftHandMiddle2': 'LeftHandMiddle1',
                'LeftHandMiddle3': 'LeftHandMiddle2',
                'LeftHandMiddle3End': 'LeftHandMiddle3',
                'LeftHandIndex1': 'LeftHand',
                'LeftHandIndex2': 'LeftHandIndex1',
                'LeftHandIndex3': 'LeftHandIndex2',
                'LeftHandIndex3End': 'LeftHandIndex3',
                'LeftHandThumb1': 'LeftHand',
                'LeftHandThumb2': 'LeftHandThumb1',
                'LeftHandThumb3': 'LeftHandThumb2',
                'LeftHandThumb3End': 'LeftHandThumb3',
                'Spine4': 'Spine3',
                'Neck': 'Spine4',
                'Head': 'Neck',
                'HeadEnd': 'Head'
                }
        self.bvh_lhand_parents = {
                'LeftHand': -1,
                'LeftHandEnd': 'LeftHand',
                'LeftHandPinky1': 'LeftHand',
                'LeftHandPinky2': 'LeftHandPinky1',
                'LeftHandPinky3': 'LeftHandPinky2',
                'LeftHandPinky3End': 'LeftHandPinky3',
                'LeftHandRing1': 'LeftHand',
                'LeftHandRing2': 'LeftHandRing1',
                'LeftHandRing3': 'LeftHandRing2',
                'LeftHandRing3End': 'LeftHandRing3',
                'LeftHandMiddle1': 'LeftHand',
                'LeftHandMiddle2': 'LeftHandMiddle1',
                'LeftHandMiddle3': 'LeftHandMiddle2',
                'LeftHandMiddle3End': 'LeftHandMiddle3',
                'LeftHandIndex1': 'LeftHand',
                'LeftHandIndex2': 'LeftHandIndex1',
                'LeftHandIndex3': 'LeftHandIndex2',
                'LeftHandIndex3End': 'LeftHandIndex3',
                'LeftHandThumb1': 'LeftHand',
                'LeftHandThumb2': 'LeftHandThumb1',
                'LeftHandThumb3': 'LeftHandThumb2',
                'LeftHandThumb3End': 'LeftHandThumb3',
                }
        
        self.bvh_rhand_parents = {
                'RightHand': -1,
                'RightHandEnd': 'RightHand',
                'RightHandPinky1': 'RightHand',
                'RightHandPinky2': 'RightHandPinky1',
                'RightHandPinky3': 'RightHandPinky2',
                'RightHandPinky3End': 'RightHandPinky3',
                'RightHandRing1': 'RightHand',
                'RightHandRing2': 'RightHandRing1',
                'RightHandRing3': 'RightHandRing2',
                'RightHandRing3End': 'RightHandRing3',
                'RightHandMiddle1': 'RightHand',
                'RightHandMiddle2': 'RightHandMiddle1',
                'RightHandMiddle3': 'RightHandMiddle2',
                'RightHandMiddle3End': 'RightHandMiddle3',
                'RightHandIndex1': 'RightHand',
                'RightHandIndex2': 'RightHandIndex1',
                'RightHandIndex3': 'RightHandIndex2',
                'RightHandIndex3End': 'RightHandIndex3',
                'RightHandThumb1': 'RightHand',
                'RightHandThumb2': 'RightHandThumb1',
                'RightHandThumb3': 'RightHandThumb2',
                'RightHandThumb3End': 'RightHandThumb3',
                }
        self.joint_order = {
                'Hips': 0,
                'LeftUpLeg': 1,
                'LeftLeg': 2,
                'LeftFoot': 3,
                'LeftToeBase': 4,
                'LeftToeBaseEnd': 5,
                'RightUpLeg': 6,
                'RightLeg': 7,
                'RightFoot': 8,
                'RightToeBase': 9,
                'RightToeBaseEnd': 10,
                'Spine': 11,
                'Spine3': 12,
                'Neck': 13,
                'HeadEnd': 14,
                'LeftShoulder': 15,
                'LeftArm': 16,
                'LeftForeArm': 17,
                'RightShoulder': 18,
                'RightArm': 19,
                'RightForeArm': 20,
                'LeftHand': 21,
                'LeftHandEnd': 22,
                'LeftHandPinky2': 23,              
                'LeftHandPinky3End': 24,
                'LeftHandRing2': 25,
                'LeftHandRing3End': 26,
                'LeftHandMiddle2': 27,
                'LeftHandMiddle3End': 28,
                'LeftHandIndex2': 29,
                'LeftHandIndex3End': 30,
                'LeftHandThumb2': 31,
                'LeftHandThumb3End': 32,
                'RightHand': 33,
                'RightHandEnd': 34,
                'RightHandPinky2': 35,              
                'RightHandPinky3End': 36,
                'RightHandRing2': 37,
                'RightHandRing3End': 38,
                'RightHandMiddle2': 39,
                'RightHandMiddle3End': 40,
                'RightHandIndex2': 41,
                'RightHandIndex3End': 42,
                'RightHandThumb2': 43,
                'RightHandThumb3End': 44,
                }
        

        self.bvh_joint_children = {
            'Hips': ['Spine', 'LeftUpLeg', 'RightUpLeg'],
            'Spine': ['Spine1'],
            'Spine1': ['Spine2'],
            'Spine2': ['Spine3'],
            'Spine3': ['Spine4', 'LeftShoulder', 'RightShoulder'],
            'Spine4': ['Neck'],
            'Neck': ['Head'],
            'Head': ['HeadEnd'],
            'HeadEnd': [],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'LeftHand': ['LeftHandThumb1', 'LeftHandIndex1', 'LeftHandMiddle1', 'LeftHandRing1', 'LeftHandPinky1', 'LeftHandEnd'],
            'LeftHandThumb1': ['LeftHandThumb2'],
            'LeftHandThumb2': ['LeftHandThumb3'],
            'LeftHandThumb3': ['LeftHandThumb3End'],
            'LeftHandThumb3End': [],
            'LeftHandIndex1': ['LeftHandIndex2'],
            'LeftHandIndex2': ['LeftHandIndex3'],
            'LeftHandIndex3': ['LeftHandIndex3End'],
            'LeftHandIndex3End': [],
            'LeftHandMiddle1': ['LeftHandMiddle2'],
            'LeftHandMiddle2': ['LeftHandMiddle3'],
            'LeftHandMiddle3': ['LeftHandMiddle3End'],
            'LeftHandMiddle3End': [],
            'LeftHandRing1': ['LeftHandRing2'],
            'LeftHandRing2': ['LeftHandRing3'],
            'LeftHandRing3': ['LeftHandRing3End'],
            'LeftHandRing3End': [],
            'LeftHandPinky1': ['LeftHandPinky2'],              
            'LeftHandPinky2': ['LeftHandPinky3'],              
            'LeftHandPinky3': ['LeftHandPinky3End'],              
            'LeftHandPinky3End': [],
            'LeftHandEnd': [],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'RightHand': ['RightHandThumb1', 'RightHandIndex1', 'RightHandMiddle1', 'RightHandRing1', 'RightHandPinky1', 'RightHandEnd'],
            'RightHandThumb1': ['RightHandThumb2'],
            'RightHandThumb2': ['RightHandThumb3'],
            'RightHandThumb3': ['RightHandThumb3End'],
            'RightHandThumb3End': [],
            'RightHandIndex1': ['RightHandIndex2'],
            'RightHandIndex2': ['RightHandIndex3'],
            'RightHandIndex3': ['RightHandIndex3End'],
            'RightHandIndex3End': [],
            'RightHandMiddle1': ['RightHandMiddle2'],
            'RightHandMiddle2': ['RightHandMiddle3'],
            'RightHandMiddle3': ['RightHandMiddle3End'],
            'RightHandMiddle3End': [],
            'RightHandRing1': ['RightHandRing2'],
            'RightHandRing2': ['RightHandRing3'],
            'RightHandRing3': ['RightHandRing3End'],
            'RightHandRing3End': [],
            'RightHandPinky1': ['RightHandPinky2'],              
            'RightHandPinky2': ['RightHandPinky3'],              
            'RightHandPinky3': ['RightHandPinky3End'],              
            'RightHandPinky3End': [],
            'RightHandEnd': [],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftToeBase'],
            'LeftToeBase': ['LeftToeBaseEnd'],
            'LeftToeBaseEnd': [],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightToeBase'],
            'RightToeBase': ['RightToeBaseEnd'],
            'RightToeBaseEnd': [],
        }
        self.bvh_joint_names = list(self.bvh_joint_parents.keys())

    def revert_original_bvh_joints_rot6d(self, data, joint_order_full, joint_order):  
        output = torch.zeros((data.shape[0], data.shape[1], len(joint_order_full.keys()), data.shape[-1])).to(data.device)
        output[..., 0] = 1.
        output[..., 4] = 1.
        for j, joint in enumerate(joint_order_full.keys()):
            if joint in joint_order.keys():
                idx_in_bvh = joint_order[joint]
                output[:, :, j] = data[:, :, idx_in_bvh]
        return output

    def revert_original_bvh_joints_rotmat(self, data, joint_order_full, joint_order):  
        output = torch.zeros((data.shape[0], data.shape[1], len(joint_order_full.keys()), 3, 3)).to(data.device)
        output[..., 0, 0] = 1.
        output[..., 1, 1] = 1.
        output[..., 2, 2] = 1.
        for j, joint in enumerate(joint_order_full.keys()):
            if joint in joint_order.keys():
                idx_in_bvh = joint_order[joint]
                output[:, :, j] = data[:, :, idx_in_bvh]
        return output

    def revert_original_bvh_joints_poses(self, data, joint_order_full, joint_order, output=None):  
        if output is None:
            output = torch.zeros((data.shape[0], data.shape[1], 
                                  len(joint_order_full.keys()), data.shape[-1])).to(data.device)
        
        for j, joint in enumerate(joint_order_full.keys()):
            if joint in joint_order.keys():
                idx_in_bvh = joint_order[joint]
                output[:, :, j] = data[:, :, idx_in_bvh]
        return output

    def select_bvh_joints(self, data, original_joint_order, new_joint_order):
        """
            IMP: the rotations file should not have the root transslation in them. 
            Remove them before sending here
        """
        
        output = torch.zeros((data.shape[0], data.shape[1], len(new_joint_order.keys()), data.shape[-1])).to(data.device)
        for j, joint in enumerate(new_joint_order.keys()):
            idx_in_bvh = original_joint_order[joint]
            output[:, :, j] = data[:, :, idx_in_bvh]
        return output
    
    def select_offset(self, offset, new_joint_order):
        output = {}
        for idx, keys in enumerate(new_joint_order.keys()):
            output[keys] = offset[keys]  
        return output
    
    def forward_kinematics(self, pose_root, full_rot, offsets, scale = 1000., rotation_format='rotmat'):
        if rotation_format == 'rotmat': 
            if full_rot.shape[-3] != 69:
                full_rot = select_bvh_joints_rotmat(full_rot, self.bvh_joint_order, self.joint_order) 
            full_rot = batch_rotmat_to_euler(full_rot)
        elif rotation_format == '6d': 
            if full_rot.shape[-2] != 69:
                full_rot = select_bvh_joints_rot6d(full_rot, self.bvh_joint_order, self.joint_order) 
            full_rot = batch_6d_to_euler_tensor(full_rot)
        worldpose = torch.zeros_like(full_rot)
        joint_trtr = {
        'Hips': [],
        'RightUpLeg': [],
        'RightLeg': [],
        'RightFoot': [],
        'RightToeBase': [],
        'RightToeBaseEnd': [],
        'LeftUpLeg': [],
        'LeftLeg': [],
        'LeftFoot': [],
        'LeftToeBase': [],
        'LeftToeBaseEnd': [],
        'Spine': [],
        'Spine1': [],
        'Spine2': [],
        'Spine3': [],
        'RightShoulder': [],
        'RightArm': [],
        'RightForeArm': [],
        'RightHand': [],
        'RightHandEnd': [],
        'RightHandPinky1': [],
        'RightHandPinky2': [],
        'RightHandPinky3': [],
        'RightHandPinky3End': [],
        'RightHandRing1': [],
        'RightHandRing2': [],
        'RightHandRing3': [],
        'RightHandRing3End': [],
        'RightHandMiddle1': [],
        'RightHandMiddle2': [],
        'RightHandMiddle3': [],
        'RightHandMiddle3End': [],
        'RightHandIndex1': [],
        'RightHandIndex2': [],
        'RightHandIndex3': [],
        'RightHandIndex3End': [],
        'RightHandThumb1': [],
        'RightHandThumb2': [],
        'RightHandThumb3': [],
        'RightHandThumb3End': [],
        'LeftShoulder': [],
        'LeftArm': [],
        'LeftForeArm': [],
        'LeftHand': [],
        'LeftHandEnd': [],
        'LeftHandPinky1': [],
        'LeftHandPinky2': [],
        'LeftHandPinky3': [],
        'LeftHandPinky3End': [],
        'LeftHandRing1': [],
        'LeftHandRing2': [],
        'LeftHandRing3': [],
        'LeftHandRing3End': [],
        'LeftHandMiddle1': [],
        'LeftHandMiddle2': [],
        'LeftHandMiddle3': [],
        'LeftHandMiddle3End': [],
        'LeftHandIndex1': [],
        'LeftHandIndex2': [],
        'LeftHandIndex3': [],
        'LeftHandIndex3End': [],
        'LeftHandThumb1': [],
        'LeftHandThumb2': [],
        'LeftHandThumb3': [],
        'LeftHandThumb3End': [],
        'Spine4': [],
        'Neck': [],
        'Head': [],
        'HeadEnd': []
    }

        num_batches = full_rot.shape[0]
        num_time_steps = full_rot.shape[1]
        device = full_rot.device
        identity = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(num_batches, num_time_steps, 1, 1).to(device)
        for joint_name, jt in self.bvh_joint_order.items():
            jt = int(jt)
            drotmat = identity.clone()
            joint_rot = full_rot[..., jt, :]                                

            strans = offsets[joint_name].unsqueeze(1).repeat(1, num_time_steps, 1) / scale
            # Transformation matrices:
            stransmat = identity.clone()
            stransmat[:, :, :3, 3] = strans

            if joint_name == 'Hips':
                dtransmat = identity.clone()
                dtransmat[:, :, :3, 3] = pose_root
                
            
            xrot = joint_rot[..., 0]
            mycos = torch.cos(xrot)
            mysin = torch.sin(xrot)
            drotmat2 = identity.clone()
            drotmat2[..., 1, 1] = mycos
            drotmat2[..., 1, 2] = -mysin
            drotmat2[..., 2, 1] = mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            yrot = joint_rot[..., 1]
            mycos = torch.cos(yrot)
            mysin = torch.sin(yrot)
            drotmat2 = identity.clone()        
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 2] = mysin
            drotmat2[..., 2, 0] = -mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            zrot = joint_rot[..., 2]
            mycos = torch.cos(zrot)
            mysin = torch.sin(zrot)
            drotmat2 = identity.clone()
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 1] = -mysin
            drotmat2[..., 1, 0] = mysin
            drotmat2[..., 1, 1] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            joint_parent_name = self.bvh_joint_parents[joint_name]
            if joint_parent_name != -1:  # Not hips
                parent_trtr = joint_trtr[joint_parent_name]  # Dictionary-based rewrite
                localtoworld = torch.einsum('btij, btjk -> btik', parent_trtr, stransmat)
            else:
                localtoworld = torch.einsum('btij, btjk -> btik', stransmat, dtransmat)
            
            trtr = torch.einsum('btij, btjk -> btik', localtoworld, drotmat)
            joint_trtr[joint_name] = trtr  # New dictionary-based approach

            # worldpos = localtoworld * ORIGIN  # worldpos should be a vec4
            worldpose[..., jt, :] = localtoworld[..., :3, 3]
        # joint.worldpos[t] = worldpos 
        return worldpose
    
    def forward_kinematics_lhand(self, pose_root, full_rot, offsets, scale = 1000., rotation_format='rotmat'):
        if rotation_format == 'rotmat': 
            if full_rot.shape[-3] != 22:
                full_rot = select_bvh_joints_rotmat(full_rot, self.bvh_joint_order, self.lh_fingers_only) 
            full_rot = batch_rotmat_to_euler(full_rot)
        elif rotation_format == '6d': 
            if full_rot.shape[-2] != 22:
                full_rot = select_bvh_joints_rot6d(full_rot, self.bvh_joint_order, self.lh_fingers_only) 
            full_rot = batch_6d_to_euler_tensor(full_rot)
        worldpose = torch.zeros_like(full_rot)
        joint_trtr = {
        'LeftHand': [],
        'LeftHandEnd': [],
        'LeftHandPinky1': [],
        'LeftHandPinky2': [],
        'LeftHandPinky3': [],
        'LeftHandPinky3End': [],
        'LeftHandRing1': [],
        'LeftHandRing2': [],
        'LeftHandRing3': [],
        'LeftHandRing3End': [],
        'LeftHandMiddle1': [],
        'LeftHandMiddle2': [],
        'LeftHandMiddle3': [],
        'LeftHandMiddle3End': [],
        'LeftHandIndex1': [],
        'LeftHandIndex2': [],
        'LeftHandIndex3': [],
        'LeftHandIndex3End': [],
        'LeftHandThumb1': [],
        'LeftHandThumb2': [],
        'LeftHandThumb3': [],
        'LeftHandThumb3End': [],
    }

        num_batches = full_rot.shape[0]
        num_time_steps = full_rot.shape[1]
        device = full_rot.device
        identity = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(num_batches, num_time_steps, 1, 1).to(device)
        for joint_name, jt in self.lh_fingers_only.items():
            jt = int(jt)
            drotmat = identity.clone()
            joint_rot = full_rot[..., jt, :]                                

            strans = torch.tensor(offsets[joint_name])[None].unsqueeze(1).repeat(1, num_time_steps, 1) / scale
            # Transformation matrices:
            stransmat = identity.clone()
            stransmat[:, :, :3, 3] = strans

            if joint_name == 'LeftHand':
                dtransmat = identity.clone()
                dtransmat[:, :, :3, 3] = pose_root
                
            
            xrot = joint_rot[..., 0]
            mycos = torch.cos(xrot)
            mysin = torch.sin(xrot)
            drotmat2 = identity.clone()
            drotmat2[..., 1, 1] = mycos
            drotmat2[..., 1, 2] = -mysin
            drotmat2[..., 2, 1] = mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            yrot = joint_rot[..., 1]
            mycos = torch.cos(yrot)
            mysin = torch.sin(yrot)
            drotmat2 = identity.clone()        
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 2] = mysin
            drotmat2[..., 2, 0] = -mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            zrot = joint_rot[..., 2]
            mycos = torch.cos(zrot)
            mysin = torch.sin(zrot)
            drotmat2 = identity.clone()
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 1] = -mysin
            drotmat2[..., 1, 0] = mysin
            drotmat2[..., 1, 1] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            joint_parent_name = self.bvh_lhand_parents[joint_name]
            if joint_parent_name != -1:  # Not hips
                parent_trtr = joint_trtr[joint_parent_name]  # Dictionary-based rewrite
                localtoworld = torch.einsum('btij, btjk -> btik', parent_trtr, stransmat)
            else:
                localtoworld = torch.einsum('btij, btjk -> btik', stransmat, dtransmat)
            
            trtr = torch.einsum('btij, btjk -> btik', localtoworld, drotmat)
            joint_trtr[joint_name] = trtr  # New dictionary-based approach

            # worldpos = localtoworld * ORIGIN  # worldpos should be a vec4
            worldpose[..., jt, :] = localtoworld[..., :3, 3]
        # joint.worldpos[t] = worldpos 
        return worldpose
    
    def forward_kinematics_rhand(self, pose_root, full_rot, offsets, scale = 1000., rotation_format='rotmat'):
        if rotation_format == 'rotmat': 
            if full_rot.shape[-3] != 22:
                full_rot = select_bvh_joints_rotmat(full_rot, self.bvh_joint_order, self.rh_fingers_only) 
            full_rot = batch_rotmat_to_euler(full_rot)
        elif rotation_format == '6d': 
            if full_rot.shape[-2] != 22:
                full_rot = select_bvh_joints_rot6d(full_rot, self.bvh_joint_order, self.rh_fingers_only) 
            full_rot = batch_6d_to_euler_tensor(full_rot)
        worldpose = torch.zeros_like(full_rot)
        joint_trtr = {
        'RightHand': [],
        'RightHandEnd': [],
        'RightHandPinky1': [],
        'RightHandPinky2': [],
        'RightHandPinky3': [],
        'RightHandPinky3End': [],
        'RightHandRing1': [],
        'RightHandRing2': [],
        'RightHandRing3': [],
        'RightHandRing3End': [],
        'RightHandMiddle1': [],
        'RightHandMiddle2': [],
        'RightHandMiddle3': [],
        'RightHandMiddle3End': [],
        'RightHandIndex1': [],
        'RightHandIndex2': [],
        'RightHandIndex3': [],
        'RightHandIndex3End': [],
        'RightHandThumb1': [],
        'RightHandThumb2': [],
        'RightHandThumb3': [],
        'RightHandThumb3End': [],
    }

        num_batches = full_rot.shape[0]
        num_time_steps = full_rot.shape[1]
        device = full_rot.device
        identity = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(num_batches, num_time_steps, 1, 1).to(device)
        for joint_name, jt in self.rh_fingers_only.items():
            jt = int(jt)
            drotmat = identity.clone()
            joint_rot = full_rot[..., jt, :]                                

            strans = torch.tensor(offsets[joint_name])[None].unsqueeze(1).repeat(1, num_time_steps, 1) / scale
            # Transformation matrices:
            stransmat = identity.clone()
            stransmat[:, :, :3, 3] = strans

            if joint_name == 'RightHand':
                dtransmat = identity.clone()
                dtransmat[:, :, :3, 3] = pose_root
                
            
            xrot = joint_rot[..., 0]
            mycos = torch.cos(xrot)
            mysin = torch.sin(xrot)
            drotmat2 = identity.clone()
            drotmat2[..., 1, 1] = mycos
            drotmat2[..., 1, 2] = -mysin
            drotmat2[..., 2, 1] = mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            yrot = joint_rot[..., 1]
            mycos = torch.cos(yrot)
            mysin = torch.sin(yrot)
            drotmat2 = identity.clone()        
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 2] = mysin
            drotmat2[..., 2, 0] = -mysin
            drotmat2[..., 2, 2] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            zrot = joint_rot[..., 2]
            mycos = torch.cos(zrot)
            mysin = torch.sin(zrot)
            drotmat2 = identity.clone()
            drotmat2[..., 0, 0] = mycos
            drotmat2[..., 0, 1] = -mysin
            drotmat2[..., 1, 0] = mysin
            drotmat2[..., 1, 1] = mycos
            drotmat = torch.einsum('btij, btjk -> btik', drotmat, drotmat2)

            joint_parent_name = self.bvh_rhand_parents[joint_name]
            if joint_parent_name != -1:  # Not right wrist
                parent_trtr = joint_trtr[joint_parent_name]  # Dictionary-based rewrite
                localtoworld = torch.einsum('btij, btjk -> btik', parent_trtr, stransmat)
            else:
                localtoworld = torch.einsum('btij, btjk -> btik', stransmat, dtransmat)
            
            trtr = torch.einsum('btij, btjk -> btik', localtoworld, drotmat)
            joint_trtr[joint_name] = trtr  # New dictionary-based approach

            # worldpos = localtoworld * ORIGIN  # worldpos should be a vec4
            worldpose[..., jt, :] = localtoworld[..., :3, 3]
        # joint.worldpos[t] = worldpos 
        return worldpose


def forward_kinematics_tensor(pose_root, full_rot, joint_order, joint_parents, offsets):
    T = full_rot.shape[0]
    worldpose = torch.zeros_like(full_rot)
    for ti in range(T):
        joint_trtr = {
        'Hips': [],
        'RightUpLeg': [],
        'RightLeg': [],
        'RightFoot': [],
        'RightToeBase': [],
        'RightToeBaseEnd': [],
        'LeftUpLeg': [],
        'LeftLeg': [],
        'LeftFoot': [],
        'LeftToeBase': [],
        'LeftToeBaseEnd': [],
        'Spine': [],
        'Spine1': [],
        'Spine2': [],
        'Spine3': [],
        'RightShoulder': [],
        'RightArm': [],
        'RightForeArm': [],
        'RightHand': [],
        'RightHandEnd': [],
        'RightHandPinky1': [],
        'RightHandPinky2': [],
        'RightHandPinky3': [],
        'RightHandPinky3End': [],
        'RightHandRing1': [],
        'RightHandRing2': [],
        'RightHandRing3': [],
        'RightHandRing3End': [],
        'RightHandMiddle1': [],
        'RightHandMiddle2': [],
        'RightHandMiddle3': [],
        'RightHandMiddle3End': [],
        'RightHandIndex1': [],
        'RightHandIndex2': [],
        'RightHandIndex3': [],
        'RightHandIndex3End': [],
        'RightHandThumb1': [],
        'RightHandThumb2': [],
        'RightHandThumb3': [],
        'RightHandThumb3End': [],
        'LeftShoulder': [],
        'LeftArm': [],
        'LeftForeArm': [],
        'LeftHand': [],
        'LeftHandEnd': [],
        'LeftHandPinky1': [],
        'LeftHandPinky2': [],
        'LeftHandPinky3': [],
        'LeftHandPinky3End': [],
        'LeftHandRing1': [],
        'LeftHandRing2': [],
        'LeftHandRing3': [],
        'LeftHandRing3End': [],
        'LeftHandMiddle1': [],
        'LeftHandMiddle2': [],
        'LeftHandMiddle3': [],
        'LeftHandMiddle3End': [],
        'LeftHandIndex1': [],
        'LeftHandIndex2': [],
        'LeftHandIndex3': [],
        'LeftHandIndex3End': [],
        'LeftHandThumb1': [],
        'LeftHandThumb2': [],
        'LeftHandThumb3': [],
        'LeftHandThumb3End': [],
        'Spine4': [],
        'Neck': [],
        'Head': [],
        'HeadEnd': []
    }
    
        for joint_name, jt in joint_order.items():
            jt = int(jt)
            drotmat = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
            joint_rot = full_rot[ti, jt]                                

            strans = torch.tensor([0., 0., 0.])
            strans[0] = offsets[joint_name][0]
            strans[1] = offsets[joint_name][1]
            strans[2] = offsets[joint_name][2]
            # Transformation matrices:
            stransmat = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
            stransmat[0, 3] = strans[0]
            stransmat[1, 3] = strans[1]
            stransmat[2, 3] = strans[2]

            if joint_name == 'Hips':
                dtransmat = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
                dtransmat[0, 3] = pose_root[ti, 0]
                dtransmat[1, 3] = pose_root[ti, 1]
                dtransmat[2, 3] = pose_root[ti, 2]
                
            
            xrot = joint_rot[0]
            mycos = cos(xrot)
            mysin = sin(xrot)
            drotmat2 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[1, 1] = mycos
            drotmat2[1, 2] = -mysin
            drotmat2[2, 1] = mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

            yrot = joint_rot[1]
            mycos = cos(yrot)
            mysin = sin(yrot)
            drotmat2 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 2] = mysin
            drotmat2[2, 0] = -mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

            zrot = joint_rot[2]
            mycos = cos(zrot)
            mysin = sin(zrot)
            drotmat2 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 1] = -mysin
            drotmat2[1, 0] = mysin
            drotmat2[1, 1] = mycos
            drotmat = np.dot(drotmat, drotmat2)

            joint_parent_name = joint_parents[joint_name]
            if joint_parent_name != -1:  # Not hips
                parent_trtr = joint_trtr[joint_parent_name]  # Dictionary-based rewrite
                localtoworld = np.dot(parent_trtr, stransmat)
            else:
                localtoworld = np.dot(stransmat, dtransmat)
            
            trtr = np.dot(localtoworld, drotmat)
            joint_trtr[joint_name] = trtr  # New dictionary-based approach

            # worldpos = localtoworld * ORIGIN  # worldpos should be a vec4
            worldpose[ti, jt] = torch.tensor([localtoworld[0, 3], localtoworld[1, 3],
                      localtoworld[2, 3]])
    # joint.worldpos[t] = worldpos 
    return worldpose

def batch_forward_kinematics(pose_root, full_rot, joint_order, joint_parents, offsets, scale=1000):
    batch_size = pose_root.shape[0]
    worldpose_ = []
    for b in range(batch_size):
        offset = {}
        for key in offsets.keys():
            offset[key] = offsets[key][b]/scale
        pos = forward_kinematics_tensor(pose_root[b], full_rot[b], joint_order, joint_parents, offset)
        worldpose_.append(pos.unsqueeze(0))
    return torch.cat(worldpose_, dim=0)

def load_files(root_path, seq, p):          #example to load the csv files and check
        file_basename = os.path.join(root_path, f'{seq:03d}', str(p))
        # path_2d = os.path.join(file_basename, 'nosh.mdd')
        # path_3d = os.path.join(file_basename, 'nosh_allCams.csv')
        path_canon_3d = os.path.join(file_basename, 'motion_worldpos.csv')
        path_dofs = os.path.join(file_basename, 'motion_rotations.csv')
        path_offsets = os.path.join(file_basename, 'motion_offsets.pkl')

        print(f"loading file {file_basename}")
        # data_2d = np.genfromtxt(path_2d, delimiter=' ', skip_header=1)
        # data_3d = np.genfromtxt(path_3d, delimiter=' ', skip_header=1) # n_c*n_f x n_dof
        canon_3d = np.genfromtxt(path_canon_3d, delimiter=',', skip_header=1) # n_c*n_f x n_dof
        dofs = np.genfromtxt(path_dofs, delimiter=',', skip_header=1) # n_c*n_f x n_dof
        n_frames = 2
        canon_3d = np.float32(canon_3d[:n_frames, 1:].reshape(n_frames, -1, 3))
        dofs = dofs[:n_frames, 1:].reshape(n_frames, -1, 3)
        with open(path_offsets, 'rb') as f:
            offset_dict = pickle.load(f)
        print(f"loading complete")
        return canon_3d, dofs, offset_dict

# if __name__ == "__main__":
#     #example to load a preprocessed csv files and check whether forward_kinematics is working
#     root_path = os.path.join('..', 'DATASETS', 'StudioDataset', 'preprocessed_skeleton')        
#     pos, rot, offset_dict = load_files(root_path, 3, 0)
#     root_pos = pos[:, 0]
#     skel_fk = InhouseStudioSkeleton()
#     fk_pose1 =  skel_fk.forward_kinematics(root_pos, rot, skel_fk.bvh_joint_order, skel_fk.bvh_joint_parents, offset_dict)
#     print(np.max(pos - fk_pose1))