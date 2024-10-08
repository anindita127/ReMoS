import numpy as np
# import torch
import os 
import glob
import sys
sys.path.append('.')
sys.path.append('..')
import pickle
from src.tools.transformations import batch_euler_to_rotmat

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

class PreProcessor():
    def __init__(self, root_dir, fps=20, split='train'):
        self.root = root_dir
        self.framerate = fps
        self.root = os.path.join(root_dir, split)
        self.sequences = os.listdir(self.root)
        

        self.annot_dict = {
            'cam': [], 
            'seq': [], 
            'contacts': [],
            'pose_canon_1':[], 'pose_canon_2':[],
            'dofs_1': [], 'dofs_2': [],
            'rotmat_1': [], 'rotmat_2': [],
            'offsets_1': [], 'offsets_2': []
            }

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

        print("creating the annot file")
        self.collate_videos()
        self.save_annot(split)
     
    def detect_contact(self, motion1, motion2, thresh=120):
        

        contact_joints = ['Hand', 'HandEnd', 
                          'HandPinky1', 'HandPinky2', 'HandPinky3', 'HandPinky3End',
                          'HandRing1', 'HandRing2', 'HandRing3','HandRing3End',
                          'HandIndex1', 'HandIndex2', 'HandIndex3','HandIndex3End',
                          'HandMiddle1', 'HandMiddle2', 'HandMiddle3','HandMiddle3End',
                          'HandThumb1', 'HandThumb2', 'HandThumb3','HandThumb3End']

        n_frames = motion1.shape[0]
       
        assert motion1.shape == motion2.shape

        ## 0 : no contact, 1: rh-rh, 2: lh-lh, 3: lh-rh , 4: rh-lh
        contact = np.zeros((n_frames, 5))

        def dist(x, y):
            return np.sqrt(np.sum((x - y)**2))
        count = 0
        for i in range(n_frames):
            for s, sides in enumerate([['Right', 'Right'], ['Left', 'Left'], ['Left', 'Right'], ['Right', 'Left']]):
                for j, joint1 in enumerate(contact_joints):
                    if contact[i, s+1] == 1:
                        break
                    for k, joint2 in enumerate(contact_joints):
                        j1 = sides[0] + joint1
                        j2 = sides[1] + joint2
                        
                        idx1 = self.bvh_joint_order[j1]
                        idx2 = self.bvh_joint_order[j2]
                        
                        d = dist(motion1[i, idx1], motion2[i, idx2]) 
                        if d <= thresh:
                            contact[i, s+1] = 1
                            count += 1
                            break


        print(count)
        return contact[:, 1:]
    
    
    def use_frames(self, motion1, motion2, thresh=1000):
       
        t_frame = motion1.shape[0] 
        xx = np.tile(motion1, (1,69,1))
        yy = np.repeat(motion2, 69, axis=1)

        diff = xx - yy
        D = np.linalg.norm(diff, axis=-1)
        contact = (D <= thresh)*1
        sum_contact = np.sum(contact, axis=1)
        contact_frames = list(np.nonzero(sum_contact)[0])
        return contact, contact_frames

    def save_annot(self, split):
        save_path = makepath(os.path.join('data', 'Ninjutsu', split+'.pkl'), isfile=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.annot_dict, f)

        
    def load_files(self, seq, fname='0'):
        file_basename = os.path.join(self.root, str(seq))
        path_canon_3d = os.path.join(file_basename, fname+'_worldpos.csv')
        path_dofs = os.path.join(file_basename, fname+'_rotations.csv')
        path_offsets = os.path.join(file_basename, fname+'_offsets.pkl')

        print(f"loading file {file_basename}")
        canon_3d = np.genfromtxt(path_canon_3d, delimiter=',', skip_header=1) # n_c*n_f x n_dof
        dofs = np.genfromtxt(path_dofs, delimiter=',', skip_header=1) # n_c*n_f x n_dof
        with open(path_offsets, 'rb') as f:
            offset_dict = pickle.load(f)
        print(f"loading complete")

        n_frames = canon_3d.shape[0]
        canon_3d = np.float32(canon_3d[:, 1:].reshape(n_frames, -1, 3))
        dofs = dofs[:, 1:].reshape(n_frames, -1, 3)
        
        use_frames = list(np.rint(np.arange(0, n_frames-1, 25/self.framerate)))
        use_frames = [int(a) for a in use_frames]
        canon_3d = canon_3d[use_frames]
        dofs = np.float32(dofs[use_frames])
        print(canon_3d.shape)
        return 0, 0, canon_3d, dofs, offset_dict
        

    def collate_videos(self):
        self.annot_dict['bvh_joint_order'] = self.bvh_joint_order
        for i, seq in enumerate(self.sequences):
            _, _, canon_3d_1, dofs_1, offsets_1 = self.load_files(seq, '0')
            _, _, canon_3d_2, dofs_2, offsets_2 = self.load_files(seq, '1')
            if canon_3d_2.shape[0] < canon_3d_1.shape[0]:
                n_frames = canon_3d_2.shape[0]
            else:
                n_frames = canon_3d_1.shape[0]
            canon_3d_1 = canon_3d_1[:n_frames]
            canon_3d_2 = canon_3d_2[:n_frames]
            rotmat_1 = batch_euler_to_rotmat(dofs_1)
            rotmat_2 = batch_euler_to_rotmat(dofs_2)
            contacts, contact_frames = self.use_frames(canon_3d_1, canon_3d_2)
            n_frames_contact = len(contact_frames) 
            canon_3d_1 = canon_3d_1[contact_frames]
            canon_3d_2 = canon_3d_2[contact_frames]
            dofs_1 = dofs_1[contact_frames]
            dofs_2 = dofs_2[contact_frames]
            contacts = contacts[contact_frames].reshape(n_frames_contact, 69, 69)
            hand_contact = self.detect_contact(motion1=canon_3d_1, motion2=canon_3d_2)
            self.annot_dict['pose_canon_1'].extend(canon_3d_1 )
            self.annot_dict['pose_canon_2'].extend(canon_3d_2 ) 
            self.annot_dict['dofs_1'].extend(dofs_1 )
            self.annot_dict['dofs_2'].extend(dofs_2 )
            self.annot_dict['contacts'].extend(hand_contact)
            self.annot_dict['rotmat_1'].extend(rotmat_1 )
            self.annot_dict['rotmat_2'].extend(rotmat_2 )
            self.annot_dict['offsets_1'].extend([offsets_1 for i in range(0, n_frames_contact)])
            self.annot_dict['offsets_2'].extend([offsets_2 for i in range(0, n_frames_contact)])
            self.annot_dict['seq'].extend([seq for i in range(0, n_frames_contact)])
            
            
        self.annot_dict['pose_canon_1'] = np.array(self.annot_dict['pose_canon_1'])
        self.annot_dict['pose_canon_2'] = np.array(self.annot_dict['pose_canon_2'])
        self.annot_dict['dofs_1'] = np.array(self.annot_dict['dofs_1'])
        self.annot_dict['dofs_2'] = np.array(self.annot_dict['dofs_2'])
        self.annot_dict['rotmat_1'] = np.array(self.annot_dict['rotmat_1'])
        self.annot_dict['rotmat_2'] = np.array(self.annot_dict['rotmat_2'])
        self.annot_dict['contacts'] = np.array(self.annot_dict['contacts'])
        print(len(self.annot_dict.keys()))
        print(len(self.annot_dict['seq']))
        print(len(self.annot_dict['pose_canon_1']))



 
if __name__ == "__main__":
    root_path = os.path.join('..', 'DATASETS', 'ReMocap', 'Ninjutsu')
    fps = 10
    pp = PreProcessor(root_path, fps, 'train')
    pp = PreProcessor(root_path, fps, 'test')