
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import logging
from copy import copy
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
# import pytorch3d.transforms as t3d


LOGGER_DEFAULT_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |'
                  ' <level>{level: <8}</level> |'
                  ' <cyan>{name}</cyan>:<cyan>{function}</cyan>:'
                  '<cyan>{line}</cyan> - <level>{message}</level>')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype).to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc - loc[idxs])/(1/float(fps))
    return vel

def vel2acc(vel,fps):
    B = vel.shape[0]
    idxs = [0] + list(range(B - 1))
    acc = (vel - vel[idxs]) / (1 / float(fps))
    return acc

def loc2acc(loc,fps):
    vel = loc2vel(loc,fps)
    acc = vel2acc(vel,fps)
    return acc, vel


def d62rotmat(pose):
    pose = torch.tensor(pose)
    reshaped_input = pose.reshape(-1, 6)
    return t3d.rotation_6d_to_matrix(reshaped_input)

def rotmat2d6(pose):
    pose = torch.tensor(pose)
    return np.array(t3d.matrix_to_rotation_6d(pose))

def rotmat2d6_tensor(pose):
    pose = torch.tensor(pose)
    return torch.tensor(t3d.matrix_to_rotation_6d(pose))

def aa2rotmat(pose):
    pose = to_tensor(pose)
    return t3d.axis_angle_to_matrix(pose)
    
def rotmat2aa(pose):
    pose = to_tensor(pose)
    quat = t3d.matrix_to_quaternion(pose)
    return t3d.quaternion_to_axis_angle(quat)
    # reshaped_input = pose.reshape(-1, 3, 3)
    # quat = t3d.matrix_to_quaternion(reshaped_input)

def d62aa(pose):
    pose = to_tensor(pose)
    return rotmat2aa(d62rotmat(pose))

def aa2d6(pose):
    pose = to_tensor(pose)
    return rotmat2d6(aa2rotmat(pose))

def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats

def batch_euler_to_rotmat(bxyz, order='xyz', units='deg'):
    br = []
    for frame in range(bxyz.shape[0]):
        # rotmat = euler(bxyz[frame], order, units)
        r1 = Rotation.from_euler('xyz', np.array(bxyz[frame]), degrees=True)
        rotmat = r1.as_matrix()
        br.append(rotmat)
    return np.stack(br).astype(np.float32)

def batch_rotmat_to_euler(rotmat, order='ZYX'):
    
    # Convert to Euler angles and permute last dimension from ZYX to XYZ to match data order
    eu = t3d.matrix_to_euler_angles(rotmat, order)[..., [2, 1, 0]]
    return eu

def batch_euler_to_6d(bxyz, order='xyz', units='deg'):
    br = []
    for frame in range(bxyz.shape[0]):
        # rotmat = euler(bxyz[frame], order, units)
        r1 = Rotation.from_euler('xyz', np.array(bxyz[frame]), degrees=True)
        rotmat = r1.as_matrix()
        d6 = rotmat2d6(rotmat)
        br.append(d6)
    return np.stack(br).astype(np.float32)

def batch_6d_to_euler(bxyz, order='XYZ'):
    br = []

    for batch in range(bxyz.shape[0]):
        br_ = []
        for frame in range(bxyz.shape[1]):
            # rotmat = t3d.rotation_6d_to_matrix(bxyz[batch, frame])
            rotmat = d62rotmat(bxyz[batch, frame])
            r  =  Rotation.from_matrix(np.array(rotmat))
            eu = r.as_euler("xyz", degrees=True)
            br_.append(np.array(eu))
        br.append(np.stack(br_).astype(np.float32))
    return np.stack(br).astype(np.float32)


def batch_6d_to_euler_tensor(bxyz, order='ZYX'):
    rotmat = t3d.rotation_6d_to_matrix(bxyz)
    # Convert to Euler angles and permute last dimension from ZYX to XYZ to match data order
    eu = t3d.matrix_to_euler_angles(rotmat, order)[..., [2, 1, 0]]
    return eu


def rotate(points,R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def rotmul(rotmat,R):
    if rotmat.ndim>3:
        rotmat = to_tensor(rotmat).squeeze()
    if R.ndim>3:
        R = to_tensor(R).squeeze()
    rot = torch.matmul(rotmat, R)
    return rot


smplx_parents =[-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
                53]
def smplx_loc2glob(local_pose):

    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()

    for i in range(1,len(smplx_parents)):
        global_pose[:,i] = torch.matmul(global_pose[:, smplx_parents[i]], global_pose[:, i].clone())

    return global_pose.reshape(bs,-1,3,3)

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

if __name__ == "__main__":
    euler_angles = np.array([0.3, -0.5, 0.7], dtype=np.float32)
    euler2matrix = t3d.euler_angles_to_matrix(torch.from_numpy(euler_angles), 'XYZ')
    matrix2euler = t3d.matrix_to_euler_angles(euler2matrix, 'XYZ')
    w = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotmat = w.as_matrix()
    r  =  Rotation.from_matrix(np.array(euler2matrix))
    eu = r.as_euler("xyz", degrees=False)
    angrot = eul2rot(euler_angles)
    print()