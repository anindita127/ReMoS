import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('.')
sys.path.append('..')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from PIL import Image
from scipy import interpolate
from src.tools.utils import makepath
from src.tools.img_gif import img2video, img2gif

LEFT_HANDSIDE = list(range(19, 24))
RIGHT_HANDSIDE = list(range(45, 48))
LEFT_FOOTSIDE = list(range(1, 5))
RIGHT_FOOTSIDE = list(range(6, 10))

kinematic_chain_full = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 65], [65, 66], [66, 67], [67, 68],       #spine, neck and head
                    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],     # right leg
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],     # left leg
                    [14, 15], [15, 16], [16, 17], [17, 18], [18, 19],       # right arm
                    [14, 40], [40, 41], [41, 42], [42, 43], [43, 44],       # left arm
                    [19, 20], [20, 21], [21, 22], [22, 23],     # right pinky
                    [19, 24], [24, 25], [25, 26], [26, 27],     # right ring
                    [19, 28], [28, 29], [29, 30], [30, 31],     # right middle
                    [19, 32], [32, 33], [33, 34], [34, 35],     # right index
                    [18, 36], [36, 37], [37, 38], [38, 39],     # right thumb
                    [44, 45], [45, 46], [46, 47], [47, 48],     # left pinky
                    [44, 49], [49, 50], [50, 51], [51, 52],     # left ring
                    [44, 53], [53, 54], [54, 55], [55, 56],     # left middle
                    [44, 57], [57, 58], [58, 59], [59, 60],     # left index
                    [43, 61], [61, 62], [62, 63], [63, 64],     # left thumb
                    ]
kinematic_chain_reduced = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 43], [43, 44], [44, 45], [45, 46],       #spine, neck and head
                    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],     # right leg
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],     # left leg
                    [14, 15], [15, 16], [16, 17], [17, 18],        # right arm
                    [14, 29], [29, 30], [30, 31], [31, 32],        # left arm
                    [18, 19], [19, 20],    # right pinky
                    [18, 21], [21, 22],     # right ring
                    [18, 23], [23, 24],     # right middle
                    [18, 25], [25, 26],    # right index
                    [18, 27], [27, 28],     # right thumb
                    [32, 33], [33, 34],     # left pinky
                    [32, 35], [35, 36],  # left ring
                    [32, 37], [37, 38],      # left middle
                    [32, 39], [39, 40],     # left index
                    [32, 41], [41, 42],     # left thumb
                    ]

kinematic_chain_short = [  
                    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [0, 11], [11, 12], [12, 13], [13, 14],
                    [14, 15], [15, 16], [16, 17], [17, 18],
                    [14, 19], [19, 20], [20, 21], [21, 22], 
                    [14, 23], [23, 24], [24, 25], [25, 26]     
                    ]
kinematic_chain_old = [  
                    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [0, 11], [11, 12], [12, 13], [13, 14],
                    [12, 15], [15, 16], [16, 17], 
                    [12, 18], [18, 19], [19, 20], 
                    [17, 21], [21, 22], [22, 23], [23, 24], [22, 25], [25, 26], [22, 27],
                    [27, 28],  [22, 29], [29, 30], [22, 31], [31, 32],
                    [20, 33], [33, 34], [34, 35], [35, 36], [34, 37], [37, 38], [34, 39], [39, 40],
                    [34, 41], [41, 42], [34, 43], [43, 44]      
                    ]
def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes('RGBA', (w, h), buf.tostring())

def plot_contacts3D(pose1, pose2=None, gt_pose2=None, savepath=None, kinematic_chain = 'full', onlyone=False, gif=False):
    
    def plot_twoperson(pose1, pose2, i, kinematic_chain, savepath, gt_pose2=None): 
        fig = plt.figure()
        
        ax = plt.subplot(projection='3d')

        ax.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim3d([-1000, 2000])
        ax.set_zlim3d([-1000, 2000])
        ax.set_ylim3d([-1000, 2000])
        ax.axis('off')
        ax.view_init(elev=0, azim=0, roll=90)
        if kinematic_chain == 'full':
            KINEMATIC_CHAIN = kinematic_chain_full
        elif  kinematic_chain == 'no_fingers':
            KINEMATIC_CHAIN = kinematic_chain_short
        elif  kinematic_chain == 'reduced':
            KINEMATIC_CHAIN = kinematic_chain_reduced
        elif  kinematic_chain == 'old':
            KINEMATIC_CHAIN = kinematic_chain_old

        for limb in KINEMATIC_CHAIN:
            xs = [pose1[i, limb[0], 0], pose1[i, limb[1], 0]]
            ys = [pose1[i, limb[0], 1], pose1[i, limb[1], 1]]
            zs = [pose1[i, limb[0], 2], pose1[i, limb[1], 2]]
            # if limb[0] in LEFT_FOOTSIDE or limb[0] in LEFT_HANDSIDE:
            #     ax.plot(xs, ys, zs, 'darkred', linewidth=2.0)
            # else:    
                # ax.plot(xs, ys, zs, 'red', linewidth=2.0)
            ax.plot(xs, ys, zs, 'red', linewidth=2.0)

            xs_ = [pose2[i, limb[0], 0], pose2[i, limb[1], 0]]
            ys_ = [pose2[i, limb[0], 1], pose2[i, limb[1], 1]]
            zs_ = [pose2[i, limb[0], 2], pose2[i, limb[1], 2]]
            # if limb[0] in LEFT_FOOTSIDE or limb[0] in LEFT_HANDSIDE:
            #     ax.plot(xs_, ys_, zs_, 'darkblue', linewidth=2.0)
            # else:    
            #     ax.plot(xs_, ys_, zs_, 'blue', linewidth=2.0)
            ax.plot(xs_, ys_, zs_, 'blue', linewidth=2.0)
            if gt_pose2 is not None:
                gt_xs_ = [gt_pose2[i, limb[0], 0], gt_pose2[i, limb[1], 0]]
                gt_ys_ = [gt_pose2[i, limb[0], 1], gt_pose2[i, limb[1], 1]]
                gt_zs_ = [gt_pose2[i, limb[0], 2], gt_pose2[i, limb[1], 2]]
                ax.plot(gt_xs_, gt_ys_, gt_zs_, 'g', linewidth=1.0)
        # min_x = min(min(pose1[i, :, 2]), min(pose2[i, :, 2])) - 100
        # min_y = min(min(pose1[i, :, 0]), min(pose2[i, :, 0])) - 100
        # max_x = max(max(pose1[i, :, 2]), max(pose2[i, :, 2])) + 100
        # max_y = max(max(pose1[i, :, 0]), max(pose2[i, :, 0])) + 100
        # x_pl, y_pl = np.meshgrid(np.linspace(min_x, max_x, 10), np.linspace(min_y, max_y, 10))
        # foot_ground_contact_p1 =min(pose1[i, :, 1]) 
        # foot_ground_contact_2 =min(pose2[i, :, 1]) 
        # ground_plane = min(foot_ground_contact_p1, foot_ground_contact_2)
        # z_pl = torch.ones((10, 10)) * ground_plane
        # ax.plot_surface(x_pl, y_pl, z_pl, color= 'y', alpha=0.1)
        filename = makepath(os.path.join(savepath, str(i) +'.png'), isfile=True)
        plt.savefig(filename)
        plt.close()
        

    def plot_oneperson(pose2, i, kinematic_chain, savepath): 
        # fig = plt.figure()
        ax = plt.subplot(projection='3d')
        ax.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        ax.axis('off')
        ax.view_init(elev=0, azim=0, roll=0)
        if kinematic_chain == 'full':
            KINEMATIC_CHAIN = kinematic_chain_full
        elif  kinematic_chain == 'no_fingers':
            KINEMATIC_CHAIN = kinematic_chain_short

        for limb in KINEMATIC_CHAIN:
            ys_ = [pose2[i, limb[0], 0], pose2[i, limb[1], 0]]
            zs_ = [pose2[i, limb[0], 1], pose2[i, limb[1], 1]]
            xs_ = [pose2[i, limb[0], 2], pose2[i, limb[1], 2]]
            if limb[0] in LEFT_FOOTSIDE or limb[0] in LEFT_HANDSIDE:
                ax.plot(xs_, ys_, zs_, 'darkred', linewidth=3.0)
            else:    
                ax.plot(xs_, ys_, zs_, 'red', linewidth=3.0)  
        filename = makepath(os.path.join(savepath, str(i) +'.png'), isfile=True)
        plt.savefig(filename)
        # plt.pause(0.001)
        plt.close()

    T = pose1.shape[0]
    is_interpolate = 0
    if is_interpolate:
        T1 = 3*T
        p1_x_interp =np.zeros((T1, pose1.shape[1]))
        p1_y_interp =np.zeros((T1, pose1.shape[1]))
        p1_z_interp =np.zeros((T1, pose1.shape[1]))
        p2_x_interp =np.zeros((T1, pose2.shape[1]))
        p2_y_interp =np.zeros((T1, pose2.shape[1]))
        p2_z_interp =np.zeros((T1, pose2.shape[1]))
        
        x = np.linspace(0, T-1 ,T)
        x_new = np.linspace(0, T-1 ,T1)
        for v1 in range(0, pose1.shape[1]):
            p1_x = pose1[:, v1, 0]
            p1_y = pose1[:, v1, 1]
            p1_z = pose1[:, v1, 2]
            p2_x = pose2[:, v1, 0]
            p2_y = pose2[:, v1, 1]
            p2_z = pose2[:, v1, 2]
            f_p1x = interpolate.interp1d(x, p1_x, kind = 'linear')
            f_p1y = interpolate.interp1d(x, p1_y, kind = 'linear')
            f_p1z = interpolate.interp1d(x, p1_z, kind = 'linear')
            f_p2x = interpolate.interp1d(x, p2_x, kind = 'linear')
            f_p2y = interpolate.interp1d(x, p2_y, kind = 'linear')
            f_p2z = interpolate.interp1d(x, p2_z, kind = 'linear')
            p1_x_interp[:, v1] = f_p1x(x_new)
            p1_y_interp[:, v1] = f_p1y(x_new)
            p1_z_interp[:, v1] = f_p1z(x_new)
            p2_x_interp[:, v1] = f_p2x(x_new)
            p2_y_interp[:, v1] = f_p2y(x_new)
            p2_z_interp[:, v1] = f_p2z(x_new)
        p1_x_interp = torch.from_numpy(p1_x_interp).unsqueeze(2)
        p1_y_interp = torch.from_numpy(p1_y_interp).unsqueeze(2)
        p1_z_interp = torch.from_numpy(p1_z_interp).unsqueeze(2)
        p1_interp = torch.cat((p1_x_interp, p1_y_interp, p1_z_interp), dim=-1)
        p2_x_interp = torch.from_numpy(p2_x_interp).unsqueeze(2)
        p2_y_interp = torch.from_numpy(p2_y_interp).unsqueeze(2)
        p2_z_interp = torch.from_numpy(p2_z_interp).unsqueeze(2)
        p2_interp = torch.cat((p2_x_interp, p2_y_interp, p2_z_interp), dim=-1)
        # verts_all = torch.cat((torch.from_numpy(verts_all[0]).unsqueeze(0), p1_interp), dim=0)
        T = T1
        pose1 = p1_interp
        pose2 = p2_interp

 
    for i in range(pose1.shape[0]):
        if onlyone:
            plot_oneperson(pose1, i, kinematic_chain, savepath)
        else:
            plot_twoperson(pose1, pose2, i, kinematic_chain, savepath, gt_pose2)
    if gif: 
        img2gif(savepath)
    else:
        img2video(savepath, fps=20)
    
