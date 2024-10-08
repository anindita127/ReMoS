import cv2
import glob
import imageio
from PIL import Image
import numpy as np
import os

def img2gif(image_folder):
    seqs = glob.glob(image_folder + '/*.jpg')
    
    out_filename = image_folder.split('/')[-1]
    int_seq = [int(seqs[i].split('/')[-1].split('.')[0]) for i in range(len(seqs))]
    # int_seq = [int(seqs[i].split('/')[-1].split('.')[0].split('_')[-1]) for i in range(len(seqs))]
    index = sorted(range(len(int_seq)), key=lambda k: int_seq[k])
    all_imgs = [seqs[index[i]] for i in range(len(index))]
    gif_name = os.path.join(image_folder, out_filename+ '.gif')
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in all_imgs:
            image = imageio.imread(filename)
            writer.append_data(image)

def img2gif_compress(fp_in):
    x = 800
    y = 400
    gif_name = os.path.join(image_folder, image_folder.split('/')[-1]+'compress.gif')
    q = 40 # Quality
    seqs = glob.glob(fp_in + '/*.jpg')
    int_seq = [int(seqs[i].split('/')[-1].split('.')[0]) for i in range(len(seqs))]
    index = sorted(range(len(int_seq)), key=lambda k: int_seq[k])
    all_imgs = [seqs[index[i]] for i in range(len(index))]
    img, *imgs = [Image.open(f).resize((x,y),Image.ANTIALIAS) for f in all_imgs] 
    img.save(fp=gif_name, format='GIF', append_images=imgs,quality=q, 
            save_all=True, loop=0, optimize=True)


def img2video(image_folder, fps, img_type='png'):
    seqs = glob.glob(image_folder + '/*.'+ img_type)
    out_filename = image_folder.split('/')[-1]
    int_seq = [int(seqs[i].split('/')[-1].split('.')[0].split('_')[-1]) for i in range(len(seqs))]
    index = sorted(range(len(int_seq)), key=lambda k: int_seq[k])
    all_imgs = [seqs[index[i]] for i in range(len(index))]
    img_array = []
    video_name =os.path.join(image_folder, out_filename+ '.avi')
    for filename in all_imgs:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(video_name ,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
        